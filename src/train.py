import os

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics.functional import auroc

from data_utils import EventTimeDataset
from loss import nll_logistic_hazard
from models import TemporalTransformer


class LitTemporalTransformer(pl.LightningModule):
    def __init__(self, loss_weights, model_config):
        super().__init__()
        self.model = TemporalTransformer(**model_config)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.loss_weights = loss_weights
        self.val_preds = []
        self.val_targets = []

    def forward(self, batch):
        return self.model(batch)

    def _get_weight(self, key):
        weight = self.loss_weights.get(key)
        if isinstance(weight, (int, float)) and weight > 0:
            return float(weight)
        return None

    def compute_loss(self, out, batch):
        loss_cls = self.criterion(out["pred_cls"], batch["labels"])
        out["loss_cls"] = loss_cls
        loss = loss_cls

        # existing losses
        for key in ["loss_pc", "loss_rec"]:
            weight = self._get_weight(key)
            if weight is not None:
                loss = loss + weight * out[key]

        key = "loss_event"
        weight = self._get_weight(key)
        if weight is not None:
            T = out["pred_phi"].size(-1)
            loss_event = nll_logistic_hazard(
                out["pred_phi"].view(-1, T), batch["duration_inds"], batch["event_inds"]
            )
            loss = loss + weight * loss_event
            out["loss_event"] = loss_event
        out["loss"] = loss
        return out

    def training_step(self, batch, batch_idx):
        out = self(batch)
        out = self.compute_loss(out, batch)
        batch_size = batch["batch_size"]

        self.log(
            "train_loss",
            out["loss"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_loss_cls",
            out["loss_cls"],
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_loss_pc",
            out["loss_pc"],
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_loss_event",
            out["loss_event"],
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        out = self.compute_loss(out, batch)

        batch_size = batch["batch_size"]
        pred_cls = out["pred_cls"].detach().float()
        labels = batch["labels"].detach().long()

        self.val_preds.append(pred_cls)
        self.val_targets.append(labels)

        for key in ["loss", "loss_cls", "loss_pc"]:
            self.log(
                f"val_{key}",
                out[key],
                prog_bar=True,
                on_epoch=True,
                batch_size=batch_size,
            )

        return out["loss"]

    def on_validation_epoch_start(self):
        # clear buffers
        self.val_preds = []
        self.val_targets = []

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds, dim=0)
        targets = torch.cat(self.val_targets, dim=0)
        # compute AUROC
        val_auroc = auroc(
            preds,
            targets.int(),  # AUROC expects ints for labels
            task="multilabel",
            num_labels=targets.shape[1],
        )
        self.log("val_auroc", val_auroc, prog_bar=True, on_epoch=True)

    def on_test_epoch_start(self):
        # clear buffers
        self.test_preds = []
        self.test_targets = []

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds, dim=0)
        targets = torch.cat(self.test_targets, dim=0)
        # compute AUROC
        test_auroc = auroc(
            preds,
            targets.int(),  # AUROC expects ints for labels
            task="multilabel",
            num_labels=targets.shape[1],
        )
        self.log("test_auroc", test_auroc, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        out = self(batch)

        pred_cls = out["pred_cls"].detach().float()
        labels = batch["labels"].detach().long()

        self.test_preds.append(pred_cls)
        self.test_targets.append(labels)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss_cls",
        }


def build_encounters(df, pid2demo, pid2event_info):
    encounters_all = []
    eventtime_all = []

    # Organize rows by PERSON_ID and age_quarter
    pid_age2rows = {}
    for row in df.itertuples(index=False):
        pid = row.PERSON_ID
        age = row.age_quarter
        if pid not in pid_age2rows:
            pid_age2rows[pid] = {}
        if age not in pid_age2rows[pid]:
            pid_age2rows[pid][age] = []
        pid_age2rows[pid][age].append(row)

    for pid, age_dict in pid_age2rows.items():
        encounters = []
        for age in sorted(age_dict.keys()):
            rows = age_dict[age]

            # Dynamic features
            fea_dyn = {r.MEASUREMENT_CONCEPT_ID: r.norm_value for r in rows}
            fea_dyn["age"] = age / 400

            # Static features
            fea_static = pid2demo.get(pid, {})

            # Combine features
            fea_input = {**fea_dyn, **fea_static}
            fea_rec = fea_dyn

            encounters.append({"time": age, "inputs": fea_input, "outputs": fea_rec})

        encounters_all.append(encounters)
        eventtime_all.append(pid2event_info.get(pid, {}))

    return encounters_all, eventtime_all


def build_vocab(encounters, key="inputs", add_cls_pad=True):
    names = set()
    for es in encounters:
        for e in es:
            names.update(e[key].keys())
    if add_cls_pad:
        measurement_vocab = {name: i + 2 for i, name in enumerate(sorted(names))}
        measurement_vocab["<CLS>"] = 1
        measurement_vocab["<PAD>"] = 0
    else:
        measurement_vocab = {name: i for i, name in enumerate(sorted(names))}
    return measurement_vocab


def get_pid2event_info(df):
    pid2event_info = {}
    for row in df.itertuples(index=False):
        pid = row.PERSON_ID
        cat = row.category
        age = row.age_quarter
        if pid not in pid2event_info:
            pid2event_info[pid] = {}
        if cat not in pid2event_info[pid]:
            pid2event_info[pid][cat] = set()
        pid2event_info[pid][cat].add(age)
    return pid2event_info


def get_pid2demo(df):
    return {
        row.PERSON_ID: {
            f"gender_{row.GENDER_CONCEPT_NAME}": 0,
            f"race_{row.RACE_CONCEPT_NAME}": 0,
        }
        for row in df.itertuples(index=False)
    }


def build_loaders(
    df,
    df_demo,
    df_code,
    batch_size_train=64,
    batch_size_eval=64,
    train_size=0.7,
    random_state=1,
    num_workers=32,
    p_drop_encounter=0.9,
    p_drop_meas=0.9,
):
    """
    Build vocabularies, split data, create datasets, and return dataloaders.
    """

    person_ids = set(df.PERSON_ID)
    df_demo = df_demo.loc[df_demo.PERSON_ID.isin(person_ids)]
    df_code = df_code.loc[df_code.PERSON_ID.isin(person_ids)]

    pid2event_info = get_pid2event_info(df_code)
    pid2demo = get_pid2demo(df_demo)

    train_enc, train_evt = build_encounters(
        df[df.split == "train"], pid2demo, pid2event_info
    )
    val_enc, val_evt = build_encounters(df[df.split == "val"], pid2demo, pid2event_info)
    test_enc, test_evt = build_encounters(
        df[df.split == "test"], pid2demo, pid2event_info
    )
    # ----- Vocabs -----
    meas_vocabs = build_vocab(train_enc)
    output_vocabs = build_vocab(train_enc, key="outputs", add_cls_pad=False)
    _vocab = build_vocab(val_enc, key="outputs", add_cls_pad=False)
    diff = set(_vocab.keys()) - set(output_vocabs.keys())
    assert len(diff) == 0, f"{diff} not in train outputs"
    diff = set(_vocab.keys()) - set(output_vocabs.keys())
    assert len(diff) == 0, f"{diff} not in train outputs"
    # ----- Label mapping -----
    id2phecode = sorted([s for s in set(df_code.category) if s != "LAST_REC"])
    # ----- Datasets -----
    train_dataset = EventTimeDataset(
        train_enc,
        meas_vocabs,
        output_vocabs,
        train_evt,
        id2phecode,
        p_drop_encounter=p_drop_encounter,
        p_drop_meas=p_drop_meas,
    )
    val_dataset = EventTimeDataset(
        val_enc, meas_vocabs, output_vocabs, val_evt, id2phecode
    )
    test_dataset = EventTimeDataset(
        test_enc, meas_vocabs, output_vocabs, test_evt, id2phecode
    )

    # ----- Loaders -----
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=EventTimeDataset.collate_events,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        collate_fn=EventTimeDataset.collate_events,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        collate_fn=EventTimeDataset.collate_events,
        num_workers=num_workers,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "meas_vocabs": meas_vocabs,
        "output_vocabs": output_vocabs,
        "id2phecode": id2phecode,
    }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print("Loaded configuration:\n", OmegaConf.to_yaml(cfg))

    # Load datasets
    df = pd.read_parquet(cfg.data.df_path)
    df_demo = pd.read_parquet(cfg.data.df_demo_path)
    df_code = pd.read_parquet(cfg.data.df_code_path)

    datasets_info = build_loaders(
        df,
        df_demo,
        df_code,
        p_drop_meas=cfg.drop.p_drop_meas,
        p_drop_encounter=cfg.drop.p_drop_encounter,
    )

    train_loader = datasets_info["train_loader"]
    val_loader = datasets_info["val_loader"]
    test_loader = datasets_info["test_loader"]

    # model config
    model_cfg = cfg.model
    model_cfg.dim_cls = len(datasets_info["id2phecode"])
    model_cfg.dim_out = len(datasets_info["output_vocabs"])

    model = LitTemporalTransformer(
        loss_weights={"loss_event": 1, "loss_pc": 1, "loss_rec": 1},
        model_config=model_cfg,
    )

    os.makedirs(cfg.trainer.root_dir, exist_ok=True)
    ckpt_dir = f"{cfg.trainer.root_dir}/checkpoints"

    # Checkpoint callback
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor=cfg.trainer.monitor_metric,
        mode="max",
        save_top_k=1,
        save_last=False,
        filename=cfg.trainer.checkpoint_name,
    )

    # Trainer
    trainer = pl.Trainer(
        default_root_dir=cfg.trainer.root_dir,
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[checkpoint_cb],
    )

    trainer.fit(model, train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
