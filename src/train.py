import pandas as pd
import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics.functional import auroc

from src.data_utils import EventTimeDataset, build_vocab
from src.models import TemporalTransformer


class LitTemporalTransformer(pl.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        self.model = TemporalTransformer(**model_kwargs)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.val_preds = []
        self.val_targets = []

    def forward(self, batch):
        return self.model(batch)

    def compute_loss(self, out, batch):
        loss_cls = self.criterion(out["pred_cls"], batch["labels"])
        loss = loss_cls + 1.0 * out["loss_pc"] + 1.0 * out["loss_rec"]
        out["loss"] = loss
        out["loss_cls"] = loss_cls
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
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        out = self.compute_loss(out, batch)

        batch_size = batch["batch_size"]
        pred_cls = out["pred_cls"].detach().float()
        labels = batch["labels"].detach().long()

        self.val_preds.append(pred_cls)
        self.val_targets.append(labels)

        self.log(
            "val_loss_cls",
            out["loss_cls"],
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val_loss_pc",
            out["loss_pc"],
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

    for (p,), d in df.groupby(["PERSON_ID"]):
        encounters = []
        last_t = 0
        for t, k in d.groupby("age_quarter"):
            fea_dyn = dict(zip(k.MEASUREMENT_CONCEPT_ID, k.norm_value))
            fea_dyn["age"] = t / 400

            fea_static = {}
            if p in pid2demo:
                fea_static = pid2demo[p]

            fea_input = {**fea_dyn, **fea_static}
            # fea_input = fea_dyn
            fea_rec = fea_dyn
            encounters.append({"time": t, "inputs": fea_input, "outputs": fea_rec})
            last_t = max(last_t, t)

        encounters = sorted(encounters, key=lambda x: x["time"])
        encounters_all.append(encounters)

        event_info = {}
        if p in pid2event_info:
            event_info = pid2event_info[p]
        eventtime_all.append(event_info)

    return encounters_all, eventtime_all


def get_pid2event_info(df):
    grouped = df.groupby(["PERSON_ID", "category"])["age_quarter"].unique()
    pid2event_info = {}
    for (pid, cat), values in grouped.items():
        if pid not in pid2event_info:
            pid2event_info[pid] = {}
        pid2event_info[pid][cat] = set(values.tolist())
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
    batch_size_eval=16,
    train_size=0.7,
    random_state=1,
    num_workers=16,
    p_drop_encounter=0.9,
    p_drop_meas=0.9,
):
    """
    Build vocabularies, split data, create datasets, and return dataloaders.
    Minimizes input variables from the user.
    """
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
    id2phecode = sorted(set(df_code.category))

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


if __name__ == "__main__":
    df = pd.read_parquet("./train_cbc_v1.parquet")
    df_demo = pd.read_parquet("./train_demo_cbc_v1.parquet")
    df_code = pd.read_parquet("./train_cbc_outcome_v1.parquet")

    datasets_info = build_loaders(
        df,
        df_demo,
        df_code,
        p_drop_meas=0.8,
        p_drop_encounter=0.9,
    )
    train_loader = datasets_info["train_loader"]
    val_loader = datasets_info["val_loader"]
    test_loader = datasets_info["test_loader"]

    dim_cls = len(datasets_info["id2phecode"])
    dim_rec = len(datasets_info["output_vocabs"])

    model = LitTemporalTransformer(
        L=128,
        d_model=64,
        nhead=8,
        dim_out=dim_rec,
        dim_cls=dim_cls,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        save_last=False,
        filename="best-auroc",
    )

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",  # or "cpu"
        devices=1,
        callbacks=[checkpoint_cb],
        # gradient_clip_val=1.0,
    )

    trainer.fit(model, train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")
