import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.optim as optim

# from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
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
        loss = loss_cls + 1.0 * out["loss_pc"]
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
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss_cls",
        }


if __name__ == "__main__":
    fn_cbc = "/workspace/ehr_setpoint/cbc_cdmdeid_demo_100k.parquet"
    fn_demo = "/workspace/ehr_setpoint/demography_cdmdeid.parquet"
    fn_cond = "/workspace/ehr_setpoint/condition_occur_BI_CV_EM.parquet"
    fn_icd2snomed = "/workspace/ehr_setpoint/icd2snomed.parquet"
    df = pd.read_parquet(fn_cbc)
    df_demo = pd.read_parquet(fn_demo)
    df_cond = pd.read_parquet(fn_cond)
    df_icd2snomed = pd.read_parquet(fn_icd2snomed)
    
    qt = QuantileTransformer(
        n_quantiles=1000,  # more quantiles = smoother ranks
        output_distribution="uniform",
        subsample=50000,
    )

    visit_types = set(
        [
            "Outpatient Hospital",
            "Outpatient Visit",
        ]
    )
    df = df[df.VISIT_TYPE.isin(visit_types)]
    id2feature = [
        "3000905",
        "3004327",
        "3008342",
        "3011948",
        "3013429",
        "3013650",
        "3013869",
        "3020416",
        "3023314",
        "3024929",
        "3028615",
        "3033575",
        "3037511",
        "3043111",
        "3023599",
        "3019897",
        "3000963",
    ]

    df["age"] = np.round(df["age_days"] / 365).astype(int)
    df["age_quarter"] = np.round(df["age_days"] / 91.25).astype(int)

    df["norm_value"] = df.groupby("MEASUREMENT_CONCEPT_ID")[
        "VALUE_AS_NUMBER"
    ].transform(lambda x: qt.fit_transform(x.values.reshape(-1, 1))[:, 0])
    person_ids = set(df.PERSON_ID)

    df_cond = df_cond[df_cond.PERSON_ID.isin(person_ids)]
    d = df_icd2snomed[
        [
            c.startswith("CV") or c.startswith("BI") or c.startswith("EM")
            for c in df_icd2snomed.category
        ]
    ]
    df_merge = df_cond.merge(
        d[
            ["SNOMED_CONCEPT_ID", "SNOMED_NAME", "category", "phecode"]
        ].drop_duplicates(),
        right_on="SNOMED_CONCEPT_ID",
        left_on="CONDITION_CONCEPT_ID",
    )
    
    df_merge = df_merge.merge(df_demo, on="PERSON_ID")
    df_merge["CONDITION_START_DATE"] = pd.to_datetime(df_merge["CONDITION_START_DATE"])
    diff = df_merge.CONDITION_START_DATE - df_merge.BIRTH_DATETIME.dt.to_pydatetime()
    df_merge["age_days"] = [d.days for d in diff]
    df_merge = df_merge[df_merge.age_days >= 0]
    df_merge["age"] = np.round(df_merge["age_days"] / 365).astype(int)
    df_merge["age_quarter"] = np.round(df_merge["age_days"] / 91.25).astype(int)
    df_code = df_merge[["PERSON_ID", "age_quarter", "category"]].drop_duplicates()
    df = (
        df.groupby(["age_quarter", "PERSON_ID", "MEASUREMENT_CONCEPT_ID"])["norm_value"]
        .mean()
        .reset_index()
    )
    df = df[df.age_quarter > 72]
    pid2event_info = {
        p: {c: set(d.age_quarter) for c, d in d.groupby("category")}
        for p, d in df_code.groupby("PERSON_ID")
    }

    df_demo = df_demo[df_demo.PERSON_ID.isin(person_ids)]
    pid2demo = {}

    for p, d in df_demo.groupby("PERSON_ID"):
        rec = {}
        gender = d.GENDER_CONCEPT_NAME.iloc[0]
        rec[f"gender_{gender}"] = 0
        race = d.RACE_CONCEPT_NAME.iloc[0]
        rec[f"race_{race}"] = 0

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

            fea_rec = fea_dyn
            encounters.append({"time": t, "inputs": fea_input, "outputs": fea_rec})
            last_t = max(last_t, t)

        encounters = sorted(encounters, key=lambda x: x["time"])
        encounters_all.append(encounters)

        event_info = {}
        if p in pid2event_info:
            event_info = pid2event_info[p]
        eventtime_all.append(event_info)
        # labels_all.append([fn2label[p]])

    meas_vocabs = build_vocab(encounters_all)
    output_vocabs = build_vocab(encounters_all, key="outputs", add_cls_pad=False)
    id2phecode = sorted(set(df_code.category))
    random_state = 1
    train_inds, test_inds = train_test_split(
        np.arange(len(encounters_all)),
        train_size=0.7,
        random_state=random_state,
    )

    val_inds, test_ind = train_test_split(
        test_inds, train_size=0.5, random_state=random_state
    )

    train_encounters = [encounters_all[i] for i in train_inds]
    train_event_infos = [eventtime_all[i] for i in train_inds]
    val_encounters = [encounters_all[i] for i in val_inds]
    val_event_infos = [eventtime_all[i] for i in val_inds]
    test_encounters = [encounters_all[i] for i in test_inds]
    test_event_infos = [eventtime_all[i] for i in test_inds]

    train_dataset = EventTimeDataset(
        train_encounters, meas_vocabs, output_vocabs, train_event_infos, id2phecode
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=EventTimeDataset.collate_events,
        num_workers=16,
    )
    val_dataset = EventTimeDataset(
        val_encounters, meas_vocabs, output_vocabs, val_event_infos, id2phecode
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=EventTimeDataset.collate_events,
        num_workers=16,
    )
    test_dataset = EventTimeDataset(
        test_encounters, meas_vocabs, output_vocabs, test_event_infos, id2phecode
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=EventTimeDataset.collate_events,
        num_workers=16,
    )

    model = LitTemporalTransformer(
        L=128,
        d_model=64,
        nhead=8,
        dim_out=18,
        dim_cls=len(train_dataset.event_keys),
    )

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",  # or "cpu"
        devices=1,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_loader, val_dataloaders=val_loader)
    trainer.test(model, test_loader)
