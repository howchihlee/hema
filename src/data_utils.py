import random

import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer
from torch.utils.data import Dataset


def create_pipeline():
    return Pipeline(
        [
            (
                "quantile",
                QuantileTransformer(
                    output_distribution="uniform", subsample=50000, random_state=1234
                ),
            ),
            ("shift", FunctionTransformer(lambda x: x * 2 - 1)),  # map [0,1] -> [-1,1]
        ]
    )


# Function to normalize each feature
def normalize_feature(group):
    qt = create_pipeline()
    vals = group["value"].values.reshape(-1, 1)
    group["value"] = qt.fit_transform(vals)
    return group


class EventTimeDataset(Dataset):
    def __init__(
        self,
        encounters,
        measurement_vocab,
        rec_vocabs,
        event_infos=None,
        event_keys=None,
        max_measurements=128,
    ):
        """
        events: list of dicts:
            - "time": float
            - "measurements": dict {measurement_name: value}
        """
        self.encounters = encounters

        if event_infos is not None:
            assert len(encounters) == len(event_infos)
            assert event_keys is not None
            self.event_infos = event_infos
            self.event_keys = event_keys

        self.max_measurements = max_measurements

        self.measurement_vocab = measurement_vocab
        self.rec_vocabs = rec_vocabs

        self.pad_token = self.measurement_vocab["<PAD>"]
        self.cls_token = self.measurement_vocab["<CLS>"]

    def __len__(self):
        return len(self.encounters)

    def encode_measurements(self, meas_dict):
        """
        Convert dict -> list of ids and list of values
        (no padding done here)
        """
        ids = [self.cls_token]
        vals = [0]

        for name, v in meas_dict.items():
            if name not in self.measurement_vocab:
                continue
            if random.random() > 0.9:
                continue
            mid = self.measurement_vocab[name]  # FIXED
            ids.append(mid)
            vals.append(float(v))

        return ids, vals

    def parse_events(self, event_info, t0, t1):
        labels = []
        for d in self.event_keys:
            label = 0
            if d in event_info:
                label = self.is_in_between(event_info[d], t0, t1)
            labels += [label]
        return {"labels": labels}

    def is_in_between(self, nums, t0, t1):
        return int(any(t0 <= n <= t1 for n in nums))

    def __getitem__(self, idx):
        encounters = self.encounters[idx]
        event_info = self.event_infos[idx]
        outs = []
        for encounter in encounters:
            meas_ids, meas_vals = self.encode_measurements(encounter["inputs"])
            output = [0] * len(self.rec_vocabs)
            output_mask = [False] * len(self.rec_vocabs)
            for o, v in encounter["outputs"].items():
                output[self.rec_vocabs[o]] = v
                output_mask[self.rec_vocabs[o]] = True

            out = {
                "time": encounter["time"],
                "meas_ids": meas_ids,
                "meas_vals": meas_vals,
                "outputs": output,
                "output_mask": output_mask,
            }
            outs.append(out)
        outs = sorted(outs, key=lambda x: x["time"])

        t0, t1 = outs[0]["time"], outs[-1]["time"]

        return {"inputs": outs, "outputs": self.parse_events(event_info, t0, t1)}

    @staticmethod
    def collate_events(batch):
        """
        batch is a list of samples returned by __getitem__.
        Performs padding and mask creation.
        """

        keys = batch[0]["outputs"].keys()
        event_outputs = {}
        for k in keys:
            event_outputs[k] = torch.tensor(
                [item["outputs"][k] for item in batch], dtype=torch.float
            )
        # event_inds = torch.tensor([item['event_ind'] for item in batch], dtype = torch.float)
        # event_times = torch.tensor([item['event_time'] for item in batch], dtype = torch.long)

        flattened_batch = [i for item in batch for i in item["inputs"]]
        sample_ids = [id for id, item in enumerate(batch) for i in item["inputs"]]
        batch_size = max(sample_ids) + 1
        sample_ids = torch.tensor(sample_ids, dtype=torch.long)  # (B,)
        times = torch.tensor(
            [item["time"] for item in flattened_batch], dtype=torch.float
        )  # (B,)
        time_ids = torch.tensor(
            [i for item in batch for i, _ in enumerate(item["inputs"])],
            dtype=torch.long,
        )  # (B,)

        # Find max measurement length in this batch
        max_len = max(len(item["meas_ids"]) for item in flattened_batch)

        padded_ids = []
        padded_vals = []
        pad_masks = []
        outputs = []
        output_mask = []
        # Find PAD token from any one sample
        pad_token = 0  # always 0 based on vocabulary construction

        for item in flattened_batch:
            ids = item["meas_ids"]
            vals = item["meas_vals"]
            L = len(ids)
            pad_len = max_len - L

            padded_ids.append(
                torch.tensor(ids + [pad_token] * pad_len, dtype=torch.long)
            )
            padded_vals.append(torch.tensor(vals + [0.0] * pad_len, dtype=torch.float))
            pad_masks.append(
                torch.tensor([True] * L + [False] * pad_len, dtype=torch.bool)
            )
            outputs.append(item["outputs"])
            output_mask.append(item["output_mask"])

        outputs = torch.tensor(outputs, dtype=torch.float)
        output_mask = torch.tensor(output_mask, dtype=torch.bool)

        return {
            "sample_ids": sample_ids,  # (B,)
            "time_vals": times,  # (B,)
            "time_ids": time_ids,  # (B,)
            "meas_ids": torch.stack(padded_ids),  # (B, max_len)
            "meas_vals": torch.stack(padded_vals),  # (B, max_len)
            "pad_masks": torch.stack(pad_masks),  # (B, max_len)
            "outputs": outputs,
            "output_mask": output_mask,
            "batch_size": batch_size,
            **event_outputs,
        }
