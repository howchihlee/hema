import random

import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer
from torch.utils.data import Dataset


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


def drop_keep_two(lst, p=0.9):
    if len(lst) <= 2:
        return lst[:]

    ind = list(range(2, len(lst) + 1))[::-1]
    weights = [p**i for i in range(len(ind))]
    keep_n = random.choices(ind, weights=weights, k=1)[0]
    return lst[:keep_n]


class EventTimeDataset(Dataset):
    def __init__(
        self,
        encounters,
        measurement_vocab,
        rec_vocabs,
        event_infos=None,
        event_keys=None,
        max_measurements=128,
        p_drop_encounter=None,
        p_drop_meas=None,
    ):
        """
        events: list of dicts:
            - "time": float
            - "measurements": dict {measurement_name: value}
        """
        self.encounters = encounters
        self.p_drop_encounter = p_drop_encounter
        self.p_drop_meas = p_drop_meas

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

        # add cls token
        ids = [self.cls_token]
        vals = [0]

        for name, v in meas_dict.items():
            if name not in self.measurement_vocab:
                continue

            if self.p_drop_meas is not None:
                if random.random() > self.p_drop_meas:
                    continue

            mid = self.measurement_vocab[name]
            ids.append(mid)
            vals.append(float(v))

        return ids, vals

    def parse_events(self, event_info, t0, t1):
        labels = []
        for d in self.event_keys:
            label = 0
            if d in event_info:
                label = self.is_in_between(event_info[d], t0 - 4, t1 + 4)
            labels += [label]
        return {"labels": labels}

    def is_in_between(self, nums, t0, t1):
        return int(any(t0 <= n <= t1 for n in nums))

    def __getitem__(self, idx):
        encounters = self.encounters[idx]
        event_info = self.event_infos[idx]
        outs = []

        if self.p_drop_encounter is not None:
            encounters = drop_keep_two(encounters, self.p_drop_encounter)

        for encounter in encounters:
            meas_ids, meas_vals = self.encode_measurements(encounter["inputs"])
            output_rec = [0] * len(self.rec_vocabs)
            output_rec_mask = [False] * len(self.rec_vocabs)
            for o, v in encounter["inputs"].items():
                if o not in self.rec_vocabs:
                    continue
                output_rec[self.rec_vocabs[o]] = v
                output_rec_mask[self.rec_vocabs[o]] = True

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
                "outputs_rec": output_rec,
                "output_rec_mask": output_rec_mask,
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
        outputs_rec = []
        output_rec_mask = []
        # Find PAD token from any one sample
        pad_token = 0  # always 0 based on vocabulary construction

        for item in flattened_batch:
            ids = item["meas_ids"]
            vals = item["meas_vals"]
            L = len(ids)
            pad_len = max_len - L

            padded_ids.append(ids + [pad_token] * pad_len)
            padded_vals.append(vals + [0.0] * pad_len)
            pad_masks.append([False] * L + [True] * pad_len)
            outputs.append(item["outputs"])
            output_mask.append(item["output_mask"])
            outputs_rec.append(item["outputs_rec"])
            output_rec_mask.append(item["output_rec_mask"])

        outputs = torch.tensor(outputs, dtype=torch.float)
        output_mask = torch.tensor(output_mask, dtype=torch.bool)

        outputs_rec = torch.tensor(outputs_rec, dtype=torch.float)
        output_rec_mask = torch.tensor(output_rec_mask, dtype=torch.bool)

        padded_ids = torch.tensor(padded_ids, dtype=torch.long)
        padded_vals = torch.tensor(padded_vals, dtype=torch.float)
        pad_masks = torch.tensor(pad_masks, dtype=torch.bool)

        return {
            "sample_ids": sample_ids,  # (B,)
            "time_vals": times,  # (B,)
            "time_ids": time_ids,  # (B,)
            "meas_ids": padded_ids,  # (B, max_len)
            "meas_vals": padded_vals,  # (B, max_len)
            "pad_masks": pad_masks,  # (B, max_len)
            "outputs": outputs,
            "output_mask": output_mask,
            "outputs_rec": outputs_rec,
            "output_rec_mask": output_rec_mask,
            "batch_size": batch_size,
            **event_outputs,
        }
