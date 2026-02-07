import math

import torch
import torch.nn as nn


def positional_encoding(time_vals: torch.Tensor, d_model: int):
    """
    time_vals: (N,) long or float tensor of time values
    returns: (N, d_model) sinusoidal positional encoding
    """
    device = time_vals.device
    pe = torch.zeros(time_vals.size(0), d_model, device=device)
    position = time_vals.unsqueeze(1).float()  # (N,1)

    scale = -math.log(10000.0) / d_model
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * scale)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def pivot_tensor(
    dims, device, sample_ids, time_ids, list_vals_to_fill, dtype=torch.float32
):
    tensor_to_pivot = torch.zeros(dims, device=device, dtype=dtype)
    n = len(list_vals_to_fill)

    for i, val in enumerate(list_vals_to_fill):
        tensor_to_pivot[sample_ids, n * time_ids + i] = val
    return tensor_to_pivot


class TemporalTransformer(nn.Module):
    def __init__(self, L=128, d_model=32, nhead=8, dim_out=34, dim_cls=101):
        super().__init__()

        self.L = L
        self.d_model = d_model

        self.emb_layer = nn.Embedding(L, d_model)

        self.enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            activation="gelu",
            batch_first=True,
        )

        self.pred_emb = nn.Parameter(torch.randn(1, d_model))
        self.val_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.cls_emb = nn.Parameter(torch.randn(1, 1, d_model))

        self.enc_layer_visit = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )

        self.f_obs = nn.Linear(d_model, dim_out)
        self.f_cls = nn.Linear(d_model, dim_cls)
        self.dim_out = dim_out

    def forward(self, batch, return_outputmask=False, compute_pc_loss=True):
        """
        batch fields:
            batch["meas_ids"]     : (N, 1)
            batch["time_ids"]     : (N,)
            batch["time_vals"]    : (N,)
            batch["sample_ids"]   : (N,)  in [0,1]
            batch["outputs"]      : (N, dim_out)
            batch["output_mask"]  : (N, dim_out)
            batch["batch_size"]  : int
        """

        # Prepare seq length
        seq_time = batch["time_ids"].max().item() + 1
        B = batch["batch_size"]  # two patients

        # --- Embedding of measurements ---
        meas_embs = self.emb_layer(batch["meas_ids"])  # (N,M,d_model)
        meas_embs = meas_embs + batch["meas_vals"].unsqueeze(-1) * self.val_emb
        meas_embs = self.enc_layer(meas_embs)  # (N,M,d_model)

        device = meas_embs.device
        # --- Time encoding ---
        pos_emb = positional_encoding(batch["time_vals"], self.d_model)
        # pos_emb_t0 = torch.zeros((B, seq_time, self.d_model), device=device)
        # pos_emb_t0[batch['sample_ids'], batch['time_ids']] = pos_emb
        dims = (B, seq_time, self.d_model)
        pos_emb_t0 = pivot_tensor(
            dims, device, batch["sample_ids"], batch["time_ids"], [pos_emb]
        )

        dims = (B, 2 * seq_time, self.d_model)
        out_per_patient = pivot_tensor(
            dims,
            device,
            batch["sample_ids"],
            batch["time_ids"],
            [meas_embs[:, 0, :], meas_embs[:, 0, :]],
        )

        # Insert event embeddings to even/odd positions
        # out_per_patient[batch['sample_ids'], 2 * batch['time_ids']] = meas_embs[:, 0, :]
        # out_per_patient[batch['sample_ids'], 2 * batch['time_ids'] + 1] = meas_embs[:, 0, :]

        # Add alternating positional encodings
        out_per_patient[:, ::2, :] += pos_emb_t0
        out_per_patient[:, 1:-1:2, :] += pos_emb_t0[:, 1:] + self.pred_emb
        out_per_patient[:, [-1], :] += pos_emb_t0[:, [-1]] + self.cls_emb

        # --- Masking ---
        dims = (B, 2 * seq_time)
        mask_per_patient = pivot_tensor(
            dims,
            device,
            batch["sample_ids"],
            batch["time_ids"],
            [False, False],
            dtype=torch.bool,
        )
        # mask_per_patient[batch['sample_ids'], 2 * batch['time_ids']] = False
        # mask_per_patient[batch['sample_ids'], 2 * batch['time_ids'] + 1] = False

        # Output fold (ground truth)
        dims = (B, seq_time, self.dim_out)
        # output_fold[batch['sample_ids'], batch['time_ids']] = batch["outputs"]
        output_fold = pivot_tensor(
            dims, device, batch["sample_ids"], batch["time_ids"], [batch["outputs"]]
        )

        # outputmask_fold = torch.zeros((B, seq_time, self.dim_out), dtype=torch.bool, device=device)
        # outputmask_fold[batch['sample_ids'], batch['time_ids']] = batch["output_mask"]
        outputmask_fold = pivot_tensor(
            dims,
            device,
            batch["sample_ids"],
            batch["time_ids"],
            [batch["output_mask"]],
            dtype=torch.bool,
        )
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(2 * seq_time, 2 * seq_time, device=device), diagonal=1
        ).bool()

        out2 = self.enc_layer_visit(
            out_per_patient,
            src_mask=causal_mask,
            src_key_padding_mask=mask_per_patient,
            is_causal=True,
        )

        # Predictions
        pred = self.f_obs(out2)  # (B, 2*seq_time, dim_out)
        pred_cls = self.f_cls(out2[:, -1])

        outputs = {
            "pred_rec": pred,
            "pred_cls": pred_cls,
            "embeddings": out2,
        }

        if return_outputmask:
            outputs["outputmask_fold"] = outputmask_fold
            outputs["mask_per_patient"] = mask_per_patient

        if compute_pc_loss:
            pred_slots = pred[:, 1:-1:2, :]  # (B, seq_time-1, dim_out)
            target_slots = output_fold[:, 1:, :]
            mask_slots = outputmask_fold[:, 1:, :]

            # L2 loss
            loss_pc = torch.sum(((pred_slots - target_slots) ** 2) * mask_slots)
            loss_pc = loss_pc.sum() / (mask_slots.sum() + 1e-8)
            outputs["loss_pc"] = loss_pc

        return outputs
