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


def pivot_tensor(tensor_to_pivot, sample_ids, time_ids, list_vals_to_fill):
    n = len(list_vals_to_fill)

    for i, val in enumerate(list_vals_to_fill):
        tensor_to_pivot[sample_ids, n * time_ids + i] = val
    return tensor_to_pivot


class TemporalTransformer(nn.Module):
    def __init__(self, n_meas=128, d_model=32, nhead=8, dim_out=29, dim_cls=101):
        super().__init__()

        self.n_meas = n_meas
        self.d_model = d_model

        self.emb_layer = nn.Embedding(n_meas, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            activation="gelu",
            batch_first=True,
        )
        self.encounter_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        self.pred_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.val_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.cls_emb = nn.Parameter(torch.randn(1, dim_cls, d_model))

        enc_layer_visit = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )

        self.traj_encoder = nn.TransformerEncoder(enc_layer_visit, num_layers=2)

        dim_tmax = 120
        self.f_obs = nn.Linear(d_model, dim_out)
        self.f_cls = nn.Linear(d_model, dim_tmax + 1)
        self.dim_out = dim_out
        self.dim_cls = dim_cls
        self.dim_tmax = dim_tmax

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
        meas_embs = self.encounter_encoder(
            meas_embs, src_key_padding_mask=batch["pad_masks"]
        )  # (N,M,d_model)

        device = meas_embs.device
        # --- Time encoding ---
        pos_emb = positional_encoding(batch["time_vals"], self.d_model)
        # pos_emb_t0 = torch.zeros((B, seq_time, self.d_model), device=device)
        # pos_emb_t0[batch['sample_ids'], batch['time_ids']] = pos_emb
        dims = (B, seq_time, self.d_model)
        pos_emb_t0 = torch.zeros(dims, device=device, dtype=torch.float)
        pos_emb_t0 = pivot_tensor(
            pos_emb_t0, batch["sample_ids"], batch["time_ids"], [pos_emb]
        )

        # --- Masking ---
        dims = (B, 2 * seq_time + self.dim_cls)
        mask_per_patient = torch.ones(dims, device=device, dtype=torch.bool)
        mask_per_patient = pivot_tensor(
            mask_per_patient,
            batch["sample_ids"],
            batch["time_ids"],
            [False, False],
        )
        mask_per_patient[:, -self.dim_cls :] = False

        # --- per patient embedding
        dims = (B, 2 * seq_time, self.d_model)
        out_per_patient = torch.zeros(dims, device=device, dtype=torch.float)
        out_per_patient = pivot_tensor(
            out_per_patient,
            batch["sample_ids"],
            batch["time_ids"],
            [meas_embs[:, 0, :], meas_embs[:, 0, :]],
        )

        # Add alternating positional encodings
        out_per_patient[:, ::2, :] += pos_emb_t0
        out_per_patient[:, 1:-1:2, :] += pos_emb_t0[:, 1:] + self.pred_emb
        # out_per_patient[batch_idx, last_real_idx, :] =  self.cls_emb

        # Output fold (ground truth)
        dims = (B, seq_time, self.dim_out)
        output_fold = torch.zeros(dims, device=device, dtype=torch.float)
        output_fold = pivot_tensor(
            output_fold, batch["sample_ids"], batch["time_ids"], [batch["outputs"]]
        )

        outputmask_fold = torch.zeros(dims, device=device, dtype=torch.bool)
        outputmask_fold = pivot_tensor(
            outputmask_fold,
            batch["sample_ids"],
            batch["time_ids"],
            [batch["output_mask"]],
        ).float()

        output_rec_fold = torch.zeros(dims, device=device, dtype=torch.float)
        output_rec_fold = pivot_tensor(
            output_rec_fold,
            batch["sample_ids"],
            batch["time_ids"],
            [batch["outputs_rec"]],
        )

        outputmask_rec_fold = torch.zeros(dims, device=device, dtype=torch.bool)
        outputmask_rec_fold = pivot_tensor(
            outputmask_rec_fold,
            batch["sample_ids"],
            batch["time_ids"],
            [batch["output_rec_mask"]],
        ).float()

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(
                2 * seq_time + self.dim_cls, 2 * seq_time + self.dim_cls, device=device
            ),
            diagonal=1,
        ).bool()
        causal_mask[-self.dim_cls :] = False

        cls_token = self.cls_emb.expand(B, -1, -1)
        out_per_patient_cls = torch.cat(
            [out_per_patient, cls_token], dim=1
        )  # [B, 2S + dim_cls, D]
        # cls_mask = torch.ones(B, self.dim_cls, dtype=mask_per_patient.dtype, device=device)
        # pad_mask_with_cls = torch.cat([mask_per_patient, cls_mask], dim=1)  # [B, S+1]

        out2 = self.traj_encoder(
            out_per_patient_cls,
            mask=causal_mask,
            src_key_padding_mask=mask_per_patient,
        )

        # Predictions
        pred = self.f_obs(out2[:, : -self.dim_cls])  # (B, 2*seq_time, dim_out)
        last_embs = out2[:, -self.dim_cls :]
        pred_event = self.f_cls(last_embs)  # (B, dim_cls, dim_tmax + 1)
        pred_cls = pred_event[:, :, 0]
        pred_phi = pred_event[:, :, 1:]

        outputs = {
            "pred_rec": pred,
            "pred_cls": pred_cls,
            "pred_phi": pred_phi,
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
            loss_pc = torch.sum(torch.abs(pred_slots - target_slots) * mask_slots)
            loss_pc = loss_pc.sum() / (mask_slots.sum() + 1e-8)
            outputs["loss_pc"] = loss_pc

            pred_slots = pred[:, :-1:2, :]  # (B, seq_time-1, dim_out)
            target_slots = output_rec_fold
            mask_slots = outputmask_rec_fold

            # L2 loss
            loss_rec = torch.sum(torch.abs(pred_slots - target_slots) * mask_slots)
            loss_rec = loss_rec.sum() / (mask_slots.sum() + 1e-8)
            outputs["loss_rec"] = loss_rec
        return outputs
