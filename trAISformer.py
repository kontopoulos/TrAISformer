#!/usr/bin/env python
# coding: utf-8
# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytorch implementation of TrAISformer---A generative transformer for
AIS trajectory prediction

https://arxiv.org/abs/2109.03958

"""
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import os
import sys
import json
import pickle
from tqdm import tqdm
import math
import logging
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

import models, trainers, datasets, utils
from config_trAISformer import Config

cf = Config()
TB_LOG = cf.tb_log
if TB_LOG:
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter()

# make deterministic
utils.set_seed(42)
torch.pi = torch.acos(torch.zeros(1)).item() * 2

if __name__ == "__main__":

    device = cf.device
    init_seqlen = cf.init_seqlen

    ## Logging
    # ===============================
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print('======= Create directory to store trained models: ' + cf.savedir)
    else:
        print('======= Directory to store trained models: ' + cf.savedir)
    utils.new_log(cf.savedir, "log")

    ## Data
    # ===============================
    moving_threshold = 0.05
    l_phases = ["train", "valid", "test"]
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    if "calib" in cf.eval_splits:
        l_phases.append("calib")
        l_pkl_filenames.append(cf.calibset_name)
    Data, aisdatasets, aisdls = {}, {}, {}
    for phase, filename in zip(l_phases, l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        print(f"Loading {datapath}...")
        with open(datapath, "rb") as f:
            l_pred_errors = pickle.load(f)
        for V in l_pred_errors:
            try:
                moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
            except:
                moving_idx = len(V["traj"]) - 1  # This track will be removed
            V["traj"] = V["traj"][moving_idx:, :]
        Data[phase] = [x for x in l_pred_errors if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
        print(len(l_pred_errors), len(Data[phase]))
        print(f"Length: {len(Data[phase])}")
        print("Creating pytorch dataset...")
        # Latter in this scipt, we will use inputs = x[:-1], targets = x[1:], hence
        # max_seqlen = cf.max_seqlen + 1.
        if cf.mode in ("pos_grad", "grad"):
            aisdatasets[phase] = datasets.AISDataset_grad(Data[phase],
                                                          max_seqlen=cf.max_seqlen + 1,
                                                          device=cf.device)
        else:
            aisdatasets[phase] = datasets.AISDataset(Data[phase],
                                                     max_seqlen=cf.max_seqlen + 1,
                                                     device=cf.device)
        if phase in ("test", "calib"):
            shuffle = False
        else:
            shuffle = True
        aisdls[phase] = DataLoader(aisdatasets[phase],
                                   batch_size=cf.batch_size,
                                   shuffle=shuffle)
    if getattr(cf, "legacy_lr_schedule", False):
        cf.final_tokens = 2 * len(aisdatasets["train"]) * cf.max_seqlen
    else:
        # cosine decay reaches its floor after max_epochs worth of real (non-padding) targets
        n_targets = sum(min(len(V["traj"]), cf.max_seqlen + 1) - 1 for V in Data["train"])
        cf.final_tokens = cf.max_epochs * n_targets

    ## Model
    # ===============================
    model = models.TrAISformer(cf, partition_model=None)

    ## Trainer
    # ===============================
    trainer = trainers.Trainer(
        model, aisdatasets["train"], aisdatasets["valid"], cf, savedir=cf.savedir, device=cf.device, aisdls=aisdls, INIT_SEQLEN=init_seqlen)

    ## Training
    # ===============================
    if cf.retrain:
        trainer.train()

    ## Evaluation
    # ===============================
    # Positions are de-normalised with the configured ROI and errors are in
    # nautical miles. Decision points are taken every eval_stride steps inside
    # each test track (only the first one if eval_stride is None): the model
    # sees the init_seqlen steps before the decision point and n_samples
    # futures are drawn. Metrics are reported over all decision points and,
    # for comparison, over the first decision point of each track only.
    # Reported per horizon:
    #   mean_nm      mean error of a single sampled future (expected error)
    #   best_nm      best-of-n_samples error (the metric used in the TrAISformer paper)
    #   point_nm     error of the sample-mean position (a point forecast)
    #   spread_nm    mean distance of the samples to their mean (spread proxy)
    #   spread_corr  Spearman correlation between spread and point error
    # This runs for every split in cf.eval_splits (e.g. "calib" and "test");
    # all sampled futures are saved to <split>_predictions.npz for the UQ step.
    model.load_state_dict(torch.load(cf.ckpt_path, map_location=cf.device))
    model.eval()

    horizons = cf.eval_horizons
    n_steps = max(horizons)
    lat_range, lon_range = cf.lat_max - cf.lat_min, cf.lon_max - cf.lon_min

    def to_deg(x):
        """Normalised (..., >=2) -> (..., 2) [lat, lon] in degrees."""
        return torch.stack((cf.lat_min + x[..., 0] * lat_range, cf.lon_min + x[..., 1] * lon_range), dim=-1)

    def haversine_nm(a, b):
        """Great-circle distance in nm between (..., 2) [lat, lon] tensors in degrees."""
        a, b = torch.deg2rad(a), torch.deg2rad(b)
        h = torch.sin((b[..., 0] - a[..., 0]) / 2) ** 2 \
            + torch.cos(a[..., 0]) * torch.cos(b[..., 0]) * torch.sin((b[..., 1] - a[..., 1]) / 2) ** 2
        return 2 * 3440.065 * torch.asin(torch.sqrt(h.clamp(0, 1)))

    def evaluate(split):
        """Sample futures from decision points of one split; save metrics and predictions."""
        # Decision points: index s means the model sees traj[s - init_seqlen:s]
        # and is scored on traj[s:s + n_steps].
        l_track, l_start = [], []
        for j, V in enumerate(Data[split]):
            last = len(V["traj"]) - n_steps
            if last < init_seqlen:
                continue
            starts = [init_seqlen] if cf.eval_stride is None else range(init_seqlen, last + 1, cf.eval_stride)
            l_track += [j] * len(starts)
            l_start += list(starts)
        track_id, start = np.array(l_track), np.array(l_start)
        X = torch.tensor(np.stack([np.minimum(Data[split][j]["traj"][s - init_seqlen:s + n_steps, :4], 0.9999)
                                   for j, s in zip(track_id, start)]), dtype=torch.float32)

        l_preds = []
        with torch.no_grad():
            for b in tqdm(range(0, len(X), cf.eval_batch)):
                seqs_init = X[b:b + cf.eval_batch, :init_seqlen].to(cf.device)
                bs = seqs_init.shape[0]
                preds = trainers.sample(model,
                                        seqs_init.repeat_interleave(cf.n_samples, dim=0),
                                        n_steps,
                                        temperature=1.0,
                                        sample=True,
                                        sample_mode=cf.sample_mode,
                                        r_vicinity=cf.r_vicinity,
                                        top_k=cf.top_k)
                preds = preds[:, init_seqlen:, :].reshape(bs, cf.n_samples, n_steps, 4)
                l_preds.append(to_deg(preds).cpu())

        preds = torch.cat(l_preds).double()            # (n_points, n_samples, n_steps, 2)
        truth = to_deg(X[:, init_seqlen:]).double()     # (n_points, n_steps, 2)
        hist = to_deg(X[:, :init_seqlen]).double()      # (n_points, init_seqlen, 2)

        errors = haversine_nm(truth[:, None], preds)       # (n_points, n_samples, n_steps)
        centre = preds.mean(dim=1)                         # (n_points, n_steps, 2)
        point_errors = haversine_nm(truth, centre)          # (n_points, n_steps)
        spread = haversine_nm(centre[:, None], preds).mean(dim=1)  # (n_points, n_steps)

        def spearman(a, b):
            ra = np.argsort(np.argsort(a)).astype(float)
            rb = np.argsort(np.argsort(b)).astype(float)
            return float(np.corrcoef(ra, rb)[0, 1])

        def summarise(sel, name):
            """Per-horizon metrics over the decision points selected by the boolean mask sel."""
            sel = torch.from_numpy(sel)
            out = {"n_points": int(sel.sum()), "horizons": {}}
            logging.info(f"[{split}: {name}] decision points: {out['n_points']}, samples per point: {cf.n_samples}")
            logging.info(f"{'horizon':>8} {'mean_nm':>8} {'best_nm':>8} {'point_nm':>9} {'median':>7} "
                         f"{'spread_nm':>10} {'spread_corr':>12}")
            for h in horizons:
                i = h - 1
                e, pe, sp = errors[sel, :, i], point_errors[sel, i], spread[sel, i]
                m = {"mean_nm": float(e.mean()),
                     "best_nm": float(e.min(dim=1).values.mean()),
                     "point_nm": float(pe.mean()),
                     "median_point_nm": float(pe.median()),
                     "spread_nm": float(sp.mean()),
                     "spread_corr": spearman(sp.numpy(), pe.numpy())}
                label = f"{h * cf.step_minutes} min"
                out["horizons"][label] = m
                logging.info(f"{label:>8} {m['mean_nm']:8.3f} {m['best_nm']:8.3f} {m['point_nm']:9.3f} "
                             f"{m['median_point_nm']:7.3f} {m['spread_nm']:10.3f} {m['spread_corr']:12.3f}")
            return out

        metrics = {"n_samples": cf.n_samples, "step_minutes": cf.step_minutes, "eval_stride": cf.eval_stride,
                   "n_tracks": int(len(np.unique(track_id))),
                   "all_points": summarise(np.ones(len(X), bool), "all decision points"),
                   "first_point": summarise(start == init_seqlen, "first decision point of each track")}

        with open(os.path.join(cf.savedir, f"{split}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        t_decision = np.array([Data[split][j]["traj"][s - 1, 4] for j, s in zip(track_id, start)])
        mmsi = np.array([Data[split][j]["mmsi"] for j in track_id])
        np.savez_compressed(os.path.join(cf.savedir, f"{split}_predictions.npz"),
                            preds=preds.float().numpy(), truth=truth.float().numpy(), hist=hist.float().numpy(),
                            hist_sog=(X[:, :init_seqlen, 2] * cf.sog_max).numpy(),
                            hist_cog=(X[:, :init_seqlen, 3] * 360.).numpy(),
                            track_id=track_id, start=start, mmsi=mmsi, t_decision=t_decision,
                            step_minutes=cf.step_minutes)
        return errors, point_errors, spread

    for split in cf.eval_splits:
        result = evaluate(split)
        if split == "test":
            errors, point_errors, spread = result

    ## Plot
    # ===============================
    v_minutes = np.arange(1, n_steps + 1) * cf.step_minutes
    plt.figure(figsize=(9, 6), dpi=150)
    plt.plot(v_minutes, errors.mean(dim=(0, 1)).numpy(), label="mean (single sample)")
    plt.plot(v_minutes, errors.min(dim=1).values.mean(dim=0).numpy(), label=f"best of {cf.n_samples}")
    plt.plot(v_minutes, point_errors.mean(dim=0).numpy(), label="sample-mean point")
    plt.plot(v_minutes, spread.mean(dim=0).numpy(), "--", label="spread")
    for h in horizons:
        plt.axvline(h * cf.step_minutes, color="grey", lw=0.5)
    plt.xlabel("Prediction horizon (minutes)")
    plt.ylabel("Error (nm)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(cf.savedir, "prediction_error.png"))
