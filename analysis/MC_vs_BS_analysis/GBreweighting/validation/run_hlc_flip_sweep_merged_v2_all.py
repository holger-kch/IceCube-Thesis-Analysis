#!/usr/bin/env python3
"""HLC flip-rate sweep on merged-v2 parquet data.

The HLC transformer is run on MC SLC pulses from the merged-v2 files.  For a
given flip rate, the top-scoring SLC pulses are changed to HLC within each
selected class, then the weighted MC/data event-level hlc_frac distributions
are compared with a 1-Wasserstein distance.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.stats import wasserstein_distance


ROOT = Path("/groups/icecube/holgerkc/Thesis_Analysis")
GB_VAL_DIR = ROOT / "MC_vs_BS_analysis/GBreweighting/validation"
DATA_DIR = GB_VAL_DIR / "data_parquet_v2"
OUT_DIR = GB_VAL_DIR / "plots" / "transformer_hlcflip_study"
TRANSFORMER_DIR = GB_VAL_DIR / "transformer"
HLC_DIR = GB_VAL_DIR / "transformer_pulse_hlc"
DYNEDGE_HLC_DIR = GB_VAL_DIR / "dynedge_pulse_hlc"
CLASSES = ("stopped", "through")
FEATURES = ["charge", "dom_x", "dom_y", "dom_z", "dom_time", "rde", "pmt_area"]
FEATURES_FULL = FEATURES + ["hlc"]

RC_PARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "pgf.rcfonts": False,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}


def _norm_charge(x): return np.log10(np.maximum(x, 1e-9)).astype(np.float32)
def _norm_xyz(x): return (x / 500.0).astype(np.float32)
def _norm_time(x): return ((x - 1.0e4) / 3.0e4).astype(np.float32)
def _norm_rde(x): return ((x - 1.25) / 0.25).astype(np.float32)
def _norm_pmt(x): return (x / 0.05).astype(np.float32)


NORMALIZERS = {
    "charge": _norm_charge,
    "dom_x": _norm_xyz,
    "dom_y": _norm_xyz,
    "dom_z": _norm_xyz,
    "dom_time": _norm_time,
    "rde": _norm_rde,
    "pmt_area": _norm_pmt,
}


def parquet_path(source: str, cls: str) -> Path:
    return DATA_DIR / f"{source}_SplitInIcePulses_{cls}_merged_v2.parquet"


def weight_path(cls: str) -> Path:
    return DATA_DIR / f"GB_and_base_weights_{cls}_2M_v2.csv"


def load_weights(cls: str, source: str) -> pd.Series:
    df = pd.read_csv(weight_path(cls), usecols=["event_no", "source", "final_weight"])
    df = df[(df["source"] == source) & df["final_weight"].notna()]
    df = df[df["final_weight"] > 0]
    return df.set_index("event_no")["final_weight"].astype(np.float64)


def read_event_hlc_frac(source: str, cls: str, weights: pd.Series) -> pd.DataFrame:
    event_set = set(int(e) for e in weights.index)
    chunks = []
    path = parquet_path(source, cls)
    pf = pq.ParquetFile(path)
    print(f"[{source}/{cls}] aggregate {path.name}: {pf.num_row_groups} row groups",
          flush=True)
    for rg_idx in range(pf.num_row_groups):
        df = pf.read_row_group(rg_idx, columns=["event_no", "hlc"]).to_pandas()
        df = df[df["event_no"].isin(event_set)]
        if df.empty:
            continue
        agg = df.groupby("event_no", sort=False)["hlc"].agg(
            n_pulses="size", n_hlc="sum"
        )
        chunks.append(agg)
        if (rg_idx + 1) % 20 == 0 or rg_idx + 1 == pf.num_row_groups:
            print(f"  row groups {rg_idx + 1}/{pf.num_row_groups}", flush=True)
    out = pd.concat(chunks).groupby(level=0).sum()
    out["class"] = cls
    out["event_no"] = out.index.astype(np.int64)
    out["weight"] = weights.reindex(out.index).to_numpy(np.float64)
    out["hlc_frac"] = out["n_hlc"] / out["n_pulses"].clip(lower=1)
    out = out.reset_index(drop=True)
    print(f"[{source}/{cls}] {len(out):,} weighted events", flush=True)
    return out[["class", "event_no", "n_pulses", "n_hlc", "hlc_frac", "weight"]]


def iter_capped_events(path: Path, weights: pd.Series, max_pulses: int):
    cols = ["event_no", "hlc", *FEATURES]
    event_set = set(int(e) for e in weights.index)
    pf = pq.ParquetFile(path)
    carry = None
    for rg_idx in range(pf.num_row_groups):
        df = pf.read_row_group(rg_idx, columns=cols).to_pandas()
        df = df[df["event_no"].isin(event_set)]
        if carry is not None:
            df = pd.concat([carry, df], ignore_index=True)
            carry = None
        if df.empty:
            continue
        last_event = df["event_no"].iloc[-1]
        carry = df[df["event_no"] == last_event].copy()
        df = df[df["event_no"] != last_event]
        if not df.empty:
            df = (df.sort_values(["event_no", "charge"], ascending=[True, False])
                    .groupby("event_no", sort=False, as_index=False)
                    .head(max_pulses))
            for _, ev in df.groupby("event_no", sort=False):
                yield ev
        if (rg_idx + 1) % 20 == 0 or rg_idx + 1 == pf.num_row_groups:
            print(f"  scored stream row groups {rg_idx + 1}/{pf.num_row_groups}",
                  flush=True)
    if carry is not None and not carry.empty:
        carry = (carry.sort_values(["event_no", "charge"], ascending=[True, False])
                      .groupby("event_no", sort=False, as_index=False)
                      .head(max_pulses))
        for _, ev in carry.groupby("event_no", sort=False):
            yield ev


def normalise_features(df: pd.DataFrame) -> np.ndarray:
    arrs = []
    for col in FEATURES:
        arrs.append(NORMALIZERS[col](df[col].to_numpy()))
    return np.column_stack(arrs).astype(np.float32)


def reduce_top(rows: list[pd.DataFrame], keep_n: int) -> list[pd.DataFrame]:
    if not rows:
        return rows
    df = pd.concat(rows, ignore_index=True)
    if len(df) > keep_n:
        df = df.nlargest(keep_n, "hlc_score")
    return [df]


def flush_batch(model, device, events, rows, keep_n: int):
    if not events:
        return []
    lengths = [x.shape[0] for x, _, _ in events]
    feat_dim = events[0][0].shape[1]
    x = torch.zeros(len(events), max(lengths), feat_dim, dtype=torch.float32)
    mask = torch.zeros(len(events), max(lengths), dtype=torch.bool)
    hlc = []
    for i, (feat, h, _) in enumerate(events):
        n = feat.shape[0]
        x[i, :n] = torch.from_numpy(feat)
        mask[i, :n] = True
        hlc.append(h)
    batch = {"x": x.to(device), "mask": mask.to(device)}
    with torch.no_grad():
        scores = torch.sigmoid(model(batch)).cpu().numpy()
    out = []
    for i, (_, h, event_no) in enumerate(events):
        slc = h == 0
        if not np.any(slc):
            continue
        pos = np.flatnonzero(slc).astype(np.int64)
        out.append(pd.DataFrame({
            "event_no": int(event_no),
            "charge_rank": pos,
            "hlc_score": scores[i, pos].astype(np.float32),
        }))
    rows.extend(out)
    if len(rows) >= 32:
        rows[:] = reduce_top(rows, keep_n)
    return []


def count_capped_slc(cls: str, weights: pd.Series, max_pulses: int) -> int:
    n_slc = 0
    for ev in iter_capped_events(parquet_path("mc", cls), weights, max_pulses):
        n_slc += int((ev["hlc"].to_numpy() == 0).sum())
    return n_slc


def score_class(cls: str, weights: pd.Series, max_pulses: int, max_pct: float,
                batch_size: int, device: torch.device) -> tuple[pd.DataFrame, int]:
    sys.path.insert(0, str(TRANSFORMER_DIR))
    from train import PulseHLCModule  # noqa: E402

    ckpt = HLC_DIR / cls / "best.ckpt"
    metrics = json.loads((HLC_DIR / cls / "metrics.json").read_text())
    if metrics.get("features") != FEATURES:
        raise ValueError(f"{cls} HLC feature mismatch: {metrics.get('features')}")

    print(f"[{cls}] counting capped MC SLC candidates", flush=True)
    n_slc = count_capped_slc(cls, weights, max_pulses)
    keep_n = int(round(n_slc * max_pct / 100.0))
    print(f"[{cls}] capped MC SLC candidates: {n_slc:,}", flush=True)
    print(f"[{cls}] keeping top {keep_n:,} candidates", flush=True)

    model = PulseHLCModule.load_from_checkpoint(str(ckpt))
    model.eval().to(device)
    events = []
    rows = []
    t0 = time.time()
    print(f"[{cls}] scoring MC SLC candidates with {ckpt}", flush=True)
    for ev in iter_capped_events(parquet_path("mc", cls), weights, max_pulses):
        feat = normalise_features(ev)
        h = ev["hlc"].to_numpy(np.int8)
        events.append((feat, h, int(ev["event_no"].iloc[0])))
        if len(events) >= batch_size:
            events = flush_batch(model, device, events, rows, keep_n)
    flush_batch(model, device, events, rows, keep_n)
    rows = reduce_top(rows, keep_n)
    inv = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["event_no", "charge_rank", "hlc_score"]
    )
    inv["class"] = cls
    inv = inv.sort_values("hlc_score", ascending=False, kind="stable")
    print(f"[{cls}] scored {len(inv):,} SLC pulses in {time.time() - t0:.0f}s",
          flush=True)
    return inv[["class", "event_no", "charge_rank", "hlc_score"]], n_slc


class _HLCGraphInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
        self._positions = list(range(len(base._indices)))

    def __len__(self):
        return len(self._positions)

    def __getitem__(self, i):
        data = self.base[self._positions[i]]
        data.hlc = data.x[:, -1].clone().contiguous()
        data.x = data.x[:, :-1].contiguous()
        return data


def _rank_lookup(ds, event_no: int, max_pulses: int) -> dict[int, int]:
    start, end = ds._offsets[int(event_no)]
    charges = ds._feat_arr[start:end, FEATURES_FULL.index("charge")]
    order = np.argsort(-charges, kind="stable")[:max_pulses]
    return {int(start + local): rank for rank, local in enumerate(order)}


def score_class_dynedge(cls: str, weights: pd.Series, max_pulses: int,
                        max_pct: float, batch_size: int,
                        num_workers: int, device: torch.device
                        ) -> tuple[pd.DataFrame, int]:
    graphnet_src = "/groups/icecube/holgerkc/graphnet/src"
    if graphnet_src not in sys.path:
        sys.path.insert(0, graphnet_src)
    from graphnet.data.dataloader import DataLoader as GraphDataLoader  # noqa: E402
    from graphnet.models.data_representation import KNNGraph  # noqa: E402
    from graphnet.models.detector.icecube import IceCube86  # noqa: E402
    from mc_vs_data_parquet_dataset import MCvsDataParquetDataset  # noqa: E402
    from train_dynedge_pulse_hlc import DynEdgeHLCModule  # noqa: E402

    class V2Dataset(MCvsDataParquetDataset):
        PARQUET_DIR = DATA_DIR
        WEIGHT_FILE_TEMPLATE = str(DATA_DIR / "GB_and_base_weights_{cls}_2M_v2.csv")
        PULSE_FILE_TEMPLATE = "{source}_SplitInIcePulses_{cls}_merged_v2.parquet"

    data_repr = KNNGraph(detector=IceCube86(), input_feature_names=FEATURES_FULL)
    V2Dataset.clear_cache()
    ds = V2Dataset(
        path="unused", pulsemaps=["SplitInIcePulses"], features=FEATURES_FULL,
        truth=["is_data", "weight"], class_name=cls,
        max_events_per_source=10_000_000, data_representation=data_repr,
        loss_weight_table="truth", loss_weight_column="weight",
        source_filter="mc",
    )

    print(f"[{cls}/dynedge] counting capped MC SLC candidates", flush=True)
    n_slc = count_capped_slc(cls, weights, max_pulses)
    keep_n = int(round(n_slc * max_pct / 100.0))
    print(f"[{cls}/dynedge] capped MC SLC candidates: {n_slc:,}", flush=True)
    print(f"[{cls}/dynedge] keeping top {keep_n:,} candidates", flush=True)

    model = DynEdgeHLCModule(nb_inputs=len(FEATURES))
    model.load_state_dict(
        torch.load(DYNEDGE_HLC_DIR / cls / "state_dict.pth", map_location="cpu")
    )
    model.eval().to(device)
    loader = GraphDataLoader(
        _HLCGraphInferenceDataset(ds), batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    rows = []
    rank_cache = {}
    t0 = time.time()
    print(f"[{cls}/dynedge] scoring MC SLC candidates", flush=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            logits = model(batch).detach().cpu().numpy()
            scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
            was_hlc = batch.hlc.detach().cpu().numpy().astype(np.int8)
            ev_per_event = batch.event_no.view(-1).detach().cpu().numpy().astype(np.int64)
            ev_per_pulse = ev_per_event[batch.batch.detach().cpu().numpy()]
            row_idxs = []
            for ev in ev_per_event:
                start, end = ds._offsets[int(ev)]
                row_idxs.append(np.arange(start, end, dtype=np.int64))
            row_idxs = np.concatenate(row_idxs)
            cand_rows = []
            for row_idx, ev, score, h in zip(row_idxs, ev_per_pulse, scores, was_hlc):
                if h != 0:
                    continue
                ev = int(ev)
                if ev not in rank_cache:
                    rank_cache[ev] = _rank_lookup(ds, ev, max_pulses)
                rank = rank_cache[ev].get(int(row_idx))
                if rank is None:
                    continue
                cand_rows.append((ev, rank, float(score)))
            if cand_rows:
                rows.append(pd.DataFrame(
                    cand_rows, columns=["event_no", "charge_rank", "hlc_score"]
                ))
            if len(rows) >= 32:
                rows[:] = reduce_top(rows, keep_n)
            if batch_idx % 250 == 0:
                print(f"  dynedge batches {batch_idx:,}", flush=True)
    rows = reduce_top(rows, keep_n)
    inv = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["event_no", "charge_rank", "hlc_score"]
    )
    inv["class"] = cls
    inv = inv.sort_values("hlc_score", ascending=False, kind="stable")
    print(f"[{cls}/dynedge] scored {len(inv):,} top SLC pulses in "
          f"{time.time() - t0:.0f}s", flush=True)
    V2Dataset.clear_cache()
    return inv[["class", "event_no", "charge_rank", "hlc_score"]], n_slc


def w1(mc: pd.DataFrame, data: pd.DataFrame) -> float:
    return float(wasserstein_distance(
        mc["hlc_frac"].to_numpy(), data["hlc_frac"].to_numpy(),
        u_weights=mc["weight"].to_numpy(), v_weights=data["weight"].to_numpy(),
    ))


def apply_flips(mc_base: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    if inv.empty:
        return mc_base.copy()
    flips = inv.groupby(["class", "event_no"]).size().rename("n_flipped_hlc")
    out = mc_base.set_index(["class", "event_no"]).copy()
    out = out.join(flips, how="left")
    out["n_flipped_hlc"] = out["n_flipped_hlc"].fillna(0).astype(np.int64)
    out["n_hlc"] = (out["n_hlc"] + out["n_flipped_hlc"]).clip(
        upper=out["n_pulses"]
    )
    out["hlc_frac"] = out["n_hlc"] / out["n_pulses"].clip(lower=1)
    return out.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pct", type=float, default=10.0)
    parser.add_argument("--step-pct", type=float, default=0.5)
    parser.add_argument("--max-pulses", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--source", choices=["transformer", "dynedge"],
                        default="transformer")
    parser.add_argument("--classes", nargs="+", default=list(CLASSES),
                        choices=list(CLASSES))
    parser.add_argument("--no-latex", action="store_true")
    args = parser.parse_args()

    rc = dict(RC_PARAMS)
    if args.no_latex:
        rc["text.usetex"] = False
    matplotlib.rcParams.update(rc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    classes = tuple(args.classes)
    class_tag = "all" if len(classes) == len(CLASSES) else "_".join(classes)
    print(f"classes={classes}", flush=True)

    weights = {
        (cls, src): load_weights(cls, src)
        for cls in classes for src in ("mc", "data")
    }
    mc_base = pd.concat(
        [read_event_hlc_frac("mc", cls, weights[(cls, "mc")]) for cls in classes],
        ignore_index=True,
    )
    data = pd.concat(
        [read_event_hlc_frac("data", cls, weights[(cls, "data")]) for cls in classes],
        ignore_index=True,
    )

    invs = []
    n_slc_by_class = {}
    for cls in classes:
        if args.source == "transformer":
            inv, n_slc = score_class(
                cls, weights[(cls, "mc")], args.max_pulses, args.max_pct,
                args.batch_size, device,
            )
        else:
            inv, n_slc = score_class_dynedge(
                cls, weights[(cls, "mc")], args.max_pulses, args.max_pct,
                args.batch_size, args.num_workers, device,
            )
        invs.append(inv)
        n_slc_by_class[cls] = n_slc
        inv.to_csv(OUT_DIR / f"hlc_flip_inventory_merged_v2_{args.source}_{cls}.csv",
                   index=False)

    pcts = np.round(
        np.arange(0.0, args.max_pct + 0.5 * args.step_pct, args.step_pct), 6
    )
    rows = []
    base = w1(mc_base, data)
    inv_by_class = {inv["class"].iloc[0]: inv for inv in invs if len(inv)}
    for pct in pcts:
        pieces = []
        for cls, inv in inv_by_class.items():
            n = int(round(n_slc_by_class[cls] * pct / 100.0))
            pieces.append(inv.head(n))
        inv_pct = pd.concat(pieces, ignore_index=True) if pieces else invs[0].head(0)
        mc = apply_flips(mc_base, inv_pct)
        dist = w1(mc, data)
        rows.append({"pct": pct, "w1": dist, "n_flip": len(inv_pct)})
        print(f"flip {pct:>4.1f}% ({len(inv_pct):>10,} pulses) W1={dist:.6f}",
              flush=True)

    df = pd.DataFrame(rows)
    tag = f"0_to_{str(args.max_pct).replace('.', 'p')}_step{str(args.step_pct).replace('.', 'p')}"
    csv_path = OUT_DIR / f"hlc_flip_rate_sweep_merged_v2_{args.source}_{class_tag}_{tag}.csv"
    df.to_csv(csv_path, index=False)

    best_idx = df["w1"].idxmin()
    best_pct = float(df.loc[best_idx, "pct"])
    best_w1 = float(df.loc[best_idx, "w1"])
    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    label = "HLC transformer" if args.source == "transformer" else "dynedge-gnn HLC"
    ax.plot(df["pct"], df["w1"], marker="o", ms=5.5, lw=1.8, color="C0",
            label=label)
    ax.scatter([best_pct], [best_w1], s=125, marker="*", color="C0",
               edgecolor="k", zorder=5,
               label=rf"best: {best_pct:g}\% (W1={best_w1:.4f})")
    ax.axhline(base, color="0.4", lw=1.0, ls="--",
               label=f"no flip baseline (W1={base:.4f})")
    ax.set_xlabel(r"HLC SLC$\to$HLC flip rate [\%]")
    ax.set_ylabel(r"1-Wasserstein distance")
    title_cls = "all classes" if class_tag == "all" else class_tag
    ax.set_title(f"HLC flip-rate sweep: merged v2, {title_cls}, {args.source}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center")
    pdf_path = OUT_DIR / f"hlc_flip_rate_sweep_merged_v2_{args.source}_{class_tag}_{tag}.pdf"
    png_path = OUT_DIR / f"hlc_flip_rate_sweep_merged_v2_{args.source}_{class_tag}_{tag}.png"
    fig.savefig(pdf_path, format="pdf", pad_inches=0)
    fig.savefig(png_path, dpi=140, pad_inches=0)
    plt.close(fig)
    print(f"saved -> {csv_path}", flush=True)
    print(f"saved -> {pdf_path}", flush=True)
    print(f"saved -> {png_path}", flush=True)


if __name__ == "__main__":
    main()
