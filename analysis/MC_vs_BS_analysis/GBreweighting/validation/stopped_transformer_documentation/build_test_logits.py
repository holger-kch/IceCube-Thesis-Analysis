#!/usr/bin/env python
"""Recover full-precision logits for the stopped/through test set.

The training script (train_stopped_transformer.py) saved test_results.csv with
the sigmoid *score* computed under AMP (float16), which rounds hard near 0 and 1
and never stores the raw logit. The standalone inference pass
(inference/run_inference.py) stored the raw `stopped_logit` (float32) for the
*same* events (event_no matches 1:1; sigmoid(logit) reproduces the rounded score
to within float16 precision). So we simply join the lost logits back in.

Output: results/stopped_transformer_2M/test_results_with_logits.csv with columns
event_no, stopped_label, stopped_logit, stopped_score, stopped_pred, osc_weight,
where stopped_score / stopped_pred are recomputed from the full-precision logit.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("/groups/icecube/holgerkc/Thesis_Analysis/ThroughOrStopped_muon/"
               "results/stopped_transformer_2M")
TEST_CSV = RESULTS / "test_results.csv"
INFER_CSV = Path("/groups/icecube/holgerkc/Thesis_Analysis/ThroughOrStopped_muon/"
                 "inference/output/"
                 "stopped_recon_mc_muons_1305k_130000_720k_139008_unmerged.csv")
OUT_CSV = RESULTS / "test_results_with_logits.csv"


def main():
    test = pd.read_csv(TEST_CSV)
    inf = pd.read_csv(INFER_CSV, usecols=["event_no", "stopped_logit"])

    merged = test.merge(inf, on="event_no", how="left")
    missing = int(merged["stopped_logit"].isna().sum())
    if missing:
        raise SystemExit(f"{missing} test events have no logit in inference CSV; "
                         "cannot recover full precision.")

    logit = merged["stopped_logit"].to_numpy(dtype=np.float64)
    score = 1.0 / (1.0 + np.exp(-logit))

    # sanity: recovered score must match the old (rounded) score
    drift = np.abs(score - merged["stopped_score"].to_numpy(float))
    print(f"recovered {len(merged):,} events; logit range "
          f"[{logit.min():.2f}, {logit.max():.2f}]")
    print(f"|recovered score - stored(rounded) score|: "
          f"median {np.median(drift):.2e}, max {drift.max():.2e}")

    out = pd.DataFrame({
        "event_no": merged["event_no"].to_numpy(),
        "stopped_label": merged["stopped_label"].to_numpy().astype(int),
        "stopped_logit": logit,
        "stopped_score": score,
        "stopped_pred": (logit > 0.0).astype(int),
        "osc_weight": merged["osc_weight"].to_numpy(),
    })
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
