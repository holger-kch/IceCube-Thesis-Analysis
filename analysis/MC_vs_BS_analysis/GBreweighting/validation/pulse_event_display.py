#!/usr/bin/env python3
"""Single-event pulse visualizations, inspired by fig. 6.14 in Debes' thesis.

For the first MC event and the first data event (stopped class) we produce:
  (A) a one-DOM 'spike' plot — charge vs dom_time as vertical lines for the
      DOM with most pulses in that event (MC vs data side by side);
  (B) an event display — all pulses in the event as spikes, one subplot for
      MC and one for data, colored by dom_z.

Saved as a separate PNG next to the pulse_level plots.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compare_weighted_mc_vs_data as full


def pick_pulse_rich_event(db: Path, event_nos: np.ndarray,
                          n_sample: int = 5000) -> int:
    """Among a random sample of candidate events, pick the one whose busiest
    DOM has the most pulses (tie-broken by total pulses in the event)."""
    rng = np.random.default_rng(full.SEED)
    sample = rng.choice(event_nos, size=min(n_sample, len(event_nos)),
                        replace=False).tolist()
    best = (-1, -1, None)  # (max_dom_pulses, total_pulses, event_no)
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        BATCH = 500
        for i in range(0, len(sample), BATCH):
            batch = sample[i:i + BATCH]
            ph = ",".join("?" * len(batch))
            sql = f"""
            SELECT event_no, MAX(n) AS max_dom_pulses, SUM(n) AS total
            FROM (
                SELECT event_no, COUNT(*) AS n
                FROM {full.PULSEMAP}
                WHERE event_no IN ({ph})
                GROUP BY event_no, dom_x, dom_y, dom_z
            )
            GROUP BY event_no
            """
            for ev, mdp, tot in c.execute(sql, batch):
                if (mdp, tot) > (best[0], best[1]):
                    best = (mdp, tot, ev)
    print(f"  picked event {best[2]}: busiest DOM has {best[0]} pulses, "
          f"{best[1]} total pulses")
    return int(best[2])


def pick_events(class_name: str) -> tuple[int, int]:
    mc_w, data_w = full.load_weights(class_name)
    print("  scanning MC for a pulse-rich event ...")
    mc_ev = pick_pulse_rich_event(full.MC_DB, mc_w.index.to_numpy())
    print("  scanning data for a pulse-rich event ...")
    dt_ev = pick_pulse_rich_event(full.DATA_DB, data_w.index.to_numpy())
    return mc_ev, dt_ev


def fetch_event_pulses(db: Path, event_no: int) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        return pd.read_sql_query(
            f"SELECT dom_time, charge, dom_x, dom_y, dom_z "
            f"FROM {full.PULSEMAP} WHERE event_no = ?",
            c, params=(event_no,),
        )


def spike_plot_one_dom(ax, df: pd.DataFrame, title: str) -> None:
    """Pick the DOM with most pulses in this event and plot charge vs time."""
    key = list(zip(df["dom_x"], df["dom_y"], df["dom_z"]))
    df = df.assign(_dom=key)
    counts = df["_dom"].value_counts()
    chosen = counts.index[0]
    sub = df[df["_dom"] == chosen].sort_values("dom_time")
    ax.vlines(sub["dom_time"], 0, sub["charge"], color="C0", lw=2.0)
    ax.scatter(sub["dom_time"], sub["charge"], color="C0", s=28, zorder=3)
    ax.axhline(0.3, color="red", ls="--", lw=1.0, label="charge cut = 0.3 PE")
    ax.set_xlabel("dom_time [ns]")
    ax.set_ylabel("charge [PE]")
    ax.set_title(f"{title}\nDOM (x={chosen[0]:.0f}, y={chosen[1]:.0f}, "
                 f"z={chosen[2]:.0f}) — {len(sub)} pulses")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def event_display(ax, df: pd.DataFrame, title: str) -> None:
    """All pulses in the event as spikes, colored by dom_z."""
    order = np.argsort(df["dom_time"].to_numpy())
    t = df["dom_time"].to_numpy()[order]
    q = df["charge"].to_numpy()[order]
    z = df["dom_z"].to_numpy()[order]
    cmap = plt.get_cmap("viridis")
    zmin, zmax = float(z.min()), float(z.max())
    zn = (z - zmin) / (zmax - zmin) if zmax > zmin else np.zeros_like(z)
    for ti, qi, zi in zip(t, q, zn):
        ax.vlines(ti, 0, qi, color=cmap(zi), lw=0.8, alpha=0.85)
    ax.set_xlabel("dom_time [ns]")
    ax.set_ylabel("charge [PE]")
    ax.set_title(f"{title}\n{len(df)} pulses, "
                 f"{df[['dom_x','dom_y','dom_z']].drop_duplicates().shape[0]} DOMs")
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=zmin, vmax=zmax))
    plt.colorbar(sm, ax=ax, label="dom_z [m]")
    ax.grid(alpha=0.3)


def main() -> None:
    class_name = "stopped"
    mc_ev, dt_ev = pick_events(class_name)
    print(f"first MC event: {mc_ev}    first data event: {dt_ev}")

    mc_df = fetch_event_pulses(full.MC_DB, mc_ev)
    dt_df = fetch_event_pulses(full.DATA_DB, dt_ev)
    print(f"  MC pulses: {len(mc_df)}   data pulses: {len(dt_df)}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    spike_plot_one_dom(axes[0, 0], mc_df, f"MC — event {mc_ev}")
    spike_plot_one_dom(axes[0, 1], dt_df, f"data — event {dt_ev}")
    event_display(axes[1, 0], mc_df, f"MC — event {mc_ev}")
    event_display(axes[1, 1], dt_df, f"data — event {dt_ev}")

    fig.suptitle(f"Single-event pulse visualization (class: {class_name}) — "
                 f"inspired by fig. 6.14 in Debes MSc thesis",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = full.PLOTS_DIR / f"event_display_{class_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
