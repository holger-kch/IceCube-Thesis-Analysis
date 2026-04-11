"""Collate-Funktionen fuer Batching von IceCube Muon Events.

Zwei Varianten:
- make_collate_flat: DOM-Gruppierung via sensor_id (braucht string + dom_number)
- make_collate_flat_by_position: DOM-Gruppierung via (dom_x, dom_y, dom_z)

Benutze make_collate_auto() um automatisch die richtige Variante zu waehlen.
"""

import torch
from typing import Callable, Dict, List

MAX_SENSOR_ID = 5160  # IceCube hat 5160 DOMs (86 Strings x 60 DOMs)


def _build_padded_output(
    dom_vectors: torch.Tensor,
    dom_event_idx: torch.Tensor,
    dom_counts_per_event: torch.Tensor,
    dom_event_starts: torch.Tensor,
    pulse_idx_in_dom: torch.Tensor,
    inverse_idx: torch.Tensor,
    all_features: torch.Tensor,
    batch_size: int,
    total_doms: int,
    max_doms: int,
    input_dim: int,
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Gemeinsame Padding-Logik fuer beide Collator-Varianten."""

    dom_idx_in_event = (
        torch.arange(total_doms, dtype=torch.long)
        - dom_event_starts[dom_event_idx]
    )

    # Falls ein Event mehr als max_doms DOMs hat: frueheste behalten
    needs_subsample = dom_counts_per_event > max_doms
    if needs_subsample.any():
        first_pulse_mask = pulse_idx_in_dom == 0
        dom_min_time = torch.full((total_doms,), float("inf"),
                                  dtype=all_features.dtype)
        dom_min_time[inverse_idx[first_pulse_mask]] = (
            all_features[first_pulse_mask, 3]  # dom_time ist immer Spalte 3
        )

        priority = -dom_min_time
        keep = torch.ones(total_doms, dtype=torch.bool)

        for ev in needs_subsample.nonzero(as_tuple=True)[0]:
            s = dom_event_starts[ev]
            e = dom_event_starts[ev + 1]
            _, top = priority[s:e].topk(max_doms, largest=True)
            keep[s:e] = False
            keep[s + top] = True

        kept_idx = keep.nonzero(as_tuple=True)[0]
        dom_vectors = dom_vectors[kept_idx]
        dom_event_idx = dom_event_idx[kept_idx]

        clamped = dom_counts_per_event.clamp(max=max_doms)
        kept_starts = torch.zeros(batch_size + 1, dtype=torch.long)
        kept_starts[1:] = clamped.cumsum(0)
        dom_idx_in_event = (
            torch.arange(dom_vectors.shape[0], dtype=torch.long)
            - kept_starts[dom_event_idx]
        )

    # Padded Tensor + Mask
    valid = dom_idx_in_event < max_doms
    ev_idx = dom_event_idx[valid]
    d_idx = dom_idx_in_event[valid]

    padded = torch.zeros(batch_size, max_doms, input_dim,
                         dtype=dom_vectors.dtype)
    mask = torch.zeros(batch_size, max_doms, dtype=torch.bool)

    padded[ev_idx, d_idx] = dom_vectors[valid]
    mask[ev_idx, d_idx] = True

    result = {
        "dom_vectors": padded,
        "padding_mask": mask,
        "event_ids": torch.stack([b["event_no"] for b in batch]),
        "batch_size": batch_size,
    }
    if "target" in batch[0]:
        result["targets"] = torch.stack([b["target"] for b in batch])

    return result


# ---------------------------------------------------------------------------
# Variante 1: Sensor-ID basiert (braucht string + dom_number Spalten)
# ---------------------------------------------------------------------------

def make_collate_flat(
    geometry,               # DOM-Positionen (sensor_id -> x, y, z)
    max_pulses_per_dom=16,
    max_doms=128,
) -> Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    """Collator mit DOM-Gruppierung via sensor_id = string*60 + dom_number.

    Erwartet pulse_features mit Spalten:
        [dom_x, dom_y, dom_z, dom_time, charge, hlc, string, dom_number]
         0      1      2      3         4       5    6       7

    DOM-Positionen kommen aus dem Geometry-Tensor (nicht aus den Pulses).
    """
    K = max_pulses_per_dom
    input_dim = 4 + 3 * K

    def collate_fn(batch):
        batch_size = len(batch)
        pulse_features_list = [event["pulse_features"] for event in batch]
        event_lengths = torch.tensor(
            [pf.shape[0] for pf in pulse_features_list], dtype=torch.long
        )
        all_features = torch.cat(pulse_features_list, dim=0)
        total_pulses = all_features.shape[0]

        # Sensor-ID aus string + dom_number
        strings = all_features[:, 6].long()
        dom_numbers = all_features[:, 7].long()
        sensor_ids = strings * 60 + dom_numbers

        pulse_event_idx = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long), event_lengths
        )

        # Pulses nach DOM gruppieren
        combined_key = pulse_event_idx * MAX_SENSOR_ID + sensor_ids
        unique_keys, inverse_idx, dom_counts = torch.unique(
            combined_key, return_inverse=True, return_counts=True, sorted=True
        )
        total_doms = unique_keys.shape[0]

        # Pulse-Index innerhalb jedes DOMs
        sort_order = torch.argsort(inverse_idx, stable=True)
        sorted_dom_idx = inverse_idx[sort_order]
        dom_starts = torch.zeros(total_doms + 1, dtype=torch.long)
        dom_starts[1:] = dom_counts.cumsum(0)
        pulse_idx_in_dom_sorted = (
            torch.arange(total_pulses, dtype=torch.long)
            - dom_starts[sorted_dom_idx]
        )
        pulse_idx_in_dom = torch.empty(total_pulses, dtype=torch.long)
        pulse_idx_in_dom[sort_order] = pulse_idx_in_dom_sorted

        # Erste K Pulses pro DOM behalten + normalisieren
        keep_mask = pulse_idx_in_dom < K
        kept_features = all_features[keep_mask]
        kept_dom_idx = inverse_idx[keep_mask]
        kept_pulse_idx = pulse_idx_in_dom[keep_mask]

        time_norm = (kept_features[:, 3] - 1e4) / 3e4
        charge_norm = torch.log10(
            kept_features[:, 4].clamp(min=1e-6)) / 3.0
        feat3_norm = kept_features[:, 5] - 0.5  # hlc: 0/1 -> -0.5/+0.5

        pulse_tensor = torch.zeros(total_doms, K, 3,
                                   dtype=all_features.dtype)
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 0] = time_norm
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 1] = charge_norm
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 2] = feat3_norm

        # DOM-Positionen aus Geometry
        dom_sensor_ids = unique_keys % MAX_SENSOR_ID
        dom_positions = geometry[dom_sensor_ids]

        n_pulses_norm = (
            torch.log1p(dom_counts.float()) / 3.0 - 1.0
        ).unsqueeze(1)

        dom_vectors = torch.cat([
            dom_positions,
            n_pulses_norm,
            pulse_tensor.reshape(total_doms, K * 3),
        ], dim=1)

        # Event-Zuordnung
        dom_event_idx = unique_keys // MAX_SENSOR_ID
        event_dom_counts = torch.bincount(
            dom_event_idx, minlength=batch_size
        )
        dom_event_starts = torch.zeros(batch_size + 1, dtype=torch.long)
        dom_event_starts[1:] = event_dom_counts.cumsum(0)

        return _build_padded_output(
            dom_vectors, dom_event_idx, event_dom_counts,
            dom_event_starts, pulse_idx_in_dom, inverse_idx,
            all_features, batch_size, total_doms, max_doms,
            input_dim, batch,
        )

    return collate_fn


# ---------------------------------------------------------------------------
# Variante 2: Position-basiert (braucht KEIN string/dom_number)
# ---------------------------------------------------------------------------

def make_collate_flat_by_position(
    max_pulses_per_dom=16,
    max_doms=128,
    pos_scale=(600.0, 600.0, 1250.0),
    pos_offset=(0.0, 0.0, 750.0),
) -> Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    """Collator mit DOM-Gruppierung via Position (dom_x, dom_y, dom_z).

    Erwartet pulse_features mit Spalten:
        [dom_x, dom_y, dom_z, dom_time, charge, third_feature]
         0      1      2      3         4       5

    third_feature kann hlc, width, oder was auch immer als 6. Spalte kommt.
    DOM-Positionen kommen direkt aus den Pulse-Daten (kein Geometry-Lookup).
    """
    K = max_pulses_per_dom
    input_dim = 4 + 3 * K

    def collate_fn(batch):
        batch_size = len(batch)
        pulse_features_list = [event["pulse_features"] for event in batch]
        event_lengths = torch.tensor(
            [pf.shape[0] for pf in pulse_features_list], dtype=torch.long
        )
        all_features = torch.cat(pulse_features_list, dim=0)
        total_pulses = all_features.shape[0]

        pulse_event_idx = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long), event_lengths
        )

        # DOM-ID aus Position: quantisiere auf 0.1m Genauigkeit und hashe
        # (DOMs am gleichen Ort haben exakt gleiche Koordinaten in der DB)
        qx = (all_features[:, 0] * 10).long()
        qy = (all_features[:, 1] * 10).long()
        qz = (all_features[:, 2] * 10).long()

        # Kombinierten Key: event_idx * grossen_offset + position_hash
        # Wir benutzen unique() auf dem kombinierten Tensor
        pos_keys = torch.stack([pulse_event_idx, qx, qy, qz], dim=1)

        # Eindeutige DOMs finden (pro Event)
        # Trick: konvertiere zu einen einzelnen Key via string-hash
        # Einfacher: nutze unique auf den rows
        unique_keys, inverse_idx, dom_counts = torch.unique(
            pos_keys, dim=0, return_inverse=True, return_counts=True,
            sorted=True,
        )
        total_doms = unique_keys.shape[0]

        # Pulse-Index innerhalb jedes DOMs
        sort_order = torch.argsort(inverse_idx, stable=True)
        sorted_dom_idx = inverse_idx[sort_order]
        dom_starts = torch.zeros(total_doms + 1, dtype=torch.long)
        dom_starts[1:] = dom_counts.cumsum(0)
        pulse_idx_in_dom_sorted = (
            torch.arange(total_pulses, dtype=torch.long)
            - dom_starts[sorted_dom_idx]
        )
        pulse_idx_in_dom = torch.empty(total_pulses, dtype=torch.long)
        pulse_idx_in_dom[sort_order] = pulse_idx_in_dom_sorted

        # Erste K Pulses behalten + normalisieren
        keep_mask = pulse_idx_in_dom < K
        kept_features = all_features[keep_mask]
        kept_dom_idx = inverse_idx[keep_mask]
        kept_pulse_idx = pulse_idx_in_dom[keep_mask]

        time_norm = (kept_features[:, 3] - 1e4) / 3e4
        charge_norm = torch.log10(
            kept_features[:, 4].clamp(min=1e-6)) / 3.0
        # 3. Feature: normalisieren (funktioniert fuer hlc UND width)
        # width: typisch 50-400ns -> (w - 200) / 200
        # hlc: 0/1 -> -0.5/+0.5
        feat3_norm = kept_features[:, 5]
        if feat3_norm.max() > 2.0:
            # Wahrscheinlich width (Werte >> 1)
            feat3_norm = (feat3_norm - 200.0) / 200.0
        else:
            # Wahrscheinlich hlc (0 oder 1)
            feat3_norm = feat3_norm - 0.5

        pulse_tensor = torch.zeros(total_doms, K, 3,
                                   dtype=all_features.dtype)
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 0] = time_norm
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 1] = charge_norm
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 2] = feat3_norm

        # DOM-Positionen aus den Pulse-Daten (erster Pulse pro DOM)
        first_pulse_of_dom = dom_starts[:total_doms]
        first_pulse_global = sort_order[first_pulse_of_dom]
        raw_positions = all_features[first_pulse_global, :3]

        # Normalisieren
        sx, sy, sz = pos_scale
        ox, oy, oz = pos_offset
        dom_positions = torch.stack([
            raw_positions[:, 0] / sx,
            raw_positions[:, 1] / sy,
            (raw_positions[:, 2] - oz) / sz,
        ], dim=1)

        n_pulses_norm = (
            torch.log1p(dom_counts.float()) / 3.0 - 1.0
        ).unsqueeze(1)

        dom_vectors = torch.cat([
            dom_positions,
            n_pulses_norm,
            pulse_tensor.reshape(total_doms, K * 3),
        ], dim=1)

        # Event-Zuordnung
        dom_event_idx = unique_keys[:, 0].long()
        event_dom_counts = torch.bincount(
            dom_event_idx, minlength=batch_size
        )
        dom_event_starts = torch.zeros(batch_size + 1, dtype=torch.long)
        dom_event_starts[1:] = event_dom_counts.cumsum(0)

        return _build_padded_output(
            dom_vectors, dom_event_idx, event_dom_counts,
            dom_event_starts, pulse_idx_in_dom, inverse_idx,
            all_features, batch_size, total_doms, max_doms,
            input_dim, batch,
        )

    return collate_fn


# ---------------------------------------------------------------------------
# Auto-Auswahl
# ---------------------------------------------------------------------------

def make_collate_auto(
    mode: str,
    geometry=None,
    max_pulses_per_dom: int = 16,
    max_doms: int = 128,
) -> Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    """Waehlt automatisch den richtigen Collator basierend auf dem Modus.

    Args:
        mode: "sensor_id" oder "position" (kommt von detect_features())
        geometry: Geometry-Tensor (nur fuer mode="sensor_id" noetig)
        max_pulses_per_dom: K Pulses pro DOM
        max_doms: Maximale DOMs pro Event
    """
    if mode == "sensor_id":
        if geometry is None:
            raise ValueError(
                "mode='sensor_id' braucht einen Geometry-Tensor"
            )
        return make_collate_flat(geometry, max_pulses_per_dom, max_doms)
    elif mode == "position":
        return make_collate_flat_by_position(max_pulses_per_dom, max_doms)
    else:
        raise ValueError(f"Unbekannter Modus: {mode}")
