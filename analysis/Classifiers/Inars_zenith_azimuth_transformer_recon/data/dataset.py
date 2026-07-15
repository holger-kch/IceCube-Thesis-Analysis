"""Dataset fuer Muon Events aus SQLite-Datenbank.

Laedt Events mit Pulsdaten (dom_x, dom_y, dom_z, dom_time, charge, ...)
und Truth-Labels (zenith, azimuth) fuer das Transformer-Training.

Unterstuetzt verschiedene DB-Formate:
- Mit string/dom_number/hlc (z.B. events.db)
- Ohne string/dom_number/hlc (z.B. Peters MuonGun-DB)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import sqlite3


TARGET_COLUMNS = ['zenith', 'azimuth']

# Mapping von Target-Name zu (SQL-Spalten, Verarbeitungsfunktion)
# Normalisierung fuer skalare Targets:
# - energy: log10(E) — range ~1.7..4.0, gut fuer LogCosh/MSE
# - position_z: (z + 575) / 500 — zentriert um 0, range ~-5..2.4
POS_Z_MEAN = -575.0
POS_Z_SCALE = 500.0

TARGET_CONFIGS = {
    "direction": {
        "columns": ["zenith", "azimuth"],
        "process": lambda row: angles_to_unit_vector(
            np.float32(row[0]), np.float32(row[1])
        ),
    },
    "energy": {
        "columns": ["energy"],
        "process": lambda row: np.array(
            [np.log10(np.float32(row[0]))], dtype=np.float32
        ),
    },
    "position_z": {
        "columns": ["position_z"],
        "process": lambda row: np.array(
            [(np.float32(row[0]) - POS_Z_MEAN) / POS_Z_SCALE], dtype=np.float32
        ),
    },
}


def angles_to_unit_vector(zenith, azimuth):
    """Konvertiert Zenith- und Azimut-Winkel in einen Einheitsvektor."""
    x = np.sin(zenith) * np.cos(azimuth)
    y = np.sin(zenith) * np.sin(azimuth)
    z = np.cos(zenith)
    return np.stack([x, y, z], axis=-1)


def detect_features(db_path, pulsemap="SplitInIcePulses"):
    """Erkennt automatisch welche Features in der DB verfuegbar sind.

    Returns:
        (feature_list, mode) wobei mode "sensor_id" oder "position" ist.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    cols = [row[1] for row in conn.execute(
        f"PRAGMA table_info({pulsemap})"
    ).fetchall()]
    conn.close()

    has_string = "string" in cols
    has_dom_number = "dom_number" in cols
    has_hlc = "hlc" in cols
    has_width = "width" in cols

    if has_string and has_dom_number and has_hlc:
        # Format mit sensor_id (z.B. events.db)
        features = ["dom_x", "dom_y", "dom_z", "dom_time", "charge",
                     "hlc", "string", "dom_number"]
        mode = "sensor_id"
    elif has_string and has_dom_number:
        # string/dom_number vorhanden, aber kein hlc
        third = "width" if has_width else "charge"
        features = ["dom_x", "dom_y", "dom_z", "dom_time", "charge",
                     third, "string", "dom_number"]
        mode = "sensor_id"
    else:
        # Kein string/dom_number -> Position-basierte Gruppierung
        third = "width" if has_width else "charge"
        features = ["dom_x", "dom_y", "dom_z", "dom_time", "charge", third]
        mode = "position"

    return features, mode


class MuonDataset(Dataset):
    """Dataset das Events aus einer SQLite-DB laedt.

    Args:
        db_path: Pfad zur SQLite-Datenbank
        features: Liste von Feature-Spalten (oder None fuer Auto-Detect)
        pulsemap: Name der Pulse-Tabelle
        selection: Optionale Liste von event_nos (z.B. aus einem Split)
        max_events: Maximale Anzahl Events
    """

    def __init__(self, db_path, features=None, pulsemap="SplitInIcePulses",
                 selection=None, max_events=None, target="direction"):
        self.db_path = db_path
        self.pulsemap = pulsemap
        self.target = target

        if target not in TARGET_CONFIGS:
            raise ValueError(f"Unbekanntes Target: {target}. "
                             f"Verfuegbar: {list(TARGET_CONFIGS.keys())}")
        self.target_config = TARGET_CONFIGS[target]

        # Features auto-detecten oder manuell setzen
        if features is None:
            self.features, self.mode = detect_features(db_path, pulsemap)
        else:
            self.features = features
            has_string = "string" in features
            has_dom_number = "dom_number" in features
            self.mode = "sensor_id" if (has_string and has_dom_number) else "position"

        self.feature_cols = ", ".join(self.features)

        # Event-Nummern laden
        if selection is not None:
            self.event_nos = list(selection)
        else:
            conn = sqlite3.connect(
                f"file:{db_path}?mode=ro&immutable=1", uri=True
            )
            query = "SELECT event_no FROM truth ORDER BY event_no"
            if max_events is not None:
                query += f" LIMIT {max_events}"
            self.event_nos = [
                row[0] for row in conn.execute(query).fetchall()
            ]
            conn.close()

        if max_events is not None and selection is not None:
            self.event_nos = self.event_nos[:max_events]

    def __len__(self):
        return len(self.event_nos)

    def __getitem__(self, idx):
        event_no = self.event_nos[idx]

        # Neue Connection pro Aufruf (wegen DataLoader-Worker)
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro&immutable=1", uri=True
        )

        pulse_rows = conn.execute(
            f"SELECT {self.feature_cols} FROM {self.pulsemap} "
            f"WHERE event_no = ?",
            (event_no,),
        ).fetchall()
        pulse_features = np.array(pulse_rows, dtype=np.float32)

        truth_cols = ", ".join(self.target_config["columns"])
        truth_row = conn.execute(
            f"SELECT {truth_cols} FROM truth WHERE event_no = ?",
            (event_no,),
        ).fetchone()
        conn.close()

        target = self.target_config["process"](truth_row)

        return {
            "pulse_features": torch.from_numpy(pulse_features),
            "target": torch.from_numpy(target),
            "event_no": torch.tensor(event_no, dtype=torch.long),
            "n_pulses": torch.tensor(pulse_features.shape[0]),
        }
