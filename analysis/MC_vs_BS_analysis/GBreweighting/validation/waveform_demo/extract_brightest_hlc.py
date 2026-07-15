#!/usr/bin/env python3
"""Scan the first N frames and pick the brightest HLC launch (highest
fADC integral) — that's our best afterpulse candidate.

Run with the icetray env-shell:
  /cvmfs/.../metaprojects/icetray/v1.11.1/env-shell.sh \
      python extract_brightest_hlc.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

from icecube import icetray, dataio, dataclasses, WaveCalibrator  # noqa: F401
from icecube.icetray import I3Tray, I3Units

I3_FILE = Path("/lustre/hpc/icecube/janikh/MINIONS_DA_sample_2015_v5.i3.zst")
GCD_FILE = Path("/lustre/hpc/icecube/janikh/GeoCalibDetectorStatus_2015.57161_V0.i3.gz")
OUT = Path(__file__).resolve().parent / "brightest_hlc_waveform.pkl"

N_FRAMES_TO_SCAN = 5000
FE_IMPEDANCE_OHM = 50.0
E_CHARGE_C = 1.602176634e-19


def main():
    import math
    best = {"q_fadc_pe": -1.0}
    seen = 0

    def scan(frame):
        nonlocal seen, best
        seen += 1
        if seen > N_FRAMES_TO_SCAN:
            return
        if "InIceRawData" not in frame or "CalibratedWaveforms" not in frame:
            return
        cal_wfs = frame["CalibratedWaveforms"]
        rd = frame["InIceRawData"]
        cal = frame["I3Calibration"]
        det = frame["I3DetectorStatus"]

        # Quick HLC-only loop: for each DOM with HLC, integrate the FADC
        # calibrated waveform → PE.
        for om, launches in rd:
            hlc_launch = next((L for L in launches if L.lc_bit), None)
            if hlc_launch is None or om not in cal_wfs:
                continue
            dc = cal.dom_cal[om]
            ds = det.dom_status[om]
            hv = float(ds.pmt_hv) / I3Units.V
            if hv <= 0:
                continue
            gain = 10.0 ** (dc.hv_gain_fit.intercept
                            + dc.hv_gain_fit.slope * math.log10(hv))
            k = 1.0 / (FE_IMPEDANCE_OHM * gain * E_CHARGE_C)

            fadc = next((w for w in cal_wfs[om] if str(w.source) == "FADC"),
                        None)
            if fadc is None:
                continue
            V = [float(v) / I3Units.V for v in fadc.waveform]
            dt_s = float(fadc.bin_width) * 1e-9
            q_pe = sum(V) * dt_s * k
            if q_pe > best["q_fadc_pe"]:
                hdr = frame["I3EventHeader"]
                best = {
                    "q_fadc_pe": q_pe,
                    "om": (int(om.string), int(om.om), int(om.pmt)),
                    "pmt_gain": float(gain),
                    "pmt_hv_volts": hv,
                    "fe_impedance_ohm": FE_IMPEDANCE_OHM,
                    "pe_per_voltsecond": float(
                        1.0 / (FE_IMPEDANCE_OHM * gain * E_CHARGE_C)
                    ),
                    "event": {
                        "run_id": hdr.run_id,
                        "event_id": hdr.event_id,
                        "sub_event_id": hdr.sub_event_id,
                    },
                    "calibrated_waveforms": [
                        {
                            "source": str(w.source),
                            "channel": int(w.channel),
                            "time_ns": float(w.time),
                            "bin_width_ns": float(w.bin_width),
                            "samples_volt": [
                                float(v) / I3Units.V for v in w.waveform
                            ],
                        }
                        for w in cal_wfs[om]
                    ],
                    "hlc_launch_time_ns": float(hlc_launch.time),
                }

    tray = I3Tray()
    tray.Add("I3Reader", FilenameList=[str(GCD_FILE), str(I3_FILE)])
    def drop_existing(frame):
        for k in ("CalibratedWaveforms", "CalibrationErrata"):
            if k in frame:
                del frame[k]
    tray.Add(drop_existing, Streams=[icetray.I3Frame.DAQ])
    tray.Add("I3WaveCalibrator", Launches="InIceRawData",
             Waveforms="CalibratedWaveforms",
             WaveformRange="CalibratedWaveformRange_recalc")
    tray.Add(scan, Streams=[icetray.I3Frame.DAQ])

    def stop(frame):
        if seen >= N_FRAMES_TO_SCAN:
            tray.RequestSuspension()
    tray.Add(stop, Streams=[icetray.I3Frame.DAQ])
    tray.Execute()

    print(f"scanned {seen} frames")
    print(f"brightest HLC DOM: {best['om']}   "
          f"Q_fadc = {best['q_fadc_pe']:.2f} PE   "
          f"event run={best['event']['run_id']} id={best['event']['event_id']}")

    with open(OUT, "wb") as f:
        pickle.dump(best, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
