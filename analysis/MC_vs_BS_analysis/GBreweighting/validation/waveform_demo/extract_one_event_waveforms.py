#!/usr/bin/env python3
"""Extract ATWD + fADC waveforms (raw ADC AND calibrated voltage) for one
HLC DOM, and the chargestamp for one SLC DOM in the same event.

Runs `I3WaveCalibrator` so we get the calibrated waveform in volts in
addition to the raw ADC counts.

Run with the icetray env-shell:
  /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/metaprojects/icetray/v1.11.1/env-shell.sh \
      python extract_one_event_waveforms.py

Output: one_event_waveforms.pkl  (loadable from any vanilla python with pickle)
"""
from __future__ import annotations

import pickle
from pathlib import Path

from icecube import icetray, dataio, dataclasses, WaveCalibrator  # noqa: F401
from I3Tray import I3Tray

I3_FILE = Path(
    "/lustre/hpc/icecube/janikh/MINIONS_DA_sample_2015_v5.i3.zst"
)
GCD_FILE = Path(
    "/lustre/hpc/icecube/janikh/GeoCalibDetectorStatus_2015.57161_V0.i3.gz"
)
OUT = Path(__file__).resolve().parent / "one_event_waveforms.pkl"

# Standard IC86 in-ice DOM sampling rates (override via GCD if needed):
ATWD_BIN_NS = 1000.0 / 300.0   # 3.333 ns/sample (300 MSPS)
FADC_BIN_NS = 25.0             # 25 ns/sample  (40 MSPS)

# Front-end impedance + electron charge to convert ∫V dt → charge in PE.
# IceCube uses 50 Ω front-end; PMT gain is read from I3DOMCalibration.
FE_IMPEDANCE_OHM = 50.0
E_CHARGE_C = 1.602176634e-19  # Coulomb


def _pick_hlc_slc(rd):
    hlc = slc = None
    for om, launches in rd:
        for L in launches:
            if L.lc_bit and hlc is None:
                hlc = (om, L)
            elif (not L.lc_bit) and slc is None:
                slc = (om, L)
            if hlc and slc:
                return hlc, slc
    return hlc, slc


def launch_to_dict(om, L) -> dict:
    """Serialize an I3DOMLaunch into pure-python primitives."""
    atwd = {ch: list(L.raw_atwd[ch]) for ch in range(len(L.raw_atwd))
            if len(L.raw_atwd[ch]) > 0}
    fadc = list(L.raw_fadc)
    chargestamp = list(L.raw_charge_stamp)

    return {
        "om": (int(om.string), int(om.om), int(om.pmt)),
        "lc_bit": bool(L.lc_bit),
        "time_ns": float(L.time),  # launch time relative to frame T0
        "is_pedestal_sub": bool(L.is_pedestal_sub),
        "trigger_type": str(L.trigger_type),
        "atwd_chip": int(L.which_atwd),
        "atwd_samples": atwd,     # {channel_index: [adc_counts...]}
        "atwd_bin_ns": ATWD_BIN_NS,
        "fadc_samples": fadc,     # [adc_counts...]
        "fadc_bin_ns": FADC_BIN_NS,
        "chargestamp_samples": chargestamp,  # used by SLC; empty for HLC
        "chargestamp_highest_sample": int(L.charge_stamp_highest_sample),
    }


def waveform_to_dict(wf) -> dict:
    """Serialize an I3Waveform — samples converted to real volts."""
    from icecube.icetray import I3Units
    return {
        "time_ns": float(wf.time),
        "bin_width_ns": float(wf.bin_width),
        "source": str(wf.source),       # ATWD or FADC
        "channel": int(wf.channel),
        "samples_volt": [float(v) / I3Units.V for v in wf.waveform],
    }


def main():
    print(f"reading data: {I3_FILE}")
    print(f"reading GCD:  {GCD_FILE}")

    captured: dict = {}

    def grab(frame):
        if "InIceRawData" not in frame or "CalibratedWaveforms" not in frame:
            return True
        rd = frame["InIceRawData"]
        hlc, slc = _pick_hlc_slc(rd)
        if hlc is None or slc is None:
            return True
        cal_wfs = frame["CalibratedWaveforms"]
        cal = frame["I3Calibration"]
        det = frame["I3DetectorStatus"]
        hdr = frame["I3EventHeader"]

        hlc_om, hlc_L = hlc
        slc_om, slc_L = slc

        hlc_d = launch_to_dict(hlc_om, hlc_L)
        slc_d = launch_to_dict(slc_om, slc_L)

        if hlc_om in cal_wfs:
            hlc_d["calibrated_waveforms"] = [
                waveform_to_dict(w) for w in cal_wfs[hlc_om]
            ]
        else:
            hlc_d["calibrated_waveforms"] = []

        dom_cal = cal.dom_cal[hlc_om]
        dom_st = det.dom_status[hlc_om]
        # gain fit:  log10(gain) = a + b * log10(HV / volt)
        # HV is stored in I3Units::volt (so float value is in volts already).
        import math
        from icecube.icetray import I3Units
        hv_volts = float(dom_st.pmt_hv) / I3Units.V
        a = float(dom_cal.hv_gain_fit.intercept)
        b = float(dom_cal.hv_gain_fit.slope)
        pmt_gain = 10.0 ** (a + b * math.log10(hv_volts))
        hlc_d["pmt_gain"] = float(pmt_gain)
        hlc_d["pmt_hv_volts"] = hv_volts
        hlc_d["fe_impedance_ohm"] = FE_IMPEDANCE_OHM
        # ∫V(t) dt  [V·s] → charge in PE
        #   Q_pe = ∫V dt / (R_FE · gain_PMT · e)
        hlc_d["pe_per_voltsecond"] = float(
            1.0 / (FE_IMPEDANCE_OHM * pmt_gain * E_CHARGE_C)
        )
        captured.update({
            "i3_file": str(I3_FILE),
            "gcd_file": str(GCD_FILE),
            "event": {
                "run_id": hdr.run_id,
                "event_id": hdr.event_id,
                "sub_event_id": hdr.sub_event_id,
                "start_time_mjd": hdr.start_time.mod_julian_day_double,
                "stop": str(frame.Stop),
            },
            "hlc": hlc_d,
            "slc": slc_d,
        })
        return True

    tray = I3Tray()
    tray.Add("I3Reader", FilenameList=[str(GCD_FILE), str(I3_FILE)])
    tray.Add("I3WaveCalibrator", "wavecal",
             Launches="InIceRawData",
             Waveforms="CalibratedWaveforms",
             WaveformRange="CalibratedWaveformRange_recalc")
    tray.Add(grab, Streams=[icetray.I3Frame.DAQ])

    def stop_when_done(frame):
        if captured:
            tray.RequestSuspension()
        return True
    tray.Add(stop_when_done, Streams=[icetray.I3Frame.DAQ])

    tray.Execute()

    if not captured:
        raise RuntimeError("No frame with HLC+SLC launches found.")

    ev = captured["event"]
    hlc = captured["hlc"]
    slc = captured["slc"]
    print(f"  event: run={ev['run_id']} id={ev['event_id']}")
    print(f"  HLC OM: {hlc['om']}  raw ATWD chans={list(hlc['atwd_samples'].keys())}  "
          f"fADC len={len(hlc['fadc_samples'])}  "
          f"calibrated waveforms: {len(hlc['calibrated_waveforms'])}")
    for w in hlc["calibrated_waveforms"]:
        print(f"     - {w['source']} ch{w['channel']}: "
              f"{len(w['samples_volt'])} samples × {w['bin_width_ns']:.3f} ns")
    print(f"  HLC PMT gain: {hlc['pmt_gain']:.3e}")
    print(f"  SLC OM: {slc['om']}  chargestamp={slc['chargestamp_samples']}")

    with open(OUT, "wb") as f:
        pickle.dump(captured, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
