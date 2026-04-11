# IceCube Thesis Analysis

Machine-learning framework for reconstructing and classifying events in the
[IceCube Neutrino Observatory](https://icecube.wisc.edu/). This repository
contains the analysis code developed for my Master's thesis at the
Niels Bohr Institute, University of Copenhagen.

The work applies graph neural networks and transformers to raw detector
pulses in order to estimate particle properties (direction, energy,
position, particle type) and to validate Monte Carlo simulations against
real detector data.

> **Note on data and weights.** IceCube simulation databases, burnsample
> data, and trained model checkpoints are not distributed with this
> repository — they live on the HEP cluster at NBI and are too large for
> version control. The code is provided as-is for reference; paths in
> configs point to `/groups/icecube/...` and must be adapted to run
> outside the NBI environment.

---

## Physics context

IceCube is a cubic-kilometre neutrino detector instrumented with ~5000
digital optical modules (DOMs) deep in the Antarctic ice. Neutrinos are
observed indirectly through the Cherenkov light produced by charged
secondary particles (mainly muons and cascades) traversing the ice.

Each event is a sparse point cloud of DOM pulses with features
*(x, y, z, t, charge)*. The analysis task is to map that point cloud to
physical quantities:

- **Particle type (PID).** Is the event a track (νμ CC), a cascade
  (νe / ντ / NC), or an atmospheric muon background?
- **Direction.** Zenith and azimuth of the incoming particle.
- **Energy.** Deposited energy, correlated with the neutrino energy.
- **Interaction vertex.** Where in the detector did the event start?
- **Track topology.** For muons: does the track stop inside the
  detector or pass straight through?

Classical reconstructions rely on hand-crafted likelihoods. The models
in this repository instead learn directly from pulse-level data using
GNNs (via [GraphNeT](https://github.com/graphnet-team/graphnet)) and
custom pulse transformers built in PyTorch / PyTorch Lightning.

---

## Repository layout

```
.
├── I3_reader/                       # Read raw IceCube .i3 simulation files
├── Classifiers/                     # All ML models (GNN + transformers)
│   ├── PID_Classifier/              #   Particle-ID multiclass GNN
│   ├── Energy_recon/                #   Energy regression GNN
│   ├── finding_the_angles/          #   Direction reconstruction (DynEdge)
│   ├── Inars_zenith_azimuth_        #   Pulse transformer for direction,
│   │   transformer_recon/           #   energy and vertex position
│   └── Muon_Reconstruction/         #   Model zoo for muon reconstruction
│       ├── dynedge_40k_l3_muons/    #     DynEdge GNN baseline (40k sample)
│       ├── dynedge_720k/            #     DynEdge GNN on 720k muons
│       ├── dynedge_transformer/     #     DynEdge + transformer hybrid
│       ├── pulse_transformer/       #     Pulse-level transformer
│       ├── dom_transformer/         #     DOM-aggregated transformer
│       ├── multi_task_transformer/  #     Multi-task (E, dir, pos) head
│       ├── transformer/             #     Base transformer experiments
│       └── reweighting_multitask_   #     GBReweighter MC→BS on reco
│           mc_to_bs/                #     features
├── MC_vs_BS_analysis/               # Monte Carlo vs. Burnsample comparison
│   ├── scripts/                     #   DB construction, inference drivers
│   ├── notebooks/                   #   Exploratory plots and checks
│   └── Data/                        #   (DBs gitignored; tiny metadata only)
└── ThroughOrStopped_muon/           # Pulse-transformer stopped/through classifier
```

---

## Components in more detail

### `I3_reader/`
Starting point of the pipeline. A notebook that opens IceCube
`.i3` simulation files with `icetray` and extracts pulses and truth-level
labels into SQLite/pandas-friendly form for downstream training.

### `Classifiers/PID_Classifier/`
Multiclass GraphNeT / DynEdge model that assigns each event a
particle-ID probability (νμ CC, νe / ντ, NC, muon background).
`unified_run.py` is a general-purpose driver that supports PID, energy,
direction and vertex tasks from a single YAML configuration.

### `Classifiers/Energy_recon/`
GNN regressor for event energy, built on GraphNeT with the IceCubeDeepCore
detector module. `config.yaml` sets the training hyperparameters;
`inference_run.py` runs a trained model on new events.

### `Classifiers/finding_the_angles/`
Collection of DynEdge direction reconstructions, trained on MuonGun
muons and low-energy neutrinos. `train_neutrino_direction.py` is the
reference training script; `evaluate_*` notebooks compare predicted
against true zenith/azimuth distributions.

### `Classifiers/Inars_zenith_azimuth_transformer_recon/`
A custom pulse transformer (each pulse = one token) trained to
simultaneously reconstruct direction, energy and the vertex-Z position
of muon events. Results are benchmarked against the DynEdge baseline
in the `notebooks/` folder.

### `Classifiers/Muon_Reconstruction/`
Model zoo used to compare reconstruction architectures on a common
720k-event muon dataset:

- **DynEdge** (GraphNeT baseline) at 40 k and 720 k events.
- **Pulse transformer** — each DOM hit is a transformer token.
- **DOM transformer** — tokens are aggregated per DOM.
- **DynEdge + transformer hybrid** — GNN feature extractor feeding a
  transformer head.
- **Multi-task transformer** — shared backbone with per-task heads for
  energy, direction and vertex position.
- **`reweighting_multitask_mc_to_bs/`** — three-step pipeline that (1)
  selects burnsample muons, (2) runs the multitask transformer on MC
  and BS, (3) fits a `GBReweighter` (hep_ml) on the reconstructed
  features to reweight MC so that its distributions align with real
  detector data. The alignment quality is evaluated with a held-out
  MC/BS classifier — AUC → 0.5 means the simulation has been brought
  into agreement with data.

### `MC_vs_BS_analysis/`
End-to-end comparison of Monte Carlo simulations against the IceCube
"burnsample" (a 10% subset of real data used for analysis development).
`scripts/` contains the tools used to build merged MC+BS SQLite
databases, drive transformer inference over the burnsample, and
produce the plots in `results/`.

### `ThroughOrStopped_muon/`
Binary pulse-level transformer that predicts P(stopped | pulses). A
stopped muon deposits all its remaining energy inside the detector
volume; a through-going muon exits the other side. The model combines
a shared pulse transformer backbone with event-level summary features
(Q_tot, N_hits, Δt, Δz, charge-weighted z, t_std) through a single
binary classification head. Trained with class-balanced BCE on SLURM
via `slurm_stopped.sh`.

---

## Tech stack

- **Python 3.9 / 3.11**, **PyTorch**, **PyTorch Lightning**
- [**GraphNeT**](https://github.com/graphnet-team/graphnet) — graph
  neural networks for neutrino telescopes (DynEdge architecture)
- **`icetray`** / **`dataio`** — official IceCube framework for reading
  `.i3` files
- **`hep_ml`** — gradient-boosted reweighting (`GBReweighter`) for the
  MC-to-data correction step
- **SQLite** — event databases produced from the I3 files
- **Weights & Biases** — experiment tracking for the transformer runs
- **SLURM** — training and inference jobs run on the NBI HEP cluster;
  `slurm_*.sh` scripts are included for reference

---

## Running the code

Most entry points assume a GraphNeT-compatible environment with CUDA
and the IceCube software stack available. Broadly:

1. **Build a database.** Use `I3_reader/` or the scripts in
   `MC_vs_BS_analysis/scripts/make_big_db_with_weights/` to convert
   `.i3` files into a SQLite database of events and pulses.
2. **Train a model.** Each subfolder of `Classifiers/` has a
   `train_*.py` (or similar) script and a YAML/JSON config with the
   data paths, model hyperparameters and loss configuration.
3. **Run inference.** Use `inference_run.py` / `run_inference.py`
   against the trained checkpoint; predictions are written as CSV.
4. **Analyze.** The `notebooks/` and `results/` folders contain the
   evaluation plots, per-event scatter plots and metric summaries
   used in the thesis.

Paths in configs currently point to the NBI cluster filesystem; they
need to be adapted for any other environment.

---

## License

Released under the [MIT License](LICENSE).

## Author

Holger Christiansen — Master's student, Niels Bohr Institute,
University of Copenhagen. Thesis analysis on IceCube data using deep
learning for event reconstruction and Monte Carlo validation.
