# Age-Aware Sensor Fusion

Evaluation code and saved model outputs for **Age-Aware Sensor Fusion: Threshold Effects and Calibration Transfer**.

The scripts evaluate threshold effects and calibration transfer on REALDISP and
MHEALTH using fixed prediction caches. Communication conditions are controlled
replays applied to recorded sensor data.

## Download and reproduce Table II

Download the [reproducibility package](https://github.com/xxy-liu/age-aware-sensor-fusion/releases/download/v1.0.5/reproducibility.zip) from
[release v1.0.5](https://github.com/xxy-liu/age-aware-sensor-fusion/releases/tag/v1.0.5) and extract it into an empty folder.
The archive includes scripts, configurations, reference results and prediction caches.
Scripts and text configurations can also be browsed in this repository; NPZ arrays are in the release archive.

With Python 3.10-3.12, run these commands from the extracted package folder:

```sh
python -m pip install -r requirements.txt
python reproduce.py --table 2
```

The second command recomputes the six rows and 18 displayed values in Table II
from saved per-query predictions on the CPU. Results are written to `outputs/`;
no model training or inference is required.

| Evaluation material | Location |
| --- | --- |
| Main evaluation script | [`reproduce.py`](reproduce.py) |
| Additional analyses | [`supplement.py`](supplement.py) |
| Separate metric and threshold checks | [`audit.py`](audit.py) |
| Participant allocations | [`config/participants.csv`](config/participants.csv) |
| Frozen threshold configurations | [`config/thresholds.json`](config/thresholds.json), [`config/target_sweep.json`](config/target_sweep.json) |
| Evaluation settings | [`config/evaluation.json`](config/evaluation.json) |
| Cached predictions | `data/` in the [complete package](https://github.com/xxy-liu/age-aware-sensor-fusion/releases/download/v1.0.5/reproducibility.zip): 30 calibration and 30 evaluation NPZ files |
| Reference results and validation | [`reference/`](reference/), [`validation.json`](validation.json) |
| Package checksum | [SHA256SUMS](https://github.com/xxy-liu/age-aware-sensor-fusion/releases/download/v1.0.5/SHA256SUMS) |

## Evaluation details

The script reconstructs the 12 matching thresholds from calibration caches and
applies the saved thresholds to evaluation caches. Metrics are calculated for each
participant, seed and direction, then averaged over the six paired runs within
each participant and equally across participants. Results are computed from the
prediction caches and checked against the reference table. Any difference in a
printed Table II value causes the check to fail.

Outputs are written inside this package to `outputs/table2.csv`,
`outputs/table2.tex`, `outputs/table2_participants.csv`, `outputs/table2_runs.csv`,
and `outputs/verification.json`. Table cells are percentages; participant/run
metrics are proportions. Existing generated outputs are replaced on rerun.

To check all packaged file hashes and recompute the base recognition metrics and
retrospective curves for all included groups:

```sh
python reproduce.py --table 2 --all --verify-files
```

This command compares every recomputed metric and curve row with the saved
results at an absolute tolerance of `2e-12`, covering all supplied components,
timing scenarios, seeds, directions, architectures and partitions. It does not
regenerate every manuscript figure or bootstrap contrast.

After the main example, `python audit.py` checks Table II with a separate
marginal-class-count calculation, verifies participant allocations, and checks
that each matching threshold is the smallest feasible observed calibration score.

## Additional analyses

```sh
python supplement.py cache
python supplement.py resampling
```

`cache` refits thresholds at four calibration targets and recomputes prediction
changes, calibration transfer and crossed readouts for the original partitions
under nine timing conditions. Thresholds and all nine output tables are checked
against the frozen analysis used for Table III and Figures 3-4.

`resampling` recomputes the calibration-sensitivity summaries, including Table V,
by applying saved block-resample thresholds to fixed evaluation predictions.
It uses the existing 1,000-resample design and frozen thresholds, without drawing
new samples or refitting thresholds. Original sampling arrays, threshold arrays,
analysis source and audit records are included.

`supplement.py` provides the package entry point for both analyses. It maps the
historical analysis scripts to the packaged prediction caches and writes results
to `outputs/`. Each run records its numerical comparisons and metadata mappings.

## Source records

[`config/provenance.json`](config/provenance.json) maps packaged files to their
research source files. [`revision.json`](revision.json) records source hashes,
publication revisions and numerical checks. The [numerical audit record](supporting/cache_analyses/原始核验.json)
contains the saved computational checks and results. Historical audit results
remain distinct from reports produced by a new package run.

## Contents and scope

- `data/`: 60 byte-identical NPZ prediction caches (30 calibration and 30
  evaluation). REALDISP: original CNN/LSTM/DeepConvLSTM and three CNN repartitions;
  MHEALTH: original CNN and three CNN repartitions. Each group contains three seeds.
- `config/participants.csv`: all Table I allocations. `O` is the original
  partition; `1`-`3` are dependent repartitions. `test` means the manuscript's
  evaluation role, and `order` preserves the listed allocation order.
- `config/thresholds.json`: actual per-seed/direction/component threshold values,
  calibration targets, and age anchors where applicable.
- `config/evaluation.json`: class sets, seeds, role assignments, file mappings,
  threshold conventions and aggregation rules.
- `config/target_sweep.json`: frozen 5%, 10%, 15% and 20% target configurations.
- `reference/`: saved per-run results/curves and the printed Table II cells.
- `supporting/`: frozen supplementary analysis designs, thresholds, saved summaries,
  scripts and audit records, including the calibration-resampling state.
- `config/provenance.json`: mapping to original project files and SHA-256 values.
- `manifest.json` in the release archive: SHA-256 and byte size for every distributed input/code file.
  Generated `outputs/` and this manifest itself are excluded.

This package reproduces evaluation from saved predictions. Training and preprocessing
are described in the paper and are not rerun by these commands. Original sensor
signals, model weights, participant names and contact details are excluded;
participant identifiers are the public datasets' numbered IDs. Previously exposed
participants and dependent repartitions retain that status. A successful replay
verifies the stated computations, not independent scientific replication.

## Prediction-array schema

Load NPZ files with `numpy.load(path, allow_pickle=False)`.

| Field | Meaning |
| --- | --- |
| `subject` | Public dataset participant ID for each query |
| `row`, `old` | Current and stale capture endpoint indices within the original record |
| `y` | Ground-truth class; 0 denotes dataset-specific background |
| `scenario`, `scenarios` | Scenario index and its name mapping; delay names are nominal milliseconds |
| `direction` | 0: sensor branch 0 current / branch 1 stale; 1: branch 0 stale / branch 1 current |
| `age` | Actual age of the stale sensor window in seconds |
| `margin` | Query-by-component maximum action logit minus background logit |
| `action` | Query-by-component action argmax, excluding background |
| `components` | Column names identifying training condition, budget and input readout |
| `record` | Record identifier where present |
| `current_idx`, `old_idx` | Original concatenated-record indices retained for provenance |
| `fake_age` | Saved shuffled-age control, where present |

For REALDISP, branches 0/1 are right-lower-arm/right-calf acceleration. For MHEALTH,
they are left-ankle/right-lower-arm acceleration. `aware_native` on original
REALDISP is the 20-joint-epoch age-aware readout; `aware40_native` is its 40-epoch
endpoint. The original LSTM and DeepConvLSTM `aware_native` names follow their
architecture-specific budgets in the paper; DeepConvLSTM uses 30 direct epochs.
Other CNN groups use explicit `aware20_*`/`aware40_*` names. `*_zero` is a fixed-model
zero-age input intervention; `blind*_native` is a separately trained age-blind model.

Prediction uses `margin > threshold`; equality is background. Saved action choices
use the smallest action index for an action-logit tie. Macro-F1 includes all 34
REALDISP or 13 MHEALTH classes, including background, with a zero contribution from
zero-denominator classes. CAR is the fraction of true actions reported with the
correct label; FPR is the fraction of background queries reported as an action.

For Table II, Zero+match uses the **delay-specific** fourth threshold, fitted to
native default FPR on calibration participants. It is distinct from the
age-interpolated rule used for extra timing conditions. For global policies each
calibration participant and each of four timing anchors has equal weight.
Interpolation uses actual mean-age anchors and clamps to the nearest endpoint.
Retrospective common-FPR curve coordinates use evaluation labels and are not
deployable calibration thresholds.

## Data attribution

The labels, numbered participant IDs and endpoint metadata derive from:

- Banos, O., Toth, M., and Amft, O. (2012). **REALDISP Activity Recognition
  Dataset**, UCI Machine Learning Repository. DOI:
  [10.24432/C5GP6D](https://doi.org/10.24432/C5GP6D).
  [Dataset and license](https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset).
- Banos, O., Garcia, R., and Saez, A. (2014). **MHEALTH**, UCI Machine Learning
  Repository. DOI: [10.24432/C5TW22](https://doi.org/10.24432/C5TW22).
  [Dataset and license](https://archive.ics.uci.edu/dataset/319/mhealth+dataset).

Both UCI entries identify the dataset license as
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), checked on 2026-11-12.
The per-query model scores and selected endpoint records are derived research
outputs, not the original sensor recordings. Cite the datasets and accompanying
paper when using this material. Original dataset creators do not endorse this study.
