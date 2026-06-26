# MDAGCN Reproducibility Guide

This repository is a cleaned code release for the MDAGCN sleep staging project. The goal of this README is to record the implementation details that are most relevant to replication, especially preprocessing, fold construction, normalization, random seed behavior, training schedules, and the purpose of each retained analysis script.

## Scope

The main released pipeline is the original two-stage workflow:

1. preprocess raw data into a dataset-level `.npz` file,
2. train `FeatureNet` to extract per-epoch multimodal features,
3. save `Feature_i.npz` for each validation fold,
4. train `MDAGCN` on contextualized feature windows,
5. evaluate saved checkpoints.

The `analysis/` directory is supplementary. It is not required for the main reproduction path, but it contains the scripts we used for stricter diagnostics, ablations, graph interpretation, visualization, and complexity profiling.

## Repository Layout

```text
.
+-- config/                  # Dataset-specific configuration files
+-- model/                   # FeatureNet, MDAGCN, data generators, shared utilities
+-- analysis/                # Retained diagnostic / ablation / visualization scripts
+-- checkpoints/             # Optional place for saved weights
+-- data/
|   +-- ISRUC_S3/            # Processed ISRUC-S3 npz file goes here
|   \-- SleepEDF_78/         # Processed SleepEDF-78 npz file goes here
+-- res/                     # Temporary arrays written by legacy helper code
+-- preprocess.py            # ISRUC-S3 preprocessing
+-- preprocess_edf.py        # SleepEDF-78 preprocessing
+-- train_FeatureNet.py      # ISRUC-S3 stage-1 feature extractor training
+-- train_featurenet_edf.py  # SleepEDF-78 stage-1 feature extractor training
+-- train_MDAGCN.py          # Stage-2 MDAGCN training on extracted features
\-- evaluate_MDAGCN.py       # Final evaluation from saved checkpoints
```

## Environment

The current code assumes Python plus PyTorch training on GPU. The minimal package list is in [requirements.txt](</C:/Users/33643/Desktop/MDAGCN_GitHub/requirements.txt>).

Core dependencies used by the released code:

- `torch`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `mne`
- `thop`

Important note:

- many training and evaluation scripts call `.cuda()` directly, so they expect CUDA to be available.

## Datasets and Processed Files

This repository does not redistribute raw datasets.

Expected processed files:

- `data/ISRUC_S3/ISRUC_S3.npz`
- `data/SleepEDF_78/SleepEDF_78.npz`

These are the files consumed by the training scripts through the paths stored in the config files.

## Preprocessing Details

### ISRUC-S3

The ISRUC-S3 preprocessing script is [preprocess.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/preprocess.py>).

Key implementation details:

- fixed 10-channel input:
  - `C3_A2`, `C4_A1`, `F3_A2`, `F4_A1`, `O1_A2`, `O2_A1`
  - `LOC_A2`, `ROC_A1`, `X1`, `X2`
- each channel is resampled to 3000 points per epoch using `scipy.signal.resample`
- labels are read from the ISRUC text files
- the last 30 labels are discarded with `label[:-ignore]` where `ignore=30`
- label remapping converts REM from `5` to `4`
- output arrays:
  - `Fold_data`: one subject per fold
  - `Fold_label`: one-hot labels
  - `Fold_len`: number of epochs per subject

### SleepEDF-78

The SleepEDF preprocessing script is [preprocess_edf.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/preprocess_edf.py>).

Key implementation details:

- selected channels:
  - `EEG Fpz-Cz`
  - `EEG Pz-Oz`
  - `EOG horizontal`
  - `EMG submental`
- raw filtering: `0.3-35 Hz`
- resampling: `100 Hz`
- epoch length: `30 s`, therefore `3000` samples per epoch
- label mapping:
  - `W -> 0`
  - `N1 -> 1`
  - `N2 -> 2`
  - `N3/N4 -> 3`
  - `REM -> 4`
  - unknown/movement -> `5`, then excluded
- recordings are clipped from 60 epochs before the first non-wake epoch to 60 epochs after the last non-wake epoch
- only retained labels `0-4` are saved
- each retained epoch is z-score normalized channel-wise in the preprocessing loop

## Fold Partition Reproducibility

### ISRUC-S3

ISRUC-S3 partitioning is deterministic and subject-based.

- [preprocess.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/preprocess.py>) writes subjects in order `1..10`
- [model/DataGenerator.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/model/DataGenerator.py>) uses that order directly
- in `getFold(i)`, subject `i` is the validation fold and the remaining subjects are concatenated as the training set

Therefore the ISRUC-S3 split is reproducible as long as the same processed `.npz` file is used.

### SleepEDF-78

SleepEDF partitioning is also deterministic in the released code.

- [preprocess_edf.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/preprocess_edf.py>) stores subjects in sorted EDF filename order
- [model/DataGenerator_EDF.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/model/DataGenerator_EDF.py>) assigns folds by contiguous subject index ranges
- `np.random.shuffle(indices)` is commented out, so no random fold shuffle is applied

Therefore the SleepEDF split is reproducible, but it is based on sorted subject order rather than a separately distributed split file.

## Normalization Strategy

Two normalization stages are used.

### Subject-wise signal normalization before FeatureNet

- [model/DataGenerator.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/model/DataGenerator.py>) and [model/DataGenerator_EDF.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/model/DataGenerator_EDF.py>) standardize each subject independently
- for each channel, mean and standard deviation are computed over all epochs and all 3000 time points of that subject

### Feature normalization before MDAGCN

- [train_MDAGCN.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/train_MDAGCN.py>) and [evaluate_MDAGCN.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/evaluate_MDAGCN.py>) compute feature mean and standard deviation from `train_feature`
- those statistics are then used to normalize both train and validation features for that fold

## Random Seed and Stochasticity

The main released scripts do not explicitly fix global random seeds:

- [train_FeatureNet.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/train_FeatureNet.py>)
- [train_featurenet_edf.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/train_featurenet_edf.py>)
- [train_MDAGCN.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/train_MDAGCN.py>)
- [evaluate_MDAGCN.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/evaluate_MDAGCN.py>)

This means the following randomness is not hard-fixed in the main path:

- PyTorch initialization
- mini-batch shuffling in dataloaders
- CUDA non-determinism unless controlled externally

If exact rerun matching is required, the recommended script is:

- [analysis/strict_protocol_diagnostic.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/analysis/strict_protocol_diagnostic.py>)

That script exposes:

- explicit seed list through `--seeds`
- deterministic mode through `--deterministic`
- explicit loss selection
- explicit optimizer selection
- optional balanced sampler
- explicit output directory for controlled reruns

### Recommended seed-controlled command

For a single-seed controlled rerun on ISRUC-S3:

```bash
python analysis/strict_protocol_diagnostic.py \
  -c config/ISRUC_S3.config \
  -g 0 \
  --seeds 2024 \
  --feature-root output_ISRUC/ \
  --save-root output_ISRUC/strict_protocol_seed2024/ \
  --selection-metric macro_f1 \
  --loss weighted_ce \
  --optimizer adam \
  --deterministic
```

For a multi-seed controlled rerun:

```bash
python analysis/strict_protocol_diagnostic.py \
  -c config/ISRUC_S3.config \
  -g 0 \
  --seeds 2024,2025,2026 \
  --feature-root output_ISRUC/ \
  --save-root output_ISRUC/strict_protocol_multiseed/ \
  --selection-metric macro_f1 \
  --loss weighted_ce \
  --optimizer adam \
  --deterministic
```

Important interpretation:

- this script is best viewed as a controlled diagnostic rerun of the MDAGCN stage, not as the original published main training entry point
- it requires `Feature_i.npz` to already exist, so stage-1 FeatureNet training must be completed first

## Training Schedule

### Stage 1: FeatureNet on ISRUC-S3

Configured by [config/ISRUC_S3.config](</C:/Users/33643/Desktop/MDAGCN_GitHub/config/ISRUC_S3.config>):

- `epoch_f = 100`
- `batch_size_f = 64`
- optimizer: `adam`
- learning rate: `1e-4`

Behavior from [train_FeatureNet.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/train_FeatureNet.py>):

- weighted cross-entropy
- inverse-frequency style class weights normalized to sum to 5
- `ReduceLROnPlateau(mode='min', factor=0.5, patience=3)`
- gradient accumulation: `2`
- early stopping after `15` non-improving epochs
- best model saved when validation accuracy improves or validation loss decreases

### Stage 1: FeatureNet on SleepEDF-78

Configured by [config/SleepEDF.config](</C:/Users/33643/Desktop/MDAGCN_GitHub/config/SleepEDF.config>):

- `epoch_f = 60`
- `batch_size_f = 256`
- optimizer: `adam`
- learning rate: `1e-4`

Behavior from [train_featurenet_edf.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/train_featurenet_edf.py>):

- standard cross-entropy
- `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)`
- gradient accumulation: `2`
- early stopping after `15` non-improving epochs
- best model saved when validation accuracy improves or validation loss decreases

### Stage 2: MDAGCN

ISRUC-S3 config in [config/ISRUC_S3.config](</C:/Users/33643/Desktop/MDAGCN_GitHub/config/ISRUC_S3.config>):

- context window: `5`
- epochs: `100`
- batch size: `64`
- optimizer: `adam`
- learning rate: `8e-5`
- graph-learning alpha: `1e-5`
- Chebyshev order: `3`
- graph conv filters: `128`
- temporal conv filters: `64`
- temporal kernel: `3`
- dropout: `0.6`
- weight decay: `0.001`

SleepEDF config in [config/SleepEDF.config](</C:/Users/33643/Desktop/MDAGCN_GitHub/config/SleepEDF.config>):

- context window: `5`
- epochs: `80`
- batch size: `128`
- optimizer: `adam`
- learning rate: `5e-4`
- graph-learning alpha: `1e-4`
- Chebyshev order: `3`
- graph conv filters: `128`
- temporal conv filters: `64`
- temporal kernel: `3`
- dropout: `0.6`
- weight decay: `0.001`

Behavior from [train_MDAGCN.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/train_MDAGCN.py>):

- input features are loaded from `Feature_i.npz`
- train features are normalized and converted to contextual windows
- `AddContext_MultiSub(...)` is used for training context
- `AddContext_SingleSub(...)` is used for validation context
- weighted cross-entropy with square-root inverse-frequency weights
- `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)`
- gradient clipping: `max_grad_norm = 2.0`
- early stopping patience: `20`
- model selection metric: `val_acc - 0.1 * val_loss`

## Evaluation Protocol

[evaluate_MDAGCN.py](</C:/Users/33643/Desktop/MDAGCN_GitHub/evaluate_MDAGCN.py>) evaluates each fold using:

- the saved `Feature_i.npz`
- fold-specific feature normalization computed from that fold's training features
- the saved `MDAGCN_Best_i.pth`
- validation context created by `AddContext_SingleSub(...)`

The script aggregates predictions over all folds and reports:

- accuracy
- macro F1
- Cohen's kappa
- per-class metrics through `PrintScore(...)`
- confusion matrix output through `ConfusionMatrix(...)`

## Minimal Reproduction Commands

### ISRUC-S3

```bash
python preprocess.py
python train_FeatureNet.py -c config/ISRUC_S3.config
python train_MDAGCN.py -c config/ISRUC_S3.config -g 0
python evaluate_MDAGCN.py -c config/ISRUC_S3.config -g 0
```

### SleepEDF-78

```bash
python preprocess_edf.py
python train_featurenet_edf.py -c config/SleepEDF.config
python train_MDAGCN.py -c config/SleepEDF.config -g 0
python evaluate_MDAGCN.py -c config/SleepEDF.config -g 0
```

## Analysis Scripts

The `analysis/` directory is supplementary. These scripts are not required for the main two-stage MDAGCN reproduction path.

### `analysis/strict_protocol_diagnostic.py`

Role:

- controlled rerun of MDAGCN with explicit seed handling and configurable loss / optimizer choices

Typical inputs:

- processed dataset from the selected config
- `Feature_i.npz`

Typical outputs:

- `summary.json`
- `fold_results.csv`
- per-seed prediction files
- optional saved checkpoints

Recommended use:

- use this script when you need a seed-controlled rerun or want to show a stricter training protocol than the legacy `train_MDAGCN.py`

### `analysis/analyze_adaptive_graphs.py`

Role:

- export and analyze the adaptive graphs learned by trained MDAGCN checkpoints

Typical inputs:

- `Feature_i.npz`
- `MDAGCN_Best_i.pth`

Typical command:

```bash
python analysis/analyze_adaptive_graphs.py \
  -c config/ISRUC_S3.config \
  --feature-root output_ISRUC/ \
  --model-root output_ISRUC/ \
  --out-dir output_ISRUC/adaptive_graph_analysis/
```

### `analysis/fine_grained_ablation.py`

Role:

- structural ablation of temporal attention, spatial attention, graph loss, and related MDAGCN components

Typical inputs:

- processed dataset from config
- `Feature_i.npz`

Typical command:

```bash
python analysis/fine_grained_ablation.py \
  -c config/ISRUC_S3.config \
  -g 0 \
  --seeds 2024,2025,2026 \
  --feature-root output_ISRUC/ \
  --save-root output_ISRUC/fine_grained_ablation/ \
  --selection-metric macro_f1 \
  --loss weighted_ce \
  --optimizer adamw
```

### `analysis/evaluate_ablation.py`

Role:

- older coarse ablation script kept mainly for historical comparison

Typical command:

```bash
python analysis/evaluate_ablation.py -c config/ISRUC_S3.config -g 0
```

### `analysis/single_modality_analysis.py`

Role:

- rerun MDAGCN using only one modality at a time to estimate the contribution of EEG, EOG, EMG, or ECG

Typical command:

```bash
python analysis/single_modality_analysis.py \
  -c config/ISRUC_S3.config \
  -g 0 \
  --variants EEG,EOG,EMG,ECG \
  --seeds 2024,2025,2026 \
  --feature-root output_ISRUC/ \
  --save-root output_ISRUC/single_modality_analysis/
```

### `analysis/mutimodal_ablation.py`

Role:

- older multimodal ablation experiment comparing EEG-only and combined modality settings

Typical command:

```bash
python analysis/mutimodal_ablation.py -c config/ISRUC_S3.config -g 0
```

### `analysis/plot_multimodal_tsne.py`

Role:

- visualize the outputs saved by `mutimodal_ablation.py`
- generate per-setting t-SNE plots, combined t-SNE plots, and performance comparison figures

Required input:

- `multimodal_ablation_results.npy` in the dataset `save` directory

Typical command:

```bash
python analysis/plot_multimodal_tsne.py -c config/ISRUC_S3.config
```

### `analysis/profile_complexity.py`

Role:

- profile parameter count, MACs, inference time, and optional memory usage for the full model and selected ablation variants

Typical command:

```bash
python analysis/profile_complexity.py \
  -c config/ISRUC_S3.config \
  --device cuda \
  --out-dir output_ISRUC/complexity_profile/
```

Useful options:

- `--variants full,no_temporal_attention,no_spatial_attention,no_attention,no_graph_loss`
- `--ablation-summary path/to/all_summary.json`
- `--coarse-ablation path/to/proper_ablation_results.npy`
- `--full-eval path/to/Result_MDAGCN_Evaluation.txt`