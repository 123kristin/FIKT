# FIKT
测试

This repository contains the implementation of **FIKT: Feature-Interaction Knowledge Tracing with Adaptive Graphs and IRT-Inspired Prediction**.

FIKT is designed for feature-enriched knowledge tracing records, where each interaction may include categorical and numerical behavioral fields such as student identifiers, item identifiers, concept labels, hint usage, timestamps, and other context signals. Instead of treating these fields as loosely attached side features, FIKT represents them as feature nodes, learns adaptive interactions among them, and combines the resulting graph-derived state with an IRT-inspired prediction module.

## Overview

The model contains three main stages:

1. Feature embedding for heterogeneous categorical and numerical fields.
2. Adaptive feature-interaction graph learning with recurrent graph-state updates.
3. IRT-inspired prediction with difficulty-related and ability-related neural factors plus a context-dependent gated readout.

The implementation is based on the FuxiCTR training framework and is organized as a model-zoo entry under `model_zoo/FIKT`.

## Repository Structure

```text
FIKT/
├── README.md
├── LICENSE
├── model_zoo/
│   └── FIKT/
│       ├── config/
│       │   ├── dataset_config.yaml
│       │   └── model_config.yaml
│       ├── run_expid.py
│       └── src/
│           └── FIKT.py
├── experiment/
│   ├── run_expid.py
│   └── run_param_tuner.py
├── fuxictr/
├── demo/
│   ├── example1_build_dataset_to_parquet.py
│   └── config/
└── data/
```

## Model Entry Points

The most relevant files are:

- Model definition: [model_zoo/FIKT/src/FIKT.py](./model_zoo/FIKT/src/FIKT.py)
- Training entry: [model_zoo/FIKT/run_expid.py](./model_zoo/FIKT/run_expid.py)
- Model hyperparameters: [model_zoo/FIKT/config/model_config.yaml](./model_zoo/FIKT/config/model_config.yaml)
- Dataset paths: [model_zoo/FIKT/config/dataset_config.yaml](./model_zoo/FIKT/config/dataset_config.yaml)

## Implementation Notes

The current implementation includes:

- Feature embeddings through the FuxiCTR `FeatureEmbedding` interface.
- Embedding-table representations for categorical fields.
- Linear projection support for numerical fields through the feature processing pipeline.
- A dense directed feature graph with attention-based edge weights.
- Recurrent graph-state refinement with `GRUCell`.
- Residual connections from the initial feature embeddings.
- Difficulty-related and ability-related neural branches.
- A gated prediction head for context-dependent readout.

The IRT-inspired prediction module is implemented as:

```python
que_diff = diff_layer(dropout(feature_emb))
stu_ability = ability_layer(dropout(h_out))
p = h_out - que_diff + stu_ability
y_pred = sigmoid(gated_readout(p))
```

The difficulty-related and ability-related quantities are learned neural factors used for prediction. They should not be interpreted as externally calibrated psychometric parameters.

## Default Training Configuration

The default configuration in [model_zoo/FIKT/config/model_config.yaml](./model_zoo/FIKT/config/model_config.yaml) uses:

- optimizer: `adam`
- learning rate: `1e-3`
- batch size: `256`
- embedding dimension: `128`
- graph layers: `1`
- residual connection: `True`
- GRU update: `True`
- parameter sharing across graph layers: `False`
- epochs: `200`
- early stopping patience: `10`
- monitor metric: `AUC`
- monitor mode: `max`
- random seed: `2024`

The training framework additionally applies gradient clipping with a default maximum norm of `10`.

## Data Preparation

This repository includes a demo-style feature configuration for `bridge2006_csv`:

- [demo/config/feature_map_config_bridge_2006/dataset_config.yaml](./demo/config/feature_map_config_bridge_2006/dataset_config.yaml)

The demo configuration uses:

- categorical fields:
  - `Anon Student Id`
  - `KC(SubSkills)`
  - `Questions`
  - `Hints`
- numerical field:
  - `First Transaction Time`
- label:
  - `correct`

To build parquet-format training data from the demo configuration:

```bash
cd demo
python example1_build_dataset_to_parquet.py
```

## Training

To run the default FIKT experiment:

```bash
cd model_zoo/FIKT
python run_expid.py --config ./config --expid FIKT_test --gpu 0
```

Use `--gpu -1` to run on CPU.

The script will:

1. load dataset and model configurations;
2. build the feature map if the input format is CSV;
3. instantiate `FIKT`;
4. train with validation monitoring;
5. evaluate the best checkpoint on validation and test sets.

## Hyperparameter Search

The repository also includes a parameter tuning entry:

- [experiment/run_param_tuner.py](./experiment/run_param_tuner.py)

Example usage:

```bash
cd experiment
python run_param_tuner.py --config ../config/tuner_config.yaml --gpu 0
```

The tuner config file is not included by default, so this command is only meaningful after adding the corresponding tuning configuration.

## Current Scope of This Public Repository

This anonymous repository is intended to document the FIKT implementation and provide the main modeling pipeline used in the paper. A few points should be noted:

- The included public configuration most directly exposes the `bridge2006_csv` demo setup.
- The manuscript reports experiments on four public KT datasets, but not every final experiment-specific configuration file is included here.
- Dataset preprocessing and feature configuration should be checked carefully before reproducing a specific table from the paper.

## Citation

If you use this repository, please cite the corresponding FIKT paper once the final bibliographic information is available.

```bibtex
@article{fikt,
  title   = {FIKT: Feature-Interaction Knowledge Tracing with Adaptive Graphs and IRT-Inspired Prediction},
  author  = {Anonymous for review},
  journal = {Applied Intelligence},
  year    = {under review}
}
```

## License

This project is released under the license provided in [LICENSE](./LICENSE).
