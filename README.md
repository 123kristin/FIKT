# FigKT

This repository contains the code and paper materials for **FigKT**, a knowledge tracing framework that combines:

- feature-level interaction modeling
- dynamic graph evolution
- an enhanced IRT-inspired prediction module

The corresponding manuscript is available in [sn-article.tex](./sn-article.tex).

## Overview

Modern educational platforms record more than question-response pairs. They also contain heterogeneous behavioral attributes such as student identifiers, question identifiers, hint usage, timestamps, and other contextual signals. FigKT is designed to model these attributes explicitly rather than treating them as loosely attached side information.

At a high level, the model consists of three stages:

1. Feature embedding for heterogeneous categorical and numerical attributes
2. Feature interaction graph learning with recurrent state evolution
3. Enhanced IRT-inspired prediction with difficulty- and ability-related factors

The paper figures are included in this repository:

- [fig1.pdf](./fig1.pdf): standard IRT vs. enhanced IRT-inspired prediction
- [fig2.pdf](./fig2.pdf): heterogeneous interaction records and evolving knowledge state
- [fig3.pdf](./fig3.pdf): overall FigKT architecture

## Important Note on Naming

The implementation file currently keeps the historical class name `FiGNN`, but the code in this repository corresponds to the **FigKT** model used in the paper.

In particular, the model implementation already includes the enhanced IRT-inspired branch:

- question-difficulty layer
- student-ability layer
- intermediate representation `p = h_out - que_diff + stu_ability`

Main implementation file:

- [model_zoo/FiGNN/src/FiGNN.py](./model_zoo/FiGNN/src/FiGNN.py)

## Repository Structure

```text
FigKT/
├── README.md
├── sn-article.tex
├── sn-bibliography.bib
├── fig1.pdf ... fig8.pdf
├── model_zoo/
│   └── FiGNN/
│       ├── config/
│       │   ├── dataset_config.yaml
│       │   └── model_config.yaml
│       ├── run_expid.py
│       └── src/
│           └── FiGNN.py
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

The most relevant files for the paper model are:

- Model definition:
  [model_zoo/FiGNN/src/FiGNN.py](./model_zoo/FiGNN/src/FiGNN.py)
- Training entry:
  [model_zoo/FiGNN/run_expid.py](./model_zoo/FiGNN/run_expid.py)
- Model hyperparameters:
  [model_zoo/FiGNN/config/model_config.yaml](./model_zoo/FiGNN/config/model_config.yaml)
- Dataset paths:
  [model_zoo/FiGNN/config/dataset_config.yaml](./model_zoo/FiGNN/config/dataset_config.yaml)

## Implementation Notes

From the current implementation, the model has the following characteristics:

- Feature embeddings are built through the FuxiCTR `FeatureEmbedding` interface.
- Numerical features are embedded with a linear projection.
- Categorical features are embedded with embedding tables.
- Feature interactions are modeled with attention-based fully connected message passing.
- Dynamic state updates are performed with a `GRUCell`.
- Residual connections are enabled.
- The final prediction combines graph output with difficulty- and ability-related factors.

The current model file includes:

- `diff_layer = Linear(embedding_dim, 1) + Tanh`
- `ability_layer = Linear(embedding_dim, 1) + Tanh`
- `p = h_out - que_diff + stu_ability`

These correspond to the enhanced IRT-inspired prediction module described in the paper.

## Default Training Configuration

The published configuration under [model_zoo/FiGNN/config/model_config.yaml](./model_zoo/FiGNN/config/model_config.yaml) contains the following default settings:

- optimizer: `adam`
- learning rate: `1e-3`
- batch size: `256`
- embedding dimension: `128`
- GNN layers: `1`
- residual connection: `True`
- GRU update: `True`
- parameter sharing across graph layers: `False`
- epochs: `200`
- early stopping patience: `10`
- monitor metric: `AUC`
- monitor mode: `max`
- random seed: `2024`

The training framework in `fuxictr` additionally uses gradient clipping with a default maximum norm of `10`.

## Data Preparation

This repository includes a demo-style feature configuration for `bridge2006_csv`:

- [demo/config/feature_map_config_bridge_2006/dataset_config.yaml](./demo/config/feature_map_config_bridge_2006/dataset_config.yaml)

The demo configuration uses:

- categorical features:
  - `Anon Student Id`
  - `KC(SubSkills)`
  - `Questions`
  - `Hints`
- numerical feature:
  - `First Transaction Time`
- label:
  - `correct`

You can build parquet-format training data with:

```bash
cd demo
python example1_build_dataset_to_parquet.py
```

## Training

To run the model from the model zoo entry:

```bash
cd model_zoo/FiGNN
python run_expid.py --config ./config --expid FiGNN_test --gpu 0
```

The script will:

1. load dataset and model configurations
2. build the feature map if the input format is CSV
3. instantiate the model
4. train with validation monitoring
5. evaluate the best checkpoint on validation and test sets

## Hyperparameter Search

The repository also includes a parameter tuning entry:

- [experiment/run_param_tuner.py](./experiment/run_param_tuner.py)

Example usage:

```bash
cd experiment
python run_param_tuner.py --config ../config/tuner_config.yaml --gpu 0
```

Note:
the tuner config file is not currently included in this repository, so this command is only meaningful after adding the corresponding tuning configuration.

## Current Limitations of the Public Repository

At the moment, the repository is most directly useful for understanding the FigKT implementation and reproducing the modeling pipeline for the provided configuration. However, a few points should be noted:

- The main model implementation still uses the historical name `FiGNN`.
- Public configuration files currently expose a demo setup most clearly for `bridge2006_csv`.
- The manuscript reports four datasets, but not all final experiment-specific configuration files are explicitly provided in this repository.
- The LaTeX paper and figures are included, but the repository is not yet packaged as a polished public release for full reproduction of every reported experiment.

## Paper Files

The manuscript source is included here:

- [sn-article.tex](./sn-article.tex)
- [sn-bibliography.bib](./sn-bibliography.bib)

These files can be used to revise the submission manuscript directly.

## Citation

If you use this repository, please cite the corresponding FigKT paper once the final bibliographic information is available.

```bibtex
@article{figkt,
  title   = {FigKT: Knowledge Tracing with Dynamic Feature Interaction Graph and Enhanced Item Response Theory},
  author  = {Anonymous for review},
  journal = {Applied Intelligence},
  year    = {under review}
}
```

## License

This project is released under the license provided in [LICENSE](./LICENSE).
