# FIKT Model-Zoo Entry

This directory contains the FIKT model implementation and its default training configuration.

- Model source: [src/FIKT.py](./src/FIKT.py)
- Default model configuration: [config/model_config.yaml](./config/model_config.yaml)
- Default dataset configuration: [config/dataset_config.yaml](./config/dataset_config.yaml)
- Training entry: [run_expid.py](./run_expid.py)

Run the default experiment with:

```bash
python run_expid.py --config ./config --expid FIKT_test --gpu 0
```
