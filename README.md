# SimBaMM on UK Biobank

This branch contains the UK Biobank implementation of **SimBaMM**, the simple
late-fusion Transformer baseline introduced in
[Fusion or Confusion? Multimodal Complexity Is Not All You Need](https://arxiv.org/abs/2512.22991).
All other benchmark methods and datasets have been removed.

## Architecture

Each configured UKB tabular modality is encoded by an independent MLP into one
token. A learned classification token is prepended, missing modalities are
masked, and a Transformer encoder produces the binary 10-year mortality logit.

## Configuration

The configuration is intentionally UKB-specific:

- `config/config.yaml`: runtime, trainer, and W&B settings
- `config/modelname/transformer.yaml`: SimBaMM and optimizer hyperparameters
- `config/encoders/ukb.yaml`: modality dimensions and the shared MLP layout
- `config/datamodule/default.yaml`: dataloader and split settings
- `config/datamodule/plugins/`: paths for the 23 UKB modalities and mortality labels

The data paths point to the original project storage. Update `splits` in
`config/datamodule/default.yaml` and the relevant `data_paths`/`eid_map_path`
entries under `config/datamodule/plugins/` when running in another environment.

## Run

The pipeline requires PyTorch, PyTorch Lightning, Hydra, torchmetrics,
schedulefree, W&B, and the internal `udm` package used to load UKB data.

```bash
python main.py
```

Hydra overrides can be supplied on the command line, for example:

```bash
python main.py trainer.max_epochs=20 wandb.mode=disabled
```

## Citation

```bibtex
@unpublished{Rheude_Eils_Wild_2025,
  title={Fusion or Confusion? Multimodal Complexity Is Not All You Need},
  url={https://arxiv.org/abs/2512.22991},
  doi={10.48550/arXiv.2512.22991},
  author={Rheude, Tillmann and Eils, Roland and Wild, Benjamin},
  year={2025},
  month=dec
}
```
