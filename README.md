# Fusion or Confusion?

<p align="center">
  <img src="./figures/SimBaMM_pumba.png" alt="SimBaMM pumba" width="40%">
</p>

## Abstract
Deep learning architectures for multimodal learning have seen a rapid increase in complexity, driven by the assumption that sophisticated multimodal-specific methods are required to improve performance. We challenge this assumption through a large-scale empirical study reimplementing 19 high-impact methods under standardized conditions, evaluating them across nine diverse datasets with up to 23 modalities, and testing their generalizability to new tasks beyond their original scope, including settings with missing modalities. We propose a Simple Baseline for Multimodal Learning (SimBaMM), a straightforward late-fusion Transformer architecture, and demonstrate that under standardized experimental conditions with rigorous hyperparameter tuning of all methods, more complex architectures do not reliably outperform SimBaMM. Statistical analysis confirms that more complex methods are at best practically equivalent to SimBaMM, and often fail to reliably outperform well-tuned unimodal baselines, particularly in the small data regime where many methods were originally, and often exclusively, evaluated. To strengthen our findings, we include a case study of a recent multimodal learning method highlighting the methodological shortcomings in the literature. To support comparable, robust, and trustworthy future results, we provide a pragmatic reliability checklist. In summary, we argue for a shift in focus: away from the pursuit of architectural novelty and toward methodological rigor.

## Get Started
#### Overview:
- All essential Python modules are located in the `codefiles` directory.
- Method re-implementations can be found primarily in `codefiles/methods` (with some located in other subfolders such as `codefiles/encoders` if necessary).
- Use `main.ipynb` for quick experimentation and debugging; use `main.py` for full-scale training and evaluation runs.

#### Running a Method on MOSI
To get started quickly, you can download the MOSI dataset (which originates from the [IMDer repository](https://github.com/mdswyz/IMDer?tab=readme-ov-file)) as described in the following. This dataset is compatible with our MOSI dataloader.
First, install the required `gdown` package:
```bash
pip install gdown
```
Then, download the dataset with:
```python
import gdown
url = 'https://drive.google.com/uc?id=1VqjkYqcgUlggZVN7B3NpXwQ-BIWIXxkj'
output = 'file.pkl'
gdown.download(url, output, quiet=False)
```
Place `file.pkl` in the appropriate data directory for your experiments and change the path in the MOSI dataset class accordingly (`codefiles/datasets/mosi_mosei.py`). Next, configure your experiment settings in `config/config.yaml` by setting the `dataset`, `encoders`, and `modelname` fields as desired. 

Note: Weights & Biases (WandB) logging is enabled by default and can be managed via the `wandb` section in the config file. If you prefer to disable WandB, you can comment out the related lines in `main.py` or `main.ipynb`.

## Add further datasets, methods, ...
TODO 

## Citation
In case you find our work helpful, we're happy if you cite us as following
```bibtex
@unpublished{fusion_or_confusion,
  title={Fusion or Confusion? Multimodal Complexity Is Not All You Need},
  author={Rheude, Tillmann and Eils, Roland and Wild, Benjamin},
  url={TODO},
  year={2025},
  publisher={arXiv},
}