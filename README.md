
# deadtrees.earth-aerial: A Multi-Resolution Aerial Image Dataset for Tree Cover and Mortality Detection

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2605.19605-b31b1b.svg)](https://arxiv.org/abs/2605.19605)
[![Project page](https://img.shields.io/badge/Project_page-deadtrees__aerial-2E7D32.svg)](https://ayushi-3536.github.io/deadtrees_aerial.github.io/)
[![Visualization](https://img.shields.io/badge/Visualization-DTE--aerial--bench-1565C0.svg)](https://deadtrees.earth/releases/dte-aerial-bench)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging_Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/ayushi3536/deadtree.earth-aerial-train_cc_by)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

</div>



This repository is the official implementation of [deadtrees.earth-aerial: A Multi-Resolution Aerial Image Dataset for Tree Cover and Mortality Detection](https://arxiv.org/abs/2605.19605).

If you use the data, models or code, please cite the paper (see [Citation](#citation)).


## Setup

```
# Clone the repository
git clone github.com/GeoSense-Freiburg/DTE-aerial

# Install and activate the conda environment
conda create -n dte python=3.10
conda activate dte
pip install -r requirements.txt


```
---

## Download Benchmark Dataset

The benchmark (DTE-aerial-bench: 525 expert-annotated patches from 25 sites at 5, 10 and 20 cm) is available from the [release page](https://deadtrees.earth/releases/dte-aerial-bench). After downloading, extract the two archives inside the `DTE-aerial-bench` folder:

```bash
tar -xf DTE-aerial-bench-tiles.tar
tar -xf DTE-aerial-bench-masks.tar
```

The folder should then have the following structure:

```text
DTE-aerial-bench/
├── DTE-aerial-bench-meta.csv
├── tiles/<site>/<name>.png
└── masks/<site>/<name>_mask.png
```
---

## Download the Training Data

DTE-aerial-train is hosted on Hugging Face in two repositories, split by licence. The training set used in the paper is the union of both.

| Repository | Licence | Content |
|---|---|---|
| [`ayushi3536/deadtree.earth-aerial-train_cc_by`](https://huggingface.co/datasets/ayushi3536/deadtree.earth-aerial-train_cc_by) | CC BY 4.0 (+ 353 MIT patches) | 308,651 training and 34,885 validation patches |
| [`ayushi3536/deadtree.earth-aerial-train_cc_by_nc-sa`](https://huggingface.co/datasets/ayushi3536/deadtree.earth-aerial-train_cc_by_nc-sa) | CC BY-NC-SA 4.0 | 2,090 training patches (5 orthophotos), non-commercial only |

The CC BY repository is stored as WebDataset shards organised by biome and resolution (`data/<split>/<biome_group>/<resolution>/`), so subsets can be downloaded selectively:

```python
from huggingface_hub import snapshot_download

# metadata for all patches, plus all boreal training shards
snapshot_download("ayushi3536/deadtree.earth-aerial-train_cc_by", repo_type="dataset",
                  allow_patterns=["metadata/*", "data/train/boreal/*"])
```

See the dataset card for streaming with `datasets`, selecting patches by metadata, and reading GeoTIFF patches and masks.

## Download Pre-trained Model

The pretrained model is available on Hugging Face: [DTE-aerial-model](https://huggingface.co/ayushi3536/DTE-aerial-model).

You can either download it programmatically using the Hugging Face Hub:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ayushi3536/DTE-aerial-model",
    local_dir="DTE-aerial-model",
)
```

or clone the repository with Git LFS:

```bash
git lfs install
git clone https://huggingface.co/ayushi3536/DTE-aerial-model
```

---

## Evaluation
```bash
# set data.input_dir in config/evaluation.yml to the DTE-aerial-bench folder
python eval.py --cfg config/evaluation.yml --checkpoint <PATH_TO_CHECKPOINT>
```

## Repository Structure

```text
DTE-aerial/
├── train.py                 # Training entry point
├── eval.py                  # Evaluation entry point
├── requirements.txt
├── README.md
│
├── config/
│   ├── train.yml
│   └── evaluation.yml
│
├── scripts/
│   └── data_download.py
│
└── src/
    ├── dataset/             # Dataset loading
    ├── model/               # Network architectures
    ├── loss/                # Loss functions
    └── utils/               # Utilities
```


## Training


To train the models described in the paper:

1. Prepare a configuration file (see [`config/`](./config/)) by specifying the path to the dataset.
2. Run the following command:

```bash
python train.py --cfg ./config/<config_file>.yaml --output <output_path>
```




## Pre-trained Models

All pre-trained models will be available soon

## Results

Benchmark results for all models (by biome, resolution and class) are reported in the [paper](https://arxiv.org/abs/2605.19605).


## Citation

If you use DTE-aerial, please cite:

```bibtex
@misc{sharma2026deadtreesearthaerialmultiresolutionaerialimage,
      title={deadtrees.earth-aerial: A Multi-Resolution Aerial Image Dataset for Tree Cover and Mortality Detection},
      author={Ayushi Sharma and Clemens Mosig and Lukas Drees and Salim Soltani and Janusch Vajna-Jehle and Aaron Sheppard and Belqis Ahmadi and Jonathan Schmid and Paul Neumeier and Nathan Jacobs and Jan Dirk Wegner and Teja Kattenborn},
      year={2026},
      eprint={2605.19605},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      doi={10.48550/arXiv.2605.19605},
      url={https://arxiv.org/abs/2605.19605},
}
```
