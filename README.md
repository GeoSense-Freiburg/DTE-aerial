
# deadtrees.earth-aerial: A Multi-Resolution Aerial Image Dataset for Tree Cover and Mortality Detection


This repository is the official implementation of [deadtrees.earth-aerial: A Multi-Resolution Aerial Image Dataset for Tree Cover and Mortality Detection](https://arxiv.org/abs/2605.19605).

**Data:** [DTE-aerial-train (CC BY)](https://huggingface.co/datasets/ayushi3536/deadtree.earth-aerial-train_cc_by) · [DTE-aerial-train (CC BY-NC-SA)](https://huggingface.co/datasets/ayushi3536/deadtree.earth-aerial-train_cc_by_nc-sa) · [DTE-aerial-bench](https://deadtrees.earth/releases/dte-aerial-bench) · **Model:** [DTE-aerial-model](https://huggingface.co/ayushi3536/DTE-aerial-model) · **Project page:** [deadtrees_aerial.github.io](https://ayushi-3536.github.io/deadtrees_aerial.github.io/)

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

## Download Benchmark Dataset

The benchmark (DTE-aerial-bench: 525 expert-annotated patches from 25 sites at 5, 10 and 20 cm) is available from the [release page](https://deadtrees.earth/releases/dte-aerial-bench). After downloading and extracting it, the folder should have the following structure:

```text
DTE-Aerial-Data/
├── DTE-aerial-bench-meta.csv
├── tiles/
└── masks/
```
---

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
update <input_dir> in ./config/evaluation.yml

python evaluation.py --cfg ./config/evaluation.yml --checkpoint <PATH_TO_CHECKPOINT>
```

## Repository Structure

```text
DTE-aerial/
├── train.py                 # Training entry point
├── evaluation.py            # Evaluation entry point
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
      url={https://arxiv.org/abs/2605.19605},
}
```
