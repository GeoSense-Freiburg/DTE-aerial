# deadtrees.earth-aerial: A Multi-Resolution Aerial Image Dataset for Tree Cover and Mortality Detection
[![arXiv](https://img.shields.io/badge/arXiv-2511.06943-b31b1b.svg)](https://arxiv.org/abs/2605.19605)
![GitHub License](https://img.shields.io/github/license/utn-air/flownav?label=License&color=%23e11d48&cacheSeconds=60)



## Setup

```
# Clone the repository
git clone github.com/GeoSense-Freiburg/DTE-aerial

# Install and activate the conda environment
conda create -n dte python=3.10
conda activate dte
pip install -r requirements.txt


```
## Download Dataset

The DeadTrees.Earth-Aerial dataset will be publicly released soon.  
Additional details regarding dataset access, licensing, and download instructions will be provided upon release.

## Training

## 🚀 Training

To train the models described in the paper:

1. Prepare a configuration file (see [`config/`](./config/)) by specifying the path to the dataset.
2. Run the following command:

```bash
python train.py --cfg ./config/<config_file>.yaml --output <output_path>
```


## Evaluation

To evaluate the model, run:

```eval
python eval.py --checkpoint <pretrained_model.pth> --cfg <./config/<config file>> --output <output_path>
```



## Pre-trained Models

Pre-trained Models will be available upon release

## Results


### Tree Mortality Segmentation (F1)

| Model                                             | Temp     | Trop | Boreal   | Drylands | 5cm      | 10cm     | 20cm     |
| ------------------------------------------------- | -------- | ---- | -------- | -------- | -------- | -------- | -------- |
| [DT-V1](https://restor-foundation.github.io/tcd/) | 0.50     | 0.61 | 0.40     | 0.55     | 0.54     | 0.47     | 0.38     |
| **MiT-B3**                                        | **0.56** | 0.64 | **0.58** | 0.56     | **0.59** | **0.55** | **0.45** |
| MiT-B1                                            | 0.54     | 0.65 | 0.57     | 0.57     | 0.58     | 0.53     | 0.42     |
| U-Net (ResNet34)                                  | 0.52     | 0.63 | 0.53     | **0.59** | 0.57     | 0.52     | 0.42     |
| M2F (Small)                                       | 0.52     | 0.65 | 0.57     | 0.58     | 0.58     | 0.54     | 0.44     |
| DeepLabV3+ (R50)                                  | 0.50     | 0.63 | 0.54     | 0.56     | 0.56     | 0.49     | 0.40     |
| DINOv2 (Base)                                     | 0.40     | 0.61 | 0.42     | 0.54     | 0.48     | 0.46     | 0.38     |

### Tree Cover Segmentation (F1)

| Model                                               | Temp     | Trop     | Boreal   | Drylands | 5cm      | 10cm     | 20cm     |
| --------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| [OAM-TCD](https://restor-foundation.github.io/tcd/) | **0.86** | **0.91** | 0.88     | 0.86     | 0.88     | **0.89** | 0.87     |
| **MiT-B3**                                          | 0.85     | **0.91** | **0.93** | **0.93** | **0.89** | **0.89** | **0.89** |
| MiT-B1                                              | 0.85     | **0.91** | **0.93** | 0.92     | **0.89** | **0.89** | 0.88     |
| U-Net (ResNet34)                                    | 0.85     | **0.91** | 0.92     | 0.92     | **0.89** | **0.89** | 0.88     |
| M2F (Small)                                         | 0.85     | **0.91** | **0.93** | 0.92     | **0.89** | **0.89** | 0.88     |
| DeepLabV3+ (R50)                                    | 0.84     | **0.91** | **0.93** | 0.92     | **0.89** | **0.89** | 0.87     |
| DINOv2 (Base)                                       | 0.84     | **0.91** | 0.87     | 0.89     | 0.87     | 0.87     | 0.84     |



## Citation
```bibtex
@article{sharma2026deadtrees,
  title={deadtrees. earth-aerial: A Multi-Resolution Aerial Image Dataset for Tree Cover and Mortality Detection},
  author={Sharma, Ayushi and Mosig, Clemens and Drees, Lukas and Soltani, Salim and Vajna-Jehle, Janusch and Sheppard, Aaron and Ahmadi, Belqis and Schmid, Jonathan and Neumeier, Paul and Jacobs, Nathan and others},
  journal={arXiv preprint arXiv:2605.19605},
  year={2026}
}
```


