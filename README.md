
# deadtrees.earth-aerial: A Multi-Resolution Aerial Image Dataset for Tree Cover and Mortality Detection


This repository is the official implementation of [deadtrees.earth-aerial: A Multi-Resolution Aerial Image Dataset for Tree Cover and Mortality Detection](Arxiv link will be shared soon). 


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
We will make our dataset public soon.

## Repository Structure
```text
DTE-Aerial/
├── train.py                 # Main training entry point
├── eval.py                  # Main evaluation entry point
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
│
├── config/                  # Configuration files
│
└── src/                     # Source code
    ├── dataset/             # Dataset loading and preprocessing
    ├── loss/                # Loss functions
    ├── model/               # Model architectures
    └── utils/               # Utility functions
```


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

Pre-trained Models will be available soon

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

License will be available soon. 

