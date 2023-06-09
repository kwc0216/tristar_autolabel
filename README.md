# Tristar: Support Code

This repository contains the support code for "A Trimodal Dataset: RGB, Thermal, and Depth for Human Segmentation and Action Recognition". It contains data loaders as well as code for training and testing the described benchmark models.

## Installation

Add here instructions on how to clone this repository and install any dependencies. This might be as simple as:

```bash
git clone https://github.com/anonymous/tristar.git
cd tristar
pip install -r requirements.txt
```

## Usage

Download the dataset from [zenodo](https://zenodo.org/record/7996570), unpack and move it to data:

```
mkdir data
cd data
wget https://zenodo.org/record/7996570/files/tristar.zip
unzip tristar.zip
cd ..
```

Create your environment and install the requirements, for example:

```
conda create -n tristar python=3.8
conda activate tristar
pip install -r requirements.txt
```

To run every valid model modality combination with 10 epochs use:

```
chmod +x train_all.sh
./train_all.sh
```

Finally, to create json files with the test scores:

```
python -m tristar.test_best
```


