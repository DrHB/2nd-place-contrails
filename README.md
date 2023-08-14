# 2nd - Place Solution Team DIT

This repository contains the source code for the 2nd place solution in the Kaggle `Google Research - Identify Contrails to Reduce Global Warming` developed by `DrHB`, `Iafoss` and `Theo Viel`. For more technical write up read [here](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/430491). To reproduce our results, please follow the instructions provided below.


# Installation
We recommend using the official `nvidia` `or` kaggle Docker images with the appropriate CUDA version for the best compatibility. 



## Data Download and Preparation

You can obtain the official competition data from the Kaggle website. After downloading, please move the data to the `data` directory. We also recommend to dowlad external data that can be found [here](https://www.kaggle.com/datasets/iafoss/identify-contrails-external)

### Preparing the Data
To process the data, execute the following command:
```bash
python prepare_data.py
```

Upon running the command, four new directories should be generated within the data folder: `train_adj2`, `val_adj2`, `train_adj2single`, and `val_adj2single`.

## Configuration Documentation

The `config.json` file contains various settings and parameters for the training process. Below is a brief explanation of each parameter:

| Parameter      | Value                   | Description                                |
| -------------- | ----------------------- | ------------------------------------------ |
| `OUT`          | `BASELINE`              | Output folder name                         |
| `PATH`         | `data/`                 | Path to the data folder                    |
| `NUM_WORKERS`  | `4`                     | Number of worker threads for data loading  |
| `SEED`         | `2023`                  | Random seed value for reproducibility      |
| `BS`           | `32`                    | Batch size for training                    |
| `BS_VALID`     | `32`                    | Batch size for validation                  |
| `LR_MAX`       | `5e-4`                  | Max learning rate                          |
| `PCT_START`    | `0.1`                   | Learning rate schedule decay `%` start     |
| `WEIGHTS`      | `false`                 | Model weights to load         |
| `LOSS_FUNC`    | `{loss_comb}` | Loss function                              |
| `METRIC`       | `F_th`                  | Evaluation metric for model selection      |



## Training
To train a model, please download the required pretrained weights from the official repository for Coat, NextVit, and SAM. When running the script below, add the argument `WEIGHTS` followed by the path to the downloaded weights.

### NeXtViT_ULSTM

To train a 5-frame sequential model using the nextVIT encoder.

```bash
train.py config.json \
       MODEL NeXtViT_ULSTM \
       OUT experiments \
       FNAME Seq_NextViT_512_0 \
       LR_MAX 3.5e-4 \
       LOSS_FUNC loss_comb \
       SEED 2023 \
       FOLD 0 \
       BS 8 \
       EPOCH 24 \
```
For our competition, we trained the model five times using different seeds.


### CoaT_ULSTM

To train a 5-frame sequential model using the CoaT encoder.

```bash
train.py config.json \
       MODEL CoaT_ULSTM \
       OUT experiments \
       FNAME Seq_CoaT_512 \
       LR_MAX 3.5e-4 \
       LOSS_FUNC loss_comb \
       SEED 2023 \
       FOLD 0 \
       BS 8 \
       EPOCH 24 \
```
For our competition, we trained the model five times using different seeds.


### CoaT_UT

To train a single frame model using the CoaT encoder.

```bash
train.py config.json \
       MODEL CoaT_UT \
       OUT experiments \
       FNAME CoaT_UT \
       LR_MAX 3.5e-4 \
       LOSS_FUNC loss_comb \
       SEED 2023 \
       FOLD 0 \
       BS 8 \
       EPOCH 24 \
```
For our competition, we trained the model five times using different seeds.


### SAM 
For `SAM`, we trained five distinct models. Each model varies slightly in terms of feature fusion. When running the script below, replace `MODEL` with one of the following options: {`SAM_U`, `SAM_USA`, `SAM_UV1`, `SAM_UV2`, `SAM_UV3`}. It's also advisable to choose with different `seeds` per seed. The training process for each `model` is divided into `three` stages:

`Stage 1`: Train for 24 epochs using a learning rate of `3.5e-4`.
After `Stage 1`, reload the weights and train for an additional `12` epochs at a `learning rate` of `3.5e-5`.
Finally, train for `12` more epochs with a learning rate of `3.5e-6`. Example script for training `SAM_U` is shown below:

```bash
train.py config.json \
       MODEL SAM_U \
       OUT experiments \
       FNAME samu \
       LR_MAX 3.5e-4 \
       LOSS_FUNC loss_comb \
       SEED 2023 \
       FOLD 0 \
       BS 8 \
       EPOCH 24 \
```



## Inference 

Our inference scripts, models, and kernels are publicly available and can be found [here](https://www.kaggle.com/code/theoviel/contrails-inference-comb?scriptVersionId=139316588)