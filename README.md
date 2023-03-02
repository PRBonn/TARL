# TARL: Temporal Consistent 3D LiDAR Representation Learning for Semantic Perception in Autonomous Driving

This repo contains the code for the self-supervised pre-training method proposed in the CVPR'23 paper: Temporal Consistent 3D LiDAR Representation Learning for Semantic Perception in Autonomous Driving.

Our approach extract temporal views as augmented versions of the same object. We aggregate sequential LiDAR scans, and by removing the ground (in an unsupervised manner) and clustering the remaining points we define coarse segments of objects in the scene to be used for self-supervised pre-training. We evaluate our pre-training by fine-tuning the pre-trained model to different downstream tasks. In our experiments we show that our approach could significantly reduce the amount of labels needed to achieve the same performance as the network trained from scratch using the full training set.

## Dependencies

To run our code first install the dependencies with:

```
sudo apt install build-essential python3-dev libopenblas-dev
pip3 install -r requirements.txt
```

Followed by installing MinkowskiEngine from the official repo:

```
pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine --install-option="--blas=openblas" -v --no-deps
```

Then you need to run the code setup with:

`pip3 install -U -e .`

## SemanticKITTI Dataset

The SemanticKITTI dataset has to be download from the official [site](http://www.semantic-kitti.org/dataset.html#download) and extracted in the following structure:

```
./
└── Datasets/
    └── SemanticKITTI
        └── dataset
          └── sequences
            ├── 00/
            │   ├── velodyne/
            |   |       ├── 000000.bin
            |   |       ├── 000001.bin
            |   |       └── ...
            │   └── labels/
            |       ├── 000000.label
            |       ├── 000001.label
            |       └── ...
            ├── 08/ # for validation
            ├── 11/ # 11-21 for testing
            └── 21/
                └── ...
```

For the unsupervised ground segmentation, you need to run [patchwork](https://github.com/LimHyungTae/patchwork) over the SemanticKITTI dataset and put the generated files over:
```
./
└── Datasets/
    └── SemanticKITTI
        └── assets
            └── patchwork
                ├── 08
                    ├── 000000.label
                    ├── 000001.label
                    └── ...
```

## Running the code

The command to run the pre-training is:

```
python3 tarl_train.py
```

In the `config/config.yaml` the parameters used in our experiments are already set.
