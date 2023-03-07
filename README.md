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

---

# Fine-tuning

For fine-tuning we have used repositories from the baselines, so after pre-training with TARL you should copy the pre-trained weights to the target task and use it for fine-tuning.

## Semantic segmentation

For fine-tuning to semantic segmentation we refer to the SegContrast [repo](https://github.com/PRBonn/segcontrast).
Clone the repo with `git clone https://github.com/PRBonn/segcontrast.git` and follow the installation instructions. Note that the requirements from
TARL and segcontrast are similar since both use `MinkowskiEngine` so you should be able to use the same environment than TARL just installing
the remaining packages missing.

After setting up the packages, copy the pre-trained model from `TARL/tarl/experiments/TARL/default/version_0/checkpoints/last.ckpt` to `segcontrast/checkpoint/contrastive/lastepoch199_model_tarl.pt` and run the following command:

```python3 downstream_train.py --use-cuda --use-intensity --checkpoint \
        tarl --contrastive --load-checkpoint --batch-size 2 \
        --sparse-model MinkUNet --epochs 15```

# Object detection

For object detection we have used the OpenPCDet [repo](https://github.com/zaiweizhang/OpenPCDet) with few modifications. In this docker [image](https://hub.docker.com/r/nuneslu/segcontrast_openpcdet) we have setted up everything to run it with `MinkUNet` and to load our pre-trained weights.
The weights should be copied to `/tmp/OpenPCDet/pretrained/lastepoch199_model_tarl.pt` inside the container and then running the command:

```
cd /tmp/OpenPCDet/tools
python3 train.py --cfg_file cfgs/kitti_models/tarl_pretrained.yaml --pretrained_model ../pretrained/lastepoch199_model_tarl.pt
```
