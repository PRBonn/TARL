# TARL: Temporal Consistent 3D LiDAR Representation Learning for Semantic Perception in Autonomous Driving

This repo contains the code for the self-supervised pre-training method proposed in the CVPR'23 paper: Temporal Consistent 3D LiDAR Representation Learning for Semantic Perception in Autonomous Driving.

Our approach extract temporal views as augmented versions of the same object. We aggregate sequential LiDAR scans, and by removing the ground (in an unsupervised manner) and clustering the remaining points we define coarse segments of objects in the scene to be used for self-supervised pre-training. We evaluate our pre-training by fine-tuning the pre-trained model to different downstream tasks. In our experiments we show that our approach could significantly reduce the amount of labels needed to achieve the same performance as the network trained from scratch using the full training set.
