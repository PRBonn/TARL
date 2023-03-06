# Training on you data

We have implemented the pre-training for SemanticKITTI, however the pre-training can be applied to different datasets.
Your data should have only the LiDAR point clouds and the scan-wise poses, to aggregate the point clouds. In case poses are not available, you can use
[kiss-icp](https://github.com/PRBonn/kiss-icp) to predict the poses. To extract the temporal views, you would need to tune the parameters for the
[ground segmentation](https://github.com/LimHyungTae/patchwork) and the hdbscan, to check that the segments correspond to the objects. Note that,
in case you face problems using the patchwork method, in this [repo](https://github.com/url-kaist/patchwork-plusplus) they have a pybind version which should
make it easier to use, or you could use RANSAC implementation from open3d as done in [SegContrast](https://github.com/PRBonn/segcontrast).

## Dataloader

In `tarl/datasets/dataloader/DataloaderTemplate.py` there is a template for implementing the dataloader for you own data, hopefully it has enough comments
and instructions. After implementing your dataloader, be aware to also add it to `tarl/datasets/dataloader/datasets.py`.
