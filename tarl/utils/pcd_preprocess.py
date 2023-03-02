import numpy as np
import open3d as o3d
import hdbscan
import matplotlib.pyplot as plt
import MinkowskiEngine as ME
import os

def overlap_clusters(cluster_i, cluster_j, min_cluster_point=10):
    # get unique labels from pcd_i and pcd_j from segments bigger than min_clsuter_point
    unique_i, count_i = np.unique(cluster_i, return_counts=True)
    unique_i = unique_i[count_i > min_cluster_point]

    unique_j, count_j = np.unique(cluster_j, return_counts=True)
    unique_j = unique_j[count_j > min_cluster_point]

    # get labels present on both pcd (intersection)
    unique_ij = np.intersect1d(unique_i, unique_j)[1:]
        
    # labels not intersecting both pcd are assigned as -1 (unlabeled)
    cluster_i[np.in1d(cluster_i, unique_ij, invert=True)] = -1
    cluster_j[np.in1d(cluster_j, unique_ij, invert=True)] = -1

    return cluster_i, cluster_j

def parse_calibration(filename):
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib

def load_poses(calib_fname, poses_fname):
    calibration = parse_calibration(calib_fname)
    poses_file = open(poses_fname)

    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)

    poses = []

    for line in poses_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses

def apply_transform(points, pose):
    hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
    return np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)[:,:3]

def undo_transform(points, pose):
    hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
    return np.sum(np.expand_dims(hpoints, 2) * np.linalg.inv(pose).T, axis=1)[:,:3]

def aggregate_pcds(data_batch, data_dir):
    # load empty pcd point cloud to aggregate
    points_set = np.empty((0,4))

    # define a delimiter to divide the aggregated pcd
    p_delimiter = np.asarray([[-np.inf, -np.inf, -np.inf, -np.inf]])
    ground_label = np.empty((0,1))
    g_delimiter = np.asarray([[-np.inf]])

    # define "namespace"
    seq_num = data_batch[0].split('/')[-3]
    fname = data_batch[0].split('/')[-1].split('.')[0]

    # load poses
    datapath = data_batch[0].split('velodyne')[0]
    poses = load_poses(os.path.join(datapath, 'calib.txt'), os.path.join(datapath, 'poses.txt'))
    
    for t in range(len(data_batch)):
        # load the next t scan and aggregate
        gname = data_batch[t].split('/')[-1].split('.')[0]
        g_set = np.fromfile(os.path.join(data_dir, 'assets', 'patchwork', seq_num, gname + '.label'), dtype=np.uint32)
        g_set = g_set.reshape((-1))[:, np.newaxis]
        # aggregate a delimiter and the next scan
        ground_label = np.vstack([ground_label, g_delimiter, g_set])

        # load the next t scan, apply pose and aggregate
        p_set = np.fromfile(data_batch[t], dtype=np.float32)
        p_set = p_set.reshape((-1, 4))
        pose_idx = int(gname)
        p_set[:,:3] = apply_transform(p_set[:,:3], poses[pose_idx])
        # aggregate a delimiter and the next scan
        points_set = np.vstack([points_set, p_delimiter, p_set])

    ground_label = np.vstack([ground_label, g_delimiter])
    points_set = np.vstack([points_set, p_delimiter])

    # get start position of each aggregated pcd
    pcd_parse_idx = np.unique(np.argwhere(ground_label == g_delimiter)[:,0])

    pose_idx = int(fname)
    points_set[:,:3] = undo_transform(points_set[:,:3], poses[pose_idx])
    points_set[pcd_parse_idx] = p_delimiter

    return points_set, ground_label, pcd_parse_idx

def clusters_hdbscan(points_set, n_clusters=50):
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1., approx_min_span_tree=True,
                                gen_min_span_tree=True, leaf_size=100,
                                metric='euclidean', min_cluster_size=20, min_samples=None
                            )

    clusterer.fit(points_set)

    labels = clusterer.labels_.copy()

    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:,1].argsort()]

    clusters_labels = cluster_info[::-1][:n_clusters, 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1

    return labels

def clusterize_pcd(points, ground):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # instead of ransac use patchwork
    inliers = list(np.where(ground == 9)[0])

    pcd_ = pcd.select_by_index(inliers, invert=True)
    labels_ = np.expand_dims(clusters_hdbscan(np.asarray(pcd_.points)), axis=-1)

    labels = np.ones((points.shape[0], 1)) * -1
    mask = np.ones(labels.shape[0], dtype=bool)
    mask[inliers] = False

    labels[mask] = labels_

    return labels

def point_set_to_coord_feats(point_set, labels, resolution, num_points, deterministic=False):
    p_feats = point_set.copy()
    p_coord = np.round(point_set[:, :3] / resolution)
    p_coord -= p_coord.min(0, keepdims=1)

    _, mapping = ME.utils.sparse_quantize(coordinates=p_coord, return_index=True)
    if len(mapping) > num_points:
        np.random.seed(42)
        mapping = np.random.choice(mapping, num_points, replace=False)

    return p_coord[mapping], p_feats[mapping], labels[mapping]

def visualize_pcd_clusters(points, labels):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    
    colors = np.zeros((len(labels), 4))
    flat_indices = np.unique(labels[:,-1])
    max_instance = len(flat_indices)
    colors_instance = plt.get_cmap("prism")(np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))

    for idx in range(len(flat_indices)):
        colors[labels[:,-1] == flat_indices[int(idx)]] = colors_instance[int(idx)]

    colors[labels[:,-1] == -1] = [0.,0.,0.,0.]

    pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])

    o3d.visualization.draw_geometries([pcd])
