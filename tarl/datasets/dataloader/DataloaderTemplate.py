import torch
from torch.utils.data import Dataset
from tarl.utils.pcd_preprocess import clusterize_pcd, visualize_pcd_clusters, point_set_to_coord_feats, overlap_clusters, aggregate_pcds
from tarl.utils.pcd_transforms import *
from tarl.utils.data_map import learning_map
import os
import numpy as np

import warnings

warnings.filterwarnings('ignore')

#################################################
################## Data loader ##################
#################################################

class TemplateSet(Dataset):
    def __init__(self, data_dir, scan_window, seqs, split, pre_training, resolution, percentage, intensity_channel, num_points=80000):
        super().__init__()
        self.data_dir = data_dir
        self.augmented_dir = 'segments_views'

        self.n_clusters = 50
        self.resolution = resolution
        self.intensity_channel = intensity_channel
        self.scan_window = scan_window
        self.num_points = num_points
        # we divide the scan window in begin, middle, end, we sample the pair of scans from the begin and end
        # and the next batch starts at the middle
        # e.g.: [0,1,2,3,4,5,6,7,8] pairs in sampled between [0,1,2] and [6,7,8] and the next sample starts in 3 (i.e. scan_window = [3,4,5,6,7,8,9,10,11])
        self.sampling_window = int(np.floor(scan_window / 3))

        self.split = split
        self.seqs = seqs

        self.datapath_pretrain()
        self.nr_data = len(self.points_datapath)

        print('The size of %s data is %d'%(self.split,len(self.points_datapath)))

    def datapath_pretrain(self):
        self.points_datapath = []

        # Load here you scans from data_dir
        scans = os.listdir(self.data_dir)

        for file_num in range(0, len(scans), self.sampling_window):
            end_file = file_num + self.scan_window if len(scans) - file_num > self.scan_window else len(scans)
            self.points_datapath.append([scan_file for scan_file in scans[file_num:end_file] ])
            if end_file == len(scans):
                break

    def transforms(self, points):
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])
        points[:,:,:3] = jitter_point_cloud(points[:,:,:3])
        points = random_drop_n_cuboids(points)

        return np.squeeze(points, axis=0)

    def _getitem__(self, index):
        # get filename
        fname = self.points_datapath[index][0].split('/')[-1].split('.')[0]

        # cluster the aggregated pcd and save the result
        cluster_path = os.path.join(self.data_dir, 'assets', self.augmented_dir)
        if os.path.isfile(os.path.join(cluster_path, fname + '.seg')):
            # if saved just load it
            segments = np.fromfile(os.path.join(cluster_path, fname + '.seg'), dtype=np.float16)
            segments = segments.reshape((-1, 1))
        else:
            # if not generate segments and save it
            points_set, ground_label, parse_idx = aggregate_pcds(self.points_datapath[index], self.data_dir)
            segments = clusterize_pcd(points_set, ground_label)
            segments[parse_idx] = -np.inf
            assert (segments.max() < np.finfo('float16').max), 'max segment id overflow float16 number'
            if not os.path.isdir(cluster_path):
                os.makedirs(cluster_path)
            segments = segments.astype(np.float16)
            segments.tofile(os.path.join(cluster_path, fname + '.seg'))

        # if it is a normal sample (sample with self.scan_window scans) sample one pcd from begining and one from end
        if len(self.points_datapath[index]) == self.scan_window:
            t_frames = np.random.choice(np.arange(self.sampling_window), 2, replace=False)
            t_frames[1] = 2*self.sampling_window + t_frames[1]
        # if it is the last sample from the seq (less samples than self.scan_window) sample two random scans
        else:
            t_frames = np.random.choice(np.arange(len(self.points_datapath[index])), 2, replace=False)

        # get start position of each aggregated pcd
        pcd_parse_idx = np.unique(np.argwhere(segments == -np.inf)[:,0])

        # get the delimiter position and access the next index (an actual point and not -infinite)
        p1 = np.fromfile(self.points_datapath[index][t_frames[0]], dtype=np.float32)
        p1 = p1.reshape((-1, 4))
        s1 = segments[pcd_parse_idx[t_frames[0]]+1:pcd_parse_idx[t_frames[0]+1]]
        # concatenate after parsing the t_frame from the point cloud and the segments generated
        ps1 = np.concatenate((p1, s1), axis=-1)
        # random augmentation
        ps1 = self.transforms(ps1)
        p1 = ps1[:,:-1]
        s1 = ps1[:,-1][:,np.newaxis]

        # get the delimiter position and access the next index (an actual point and not -infinite)
        p2 = np.fromfile(self.points_datapath[index][t_frames[1]], dtype=np.float32)
        p2 = p2.reshape((-1, 4))
        s2 = segments[pcd_parse_idx[t_frames[1]]+1:pcd_parse_idx[t_frames[1]+1]]
        # concatenate after parsing the t_frame from the point cloud and the segments generated
        ps2 = np.concatenate((p2, s2), axis=-1)
        # random augmentation
        ps2 = self.transforms(ps2)
        p2 = ps2[:,:-1]
        s2 = ps2[:,-1][:,np.newaxis]

        ############## TO VISUALIZE ##############
        # p = np.concatenate((p1, p2))
        # s = np.concatenate((s1, s2))
        # visualize_pcd_clusters(p, s)
        ##########################################

        # Here the quantization for the voxels happens in case you don't need voxels just remove those two lines
        coord_t, feats_t, cluster_t = point_set_to_coord_feats(p1, s1, self.resolution, self.num_points)
        coord_tn, feats_tn, cluster_tn = point_set_to_coord_feats(p2, s2, self.resolution, self.num_points)

        # guarantee that only overlapping segments will be in the t_frames randomly selected
        cluster_t, cluster_tn = overlap_clusters(cluster_t, cluster_tn)

        return (torch.from_numpy(coord_t), torch.from_numpy(feats_t), cluster_t, t_frames[0]), \
                (torch.from_numpy(coord_tn), torch.from_numpy(feats_tn), cluster_tn, t_frames[1])

    def __len__(self):
        #print('DATA SIZE: ', np.floor(self.nr_data / self.sampling_window), self.nr_data % self.sampling_window)
        return self.nr_data

##################################################################################################
