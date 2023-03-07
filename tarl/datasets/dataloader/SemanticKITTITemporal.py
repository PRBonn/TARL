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

class TemporalKITTISet(Dataset):
    def __init__(self, data_dir, scan_window, seqs, split, resolution, percentage, intensity_channel, use_ground_pred=True, num_points=80000):
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
        self.use_ground_pred = use_ground_pred

        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_pretrain()
        self.nr_data = len(self.points_datapath)

        print('The size of %s data is %d'%(self.split,len(self.points_datapath)))

    def datapath_pretrain(self):
        self.points_datapath = []

        for seq in self.seqs:
            point_seq_path = os.path.join(self.data_dir, 'dataset', 'sequences', seq, 'velodyne')
            point_seq_bin = os.listdir(point_seq_path)
            point_seq_bin.sort()
            
            for file_num in range(0, len(point_seq_bin), self.sampling_window):
                end_file = file_num + self.scan_window if len(point_seq_bin) - file_num > self.scan_window else len(point_seq_bin)
                self.points_datapath.append([os.path.join(point_seq_path, point_file) for point_file in point_seq_bin[file_num:end_file] ])
                if end_file == len(point_seq_bin):
                    break

        #self.points_datapath = self.points_datapath[:10]

    def transforms(self, points):
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])
        points[:,:,:3] = jitter_point_cloud(points[:,:,:3])
        points = random_drop_n_cuboids(points)

        return np.squeeze(points, axis=0)

    def datapath_list(self):
        self.points_datapath = []
        self.labels_datapath = []

        for seq in self.seqs:
            point_seq_path = os.path.join(self.data_dir, 'dataset', 'sequences', seq, 'velodyne')
            point_seq_bin = os.listdir(point_seq_path)
            point_seq_bin.sort()
            self.points_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]

            label_seq_path = os.path.join(self.data_dir, 'dataset', 'sequences', seq, 'labels')
            point_seq_label = os.listdir(label_seq_path)
            point_seq_label.sort()
            self.labels_datapath += [ os.path.join(label_seq_path, label_file) for label_file in point_seq_label ]

    def __getitem__(self, index):
        # define "namespace"
        seq_num = self.points_datapath[index][0].split('/')[-3]
        fname = self.points_datapath[index][0].split('/')[-1].split('.')[0]

        # cluster the aggregated pcd and save the result
        cluster_path = os.path.join(self.data_dir, 'assets', self.augmented_dir, seq_num)
        if os.path.isfile(os.path.join(cluster_path, fname + '.seg')):
            segments = np.fromfile(os.path.join(cluster_path, fname + '.seg'), dtype=np.float16)
            segments = segments.reshape((-1, 1))
        else:
            points_set, ground_label, parse_idx = aggregate_pcds(self.points_datapath[index], self.data_dir, self.use_ground_pred)
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
        # if it is the last sample from the seq (less samples than self.scan_window) sample two random scans (w/o begin-end restriction)
        else:
            t_frames = np.random.choice(np.arange(len(self.points_datapath[index])), 2, replace=False)

        # get start position of each aggregated pcd
        pcd_parse_idx = np.unique(np.argwhere(segments == -np.inf)[:,0])

        # get the delimiter position and access the next index (an actual point and not -infinite)
        p1 = np.fromfile(self.points_datapath[index][t_frames[0]], dtype=np.float32)
        p1 = p1.reshape((-1, 4))
        s1 = segments[pcd_parse_idx[t_frames[0]]+1:pcd_parse_idx[t_frames[0]+1]]
        ps1 = np.concatenate((p1, s1), axis=-1)
        ps1 = self.transforms(ps1)
        p1 = ps1[:,:-1]
        s1 = ps1[:,-1][:,np.newaxis]

        # get the delimiter position and access the next index (an actual point and not -infinite)
        p2 = np.fromfile(self.points_datapath[index][t_frames[1]], dtype=np.float32)
        p2 = p2.reshape((-1, 4))
        s2 = segments[pcd_parse_idx[t_frames[1]]+1:pcd_parse_idx[t_frames[1]+1]]
        ps2 = np.concatenate((p2, s2), axis=-1)
        ps2 = self.transforms(ps2)
        p2 = ps2[:,:-1]
        s2 = ps2[:,-1][:,np.newaxis]

        ############## TO VISUALIZE ##############
        # p = np.concatenate((p1, p2))
        # s = np.concatenate((s1, s2))
        # visualize_pcd_clusters(p, s)
        ##########################################

        # we voxelize it here to avoid having a for loop during the collation
        coord_t, feats_t, cluster_t = point_set_to_coord_feats(p1, s1, self.resolution, self.num_points)
        coord_tn, feats_tn, cluster_tn = point_set_to_coord_feats(p2, s2, self.resolution, self.num_points)

        cluster_t, cluster_tn = overlap_clusters(cluster_t, cluster_tn)

        return (torch.from_numpy(coord_t), torch.from_numpy(feats_t), cluster_t, t_frames[0]), \
                (torch.from_numpy(coord_tn), torch.from_numpy(feats_tn), cluster_tn, t_frames[1])

    def __len__(self):
        #print('DATA SIZE: ', np.floor(self.nr_data / self.sampling_window), self.nr_data % self.sampling_window)
        return self.nr_data

##################################################################################################
