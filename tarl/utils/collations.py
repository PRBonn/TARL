import numpy as np
import MinkowskiEngine as ME
import torch
import torch.nn.functional as F

def pad_batch(coors, feats):
    """
    From a list of features create a batched tensors with 
    features padded to the max number of samples in the batch.

    returns: 
        feats: batched feature Tensors (features for each sample)
        coors: batched coordinate Tensors
        pad_masks: batched bool Tensors indicating where the padding was performed
    """
    sh = [f.shape[0] for f in feats]
    coors = torch.stack([F.pad(c, (0, 0, 0, max(sh) - c.shape[0])) for c in coors])
    pad_masks = torch.stack([F.pad(torch.zeros_like(f[:, 0]), (0, max(sh) - f.shape[0]), value=1).bool() for f in feats])
    feats = torch.stack([F.pad(f, (0, 0, 0, max(sh) - f.shape[0])) for f in feats])
    return coors, feats, pad_masks

def sample_list_seg(seg, p_coord, p_feats, sample_points, avg_feats):
    if not avg_feats:
        np.random.shuffle(seg)
        seg = seg[:sample_points]

    coord = p_coord[seg]
    feats = p_feats[seg].mean(0) if avg_feats else p_feats[seg]

    return coord, feats

def list_segments_points(p_coord, p_feats, labels, sample_points, avg_feats=False):
    # labels are batch_id segs_id and we now concatenate the point index (the same as the coords and feats)
    idx_labels = np.concatenate((labels, np.arange(len(labels))[:,None]), axis=-1)

    # we hash the segs_id to be unique over all the batch and the remove the ground points
    idx_labels = idx_labels[idx_labels[:,1] != -1]
    idx_labels[:,1] = idx_labels[:,0] * 10000 + idx_labels[:,1]

    # remove the batch_id because we dont need this anymore after hashing
    idx_hash_labels = idx_labels[:,1:]

    # sort to "group" together the points belonging to the same segment
    idx_hash_labels = idx_hash_labels[idx_hash_labels[:,0].argsort()]
    idx_labels = np.split(idx_hash_labels[:,1], np.unique(idx_hash_labels[:, 0], return_index=True)[1][1:])
    idx_sample_labels = [ sample_list_seg(seg, p_coord, p_feats, sample_points, avg_feats) for seg in idx_labels ]

    seg_coord, seg_feats = list(zip(*idx_sample_labels))

    return (seg_coord, seg_feats)

def list_segments_points_slow(p_coord, p_feats, labels, sample_points, avg_feats=False):
    c_coord = []
    c_feats = []

    seg_batch_count = 0

    for batch_num in range(labels.shape[0]):
        for segment_lbl in np.unique(labels[batch_num]):
            if segment_lbl == -1:
                continue

            batch_ind = p_coord[:,0] == batch_num
            segment_ind = labels[batch_num] == segment_lbl

            # we are listing from sparse tensor, the first column is the batch index, which we drop
            segment_coord = p_coord[batch_ind][segment_ind[:,0]][:,:]
            segment_coord[:,0] = seg_batch_count
            segment_coord = torch.cat((batch_num*torch.ones_like(segment_coord[:,0]).unsqueeze(axis=-1), segment_coord), axis=-1)
            seg_batch_count += 1

            segment_feats = p_feats[batch_ind][segment_ind[:,0]]

            sampling = np.arange(len(segment_coord))
            # if len(sampling) > sample_points:
            #     sampling = np.random.choice(sampling, sample_points, replace=False)

            c_coord.append(segment_coord[sampling])
            c_feats.append(segment_feats[sampling] if not avg_feats else segment_feats.mean(0))

    seg_coord = c_coord
    seg_feats = c_feats

    return (seg_coord, seg_feats)


def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(p_coord, dtype=torch.float32)
    p_feats = torch.vstack(p_feats).float()

    if p_label is not None:
        # we batch the segs id to later have unique labels per point
        p_label = ME.utils.batched_coordinates(p_label, device=torch.device('cpu')).numpy()
    
        return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            ), p_label

    return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            )

class SparseAugmentedCollation:
    def __init__(self, resolution):
        return

    def __call__(self, data):
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        pt, ptn = list(zip(*data))

        # transform batch((coord, feats, segs)) to batch(coord), batch(feats), batch(segs)
        pt_coord, pt_feats, segment_t, frame_t = list(zip(*pt))
        ptn_coord, ptn_feats, segment_tn, frame_tn = list(zip(*ptn))

        # if not segment_contrast segment_t and segment_tn will be an empty list
        return {
                'pcd_t': {'coord': pt_coord, 'feats': pt_feats, 'segs': segment_t, 't_frame': frame_t},
                'pcd_tn': {'coord': ptn_coord, 'feats': ptn_feats, 'segs': segment_tn, 't_frame': frame_tn},
            }
