import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tarl.models.minkunet as minknet
from tarl.models.blocks import TransformerProjector, TemperatureCosineSim, PositionalEncoder
from tarl.utils.pcd_preprocess import visualize_pcd_clusters
from utils.collations import numpy_to_sparse_tensor, list_segments_points, pad_batch
import MinkowskiEngine as ME
import open3d as o3d
import matplotlib.pyplot as plt

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule

class TARLTrainer(LightningModule):
    def __init__(self, cfg: dict, data_module: LightningDataModule = None):
        super().__init__()
        # name you hyperparameter hparams, then it will be saved automagically.
        self.save_hyperparameters(cfg)
        self.data_module = data_module

        self.model_q = minknet.MinkUNet(in_channels=4 if self.hparams['data']['intensity'] else 3, out_channels=self.hparams['model']['out_dim'])
        self.proj_head_q = TransformerProjector(d_model=self.hparams['model']['out_dim'], num_layer=1)
        self.model_k = minknet.MinkUNet(in_channels=4 if self.hparams['data']['intensity'] else 3, out_channels=self.hparams['model']['out_dim'])
        self.proj_head_k = TransformerProjector(d_model=self.hparams['model']['out_dim'], num_layer=1)

        # initialize model k and q
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.proj_head_q.parameters(), self.proj_head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.pos_enc = PositionalEncoder()
        self.predictor = TransformerProjector(d_model=self.hparams['model']['out_dim'], num_layer=1)
        self.cos_sim = TemperatureCosineSim()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.hparams['train']['momentum'] + param_q.data * (1. - self.hparams['train']['momentum'])

        for param_q, param_k in zip(self.proj_head_q.parameters(), self.proj_head_k.parameters()):
            param_k.data = param_k.data * self.hparams['train']['momentum'] + param_q.data * (1. - self.hparams['train']['momentum'])

    def save_backbone(self):
        state = {
            'model': self.model_q.state_dict(),
            'epoch': self.current_epoch,
            'params': self.hparams,
        }

        torch.save(state, 'lastepoch199_model_tarl.pt')


    def forward(self, x_t:ME.SparseTensor, s_t, x_tn:ME.SparseTensor, s_tn):
        # compute point-wise features
        h_t = self.model_q(x_t)
        # list the segments and its point-wise features
        (h_tc, h_tf) = list_segments_points(h_t.C, h_t.F, s_t, self.hparams['train']['sample_points'])

        h_tc, h_tf, pad_masks = pad_batch(h_tc, h_tf)

        # projection head over the point-wise features for each segment
        z_t = self.proj_head_q(h_tf)
        # predictor over the projected fetures
        z_t_tn = self.predictor(z_t)
        z_t_tn = nn.functional.normalize(z_t_tn, dim=-1)

        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder

            # compute point-wise features
            h_tn = self.model_k(x_tn)
            # list segments and the mean over its point-wise features
            (h_tnc, h_tnf) = list_segments_points(h_tn.C, h_tn.F, s_tn, self.hparams['train']['sample_points'], avg_feats=True)

            # h_tnf shape is num_clusters x feat_dim
            h_tnf = torch.vstack(h_tnf)

            # projection head over the segment-wise mean representation
            z_tn = self.proj_head_k(torch.unsqueeze(h_tnf, dim=0))
            z_tn = nn.functional.normalize(z_tn, dim=-1)

        # compute attention (similarity) between points and segs means
        temporal_attn = self.cos_sim(z_t_tn, z_tn, z_tn, self.hparams['train']['tau'])
        pred_attn = temporal_attn[1]

        # define the target segments for each point
        target_attn = torch.arange(pred_attn.shape[0], device=self.device).unsqueeze(-1).expand(pred_attn.shape[0], pred_attn.shape[1])

        #####################################################################
        # compute the positives and negatives attentions values for logging
        #####################################################################
        target_idx = [*range(len(pred_attn))]
        target_idx_ = torch.zeros_like(pred_attn)
        target_idx_[target_idx, :, target_idx] = 1.
        target_idx = target_idx_.bool()
        avg_attn = pred_attn.softmax(-1)
        avg_attn_pos = avg_attn[target_idx].mean().item()
        avg_attn_neg = avg_attn[~target_idx].mean().item()

        self.log('mean_pos_attn', avg_attn_pos)
        self.log('mean_neg_attn', avg_attn_neg)
        #####################################################################

        # remove the padded values (zeros) to compute the loss
        loss = F.cross_entropy(pred_attn[~pad_masks], target_attn[~pad_masks])

        return loss

    def training_step(self, batch:dict, batch_idx):
        torch.cuda.empty_cache()

        # these two lines gets the coords and feats and turn them into a SparseTensor
        # if you are using a different backbone then MinkUNet replace it by your voxelization function
        # or any other data preparation scheme
        x_t, segs_t = numpy_to_sparse_tensor(batch['pcd_t']['coord'], batch['pcd_t']['feats'], batch['pcd_t']['segs'])
        x_tn, segs_tn = numpy_to_sparse_tensor(batch['pcd_tn']['coord'], batch['pcd_tn']['feats'], batch['pcd_tn']['segs'])

        loss = self.forward(x_t, segs_t, x_tn, segs_tn)
        torch.cuda.empty_cache()
        loss += self.forward(x_tn, segs_tn, x_t, segs_t)

        self.log('loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams['train']['lr'])
        self.optimizer = optimizer

        return optimizer

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    # def val_dataloader(self):
    #     return self.data_module.val_dataloader(batch_size=self.hparams['batch_size'])

    # def test_dataloader(self):
    #     return self.data_module.test_dataloader(batch_size=self.hparams['batch_size'])
