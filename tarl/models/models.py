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
        # self.model.load_state_dict(checkpoint['model'])

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
        #self.cross_attn = nn.MultiheadAttention(self.hparams['model']['out_dim'], 1, batch_first=True)

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
        h_t = self.model_q(x_t)
        # compute positional encoding
        # h_t_pos_embed = self.pos_enc(x_t.F[:,:-1].clone())
        (h_tc, h_tf) = list_segments_points(h_t.C, h_t.F, s_t, self.hparams['train']['sample_points'])

        h_tc, h_tf, pad_masks = pad_batch(h_tc, h_tf)

        z_t = self.proj_head_q(h_tf)
        z_t_tn = self.predictor(z_t)
        z_t_tn = nn.functional.normalize(z_t_tn, dim=-1)

        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder

            h_tn = self.model_k(x_tn)
            # compute positional encoding
            # h_tn_pos_embed = self.pos_enc(x_tn.F[:,:-1].clone())
            (h_tnc, h_tnf) = list_segments_points(h_tn.C, h_tn.F, s_tn, self.hparams['train']['sample_points'], avg_feats=True)

            # h_tnf shape is num_clusters x feat_dim
            h_tnf = torch.vstack(h_tnf)

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
        # batch['pcd_tn']['coord'][0][:,2] += 500.
        # p = np.concatenate((batch['pcd_t']['coord'][0], batch['pcd_tn']['coord'][0]))
        # s = np.concatenate((batch['pcd_t']['segs'][0], batch['pcd_tn']['segs'][0]))
        # visualize_pcd_clusters(p, s)
        torch.cuda.empty_cache()
        x_t, segs_t = numpy_to_sparse_tensor(batch['pcd_t']['coord'], batch['pcd_t']['feats'], batch['pcd_t']['segs'])
        x_tn, segs_tn = numpy_to_sparse_tensor(batch['pcd_tn']['coord'], batch['pcd_tn']['feats'], batch['pcd_tn']['segs'])
        loss = self.forward(x_t, segs_t, x_tn, segs_tn)
        torch.cuda.empty_cache()
        loss += self.forward(x_tn, segs_tn, x_t, segs_t)

        self.log('loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams['train']['lr'])
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams['train']['lr'], momentum=0.9, weight_decay=self.hparams['train']['decay_lr'], nesterov=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams['train']['max_epoch'], eta_min=self.hparams['train']['lr'] / 1000)

        #self.scheduler = scheduler
        self.optimizer = optimizer

        return optimizer#[optimizer], [scheduler]

    def visualize_attention(self, coord_t, coord_tn, attn):
        pcd_t = o3d.geometry.PointCloud()
        coord_t -= coord_t.min()
        pcd_t.points = o3d.utility.Vector3dVector(coord_t)

        pcd_tn = o3d.geometry.PointCloud()
        coord_tn -= coord_tn.min()
        coord_tn[:,0] += 2
        pcd_tn.points = o3d.utility.Vector3dVector(coord_tn)
        colors_tn = np.zeros_like(coord_tn)
        pcd_tn.colors = o3d.utility.Vector3dVector(colors_tn)

        color_t = plt.get_cmap('viridis')(attn)

        pcd_t.colors = o3d.utility.Vector3dVector(color_t[:,:3])
        o3d.visualization.draw_geometries([pcd_t, pcd_tn])

    def compute_grad(self):
        param_count = 0
        grad_ = 0.0

        # get grad for model parameters
        for f in self.model.parameters():
            param_count += 1
            if f.grad is None:
                continue
            grad_ += torch.sum(torch.abs(f.grad))

        grad_ /= param_count

        self.log('grad', grad_)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    # def val_dataloader(self):
    #     return self.data_module.val_dataloader(batch_size=self.hparams['batch_size'])

    # def test_dataloader(self):
    #     return self.data_module.test_dataloader(batch_size=self.hparams['batch_size'])
