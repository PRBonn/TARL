import torch.nn as nn
from torch import logspace, cat
from numpy import sqrt, pi, floor, log
import MinkowskiEngine as ME

class PositionalEncoder(nn.Module):
    def __init__(self, max_freq=10000, feat_size=96, dim=3, base=2):
        super().__init__()
        self.max_freq = max_freq
        self.dimensionality = dim
        self.num_bands = int(floor(feat_size / dim / 2))
        self.base = base
        pad = feat_size - self.num_bands * 2 * dim
        self.zero_pad = nn.ZeroPad2d((pad, 0, 0, 0))  # left padding

    def forward(self, x):
        x[:,:2] = x[:,:2] / 50
        x[:,2] = x[:,2] / 4
        x = x.unsqueeze(-1)

        scales = logspace(
            0.0,
            log(self.max_freq / 2) / log(self.base),
            self.num_bands,
            base=self.base,
            device=x.device,
            dtype=x.dtype,
        )
        
        # reshaping
        scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
        x = x * scales * pi

        x = cat([x.sin(), x.cos()], dim=-1)
        x = x.flatten(1)
        enc = self.zero_pad(x)
        return enc

class TemperatureCosineSim(nn.Module):
    def forward(self,q,k,v,tau):
        dk = q.shape[-1]
        w = q@k.transpose(-1,-2)

        # divide the dot product by the temperature
        w_ = w / tau

        f = (w / sqrt(dk)).softmax(-1) @ v
        return f,w_

class TransformerProjector(nn.Module):
    def __init__(self,
                 d_model,
                 num_layer=2,
                 nhead=8,
                 dim_feedforward=512,
                 dropout=0
                 ):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=d_model,
                                         nhead=nhead,
                                         dim_feedforward=dim_feedforward,
                                         dropout=dropout,
                                         batch_first=True)
                                         #norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=num_layer)

        # dec = nn.TransformerDecoderLayer(d_model=d_model,
        #                                  nhead=nhead,
        #                                  dim_feedforward=dim_feedforward,
        #                                  dropout=dropout,
        #                                  batch_first=True)
        #                                  #norm_first=True)
        # self.dec = nn.TransformerDecoder(dec, num_layers=num_layer)
        self.num_layer = num_layer

    def forward(self, encoding, enc_mask=None):
        if self.num_layer > 0:
            enc_mask = enc_mask if enc_mask is None else torch.logical_not(
                enc_mask[..., 0])
            # dec_mask = dec_mask if dec_mask is None else torch.logical_not(
            #     dec_mask[..., 0])

            encoding = self.enc(
                src=encoding, src_key_padding_mask=enc_mask)

            # decoding = self.dec(tgt=decoding,
            #                     memory=encoding,
            #                     tgt_key_padding_mask=dec_mask,
            #                     memory_key_padding_mask=enc_mask)

        return encoding#, decoding

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(BasicBlock, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(Bottleneck, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=1, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

        self.conv3 = ME.MinkowskiConvolution(
            planes, planes * self.expansion, kernel_size=1, dimension=dimension)
        self.norm3 = ME.MinkowskiBatchNorm(
            planes * self.expansion, momentum=bn_momentum)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SegmentationClassifierHead(nn.Module):
    def __init__(self, in_channels=512, out_channels=26):
        nn.Module.__init__(self)

        self.fc = nn.Sequential(nn.Linear(in_channels, out_channels))

    def forward(self, x):
        return self.fc(x.F)

