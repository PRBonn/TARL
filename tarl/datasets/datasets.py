import warnings
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tarl.datasets.dataloader.SemanticKITTITemporal import TemporalKITTISet
from tarl.datasets.dataloader.SemanticKITTI import KITTISet
from tarl.datasets.dataloader.DataloaderTemplate import TemplateSet
from tarl.utils.collations import SparseAugmentedCollation

warnings.filterwarnings('ignore')

__all__ = ['TemporalKittiDataModule', 'KittiDataModule', 'TemplateDataModule']


class TemporalKittiDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseAugmentedCollation(self.cfg['train']['resolution'])

        data_set = TemporalKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            scan_window=self.cfg['train']['scan_window'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['train']['resolution'],
            percentage=self.cfg['data']['percentage'],
            intensity_channel=self.cfg['data']['intensity'],
            use_ground_pred=self.cfg['data']['use_ground_pred'],
            num_points=self.cfg['train']['num_points'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

class KittiDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseAugmentedCollation(self.cfg['train']['resolution'])

        data_set = KITTISet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            split=self.cfg['data']['split'],
            pre_training=self.cfg['train']['pre_train'],
            resolution=self.cfg['train']['resolution'],
            percentage=self.cfg['data']['percentage'],
            intensity_channel=self.cfg['data']['intensity'],
            num_points=self.cfg['train']['num_points'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

class TemplateDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        # this collation list points and objects views and create a dict out of it  
        collate = SparseAugmentedCollation(self.cfg['train']['resolution'])

        data_set = TemplateSet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            split=self.cfg['data']['split'],
            pre_training=self.cfg['train']['pre_train'],
            resolution=self.cfg['train']['resolution'],
            percentage=self.cfg['data']['percentage'],
            intensity_channel=self.cfg['data']['intensity'],
            num_points=self.cfg['train']['num_points'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

data_modules = {
    'TemporalSemKITTI': TemporalKittiDataModule,
    'SemKITTI': KittiDataModule,
    'TemplateDataset': TemplateDataModule,
}
