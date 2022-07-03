import os
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.datasets.core.speech_datasets import SpeechDataset

class LightningSpeechDataset(pl.LightningDataModule):
    def __init__(self, cfg: dict, batch_size: int = 32):

        super(LightningSpeechDataset, self).__init__()
        self.cfg = cfg
        self.batch_size = batch_size

        self.ds = {}

    def setup(self, stage: Optional[str] = None):

        if stage in (None, 'fit'):
            self.train_data = SpeechDataset(
                noisy_data_dir = self.cfg.train_noisy_data_dir,
                noisy_manifest_path = self.cfg.train_noisy_manifest_path,
                clean_data_dir = self.cfg.train_clean_data_dir,
                clean_manifest_path = self.cfg.train_clean_manifest_path,
                n_fft = self.cfg.n_fft,
                hop_length = self.cfg.hop_length,
                max_len = self.cfg.max_len
            )
            self.valid_data = SpeechDataset(
                noisy_data_dir = self.cfg.valid_noisy_data_dir,
                noisy_manifest_path = self.cfg.valid_noisy_manifest_path,
                clean_data_dir = self.cfg.valid_clean_data_dir,
                clean_manifest_path = self.cfg.valid_clean_manifest_path,
                n_fft = self.cfg.n_fft,
                hop_length = self.cfg.hop_length,
                max_len = self.cfg.max_len
            )
        if stage == 'test':
            self.test_data = SpeechDataset(
                noisy_data_dir = self.cfg.test_noisy_data_dir,
                noisy_manifest_path = self.cfg.test_noisy_manifest_path,
                clean_data_dir = self.cfg.test_clean_data_dir,
                clean_manifest_path = self.cfg.test_clean_manifest_path,
                n_fft = self.cfg.n_fft,
                hop_length = self.cfg.hop_length,
                max_len = self.cfg.max_len
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, batch_size=self.batch_size)