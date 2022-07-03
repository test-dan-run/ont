from typing import List
from omegaconf import DictConfig

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

import pytorch_lightning as pl

from src.model.dcunet import DCUnet10
from src.losses import RegularizedLoss
from src.datasets.core.speech_datasets import SubSample

class LightningONT(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, optim_cfg: DictConfig, run_cfg: DictConfig, n_fft: int = 1022, hop_length: int = 256):

        super(LightningONT, self).__init__()

        self.dcunet = DCUnet10(**model_cfg)
        self.subsample = SubSample(k=2)
        self.loss_fn = RegularizedLoss()

        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.optim_cfg = optim_cfg
        self.run_cfg = run_cfg


    def model(self, x_noisy_stft, g1_stft, g1_wav, g2_wav):

        # for base training
        fg1_wav = self.dcunet(g1_stft, n_fft=self.n_fft, hop_length=self.hop_length)

        # for regularization loss
        fx_wav = self.dcunet(x_noisy_stft, n_fft=self.n_fft, hop_length=self.hop_length)
        g1fx, g2fx = self.subsample(fx_wav)

        loss = self.loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)

        return fx_wav, loss

    def forward(self, x_noisy_stft, g1_stft):
        
        # for base training
        fg1_wav = self.dcunet(g1_stft, n_fft=self.n_fft, hop_length=self.hop_length)

        # for regularization loss
        fx_wav = self.dcunet(x_noisy_stft, n_fft=self.n_fft, hop_length=self.hop_length)
        g1fx, g2fx = self.subsample(fx_wav)

        return fx_wav, fg1_wav, g1fx, g2fx

    def training_step(self, batch, batch_idx):

        x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft = batch
        fx_wav, fg1_wav, g1fx, g2fx = self(x_noisy_stft, g1_stft)

        loss = self.loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)
        self.log('train_loss', loss, on_step=True, on_epoch=False, sync_dist=self.run_cfg.distributed)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft = batch
        fx_wav, fg1_wav, g1fx, g2fx = self(x_noisy_stft, g1_stft)

        loss = self.loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=self.run_cfg.distributed)

        return {'valid_loss': loss}

    def test_step(self, batch, batch_idx):

        x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft = batch
        fx_wav, fg1_wav, g1fx, g2fx = self(x_noisy_stft, g1_stft)

        loss = self.loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)

        return {'test_loss': loss}

    def predict_step(self, batch, batch_idx):

        x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft = batch
        fx_wav = self.dcunet(x_noisy_stft, n_fft=self.n_fft, hop_length=self.hop_length)

        return fx_wav

    def configure_optimizers(self):

        if self.optim_cfg.optim == 'adam':
            optim = Adam(self.parameters(), lr = self.optim_cfg.lr)

        if self.optim_cfg.scheduler == 'decay':
            scheduler = ReduceLROnPlateau(
                optimizer = optim,
                mode = self.optim_cfg.mode,
                factor = self.optim_cfg.factor,
                patience = self.optim_cfg.patience,
                cooldown = self.optim_cfg.cooldown,
                eps = self.optim_cfg.eps
            )
        elif self.optim_cfg.scheduler == 'cyclic':
            scheduler = CyclicLR(
                optimizer = optim,
                mode = self.optim_cfg.mode,
                lr = self.optim_cfg.lr,
                max_lr = self.optim_cfg.max_lr,
                epoch_size_up = self.optim_cfg.epoch_size_up,
                epoch_size_down = self.optim_cfg.epoch_size_down
            )

        return {
            'optimizer': optim,
            'monitor': 'valid_loss',
            'lr_scheduler': scheduler,
            'interval': self.optim_cfg.interval
        }