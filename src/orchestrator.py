from typing import List
import random
import matplotlib.pyplot as plt
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

        x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft, x_noisy = batch
        fx_wav, fg1_wav, g1fx, g2fx = self(x_noisy_stft, g1_stft)

        loss = self.loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)
        self.log('train_loss', loss, on_step=True, on_epoch=False, sync_dist=self.run_cfg.distributed)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft, x_noisy = batch
        fx_wav, fg1_wav, g1fx, g2fx = self(x_noisy_stft, g1_stft)

        loss = self.loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=self.run_cfg.distributed)

        return {'valid_loss': loss, 'x_noisy': x_noisy[0], 'x_gen': fx_wav[0]}

    def validation_epoch_end(self, validation_step_outputs):

        noisy_list = [x['x_noisy'] for x in validation_step_outputs]
        gened_list = [x['x_gen'] for x in validation_step_outputs]

        rand_indices = random.sample(range(len(noisy_list)), k=min(len(noisy_list), 5))
        for idx, rand_idx in enumerate(rand_indices):
            noisy_wav, gen_wav = noisy_list[rand_idx], gened_list[rand_idx]
            figure = plt.figure(figsize=(12,10))
            plt.subplot(2,1,1, title=f'Noisy WAV (batch {rand_idx})')
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.plot(noisy_wav.squeeze().cpu().numpy())

            plt.subplot(2,1,2, title=f'Generated WAV (batch {rand_idx})')
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.plot(gen_wav.cpu().numpy())
            self.logger.experiment.add_figure(f'Waveform Comparison - {idx}', figure, global_step=self.current_epoch)
            self.logger.experiment.add_audio(f'Noisy WAV - {idx}', noisy_wav.squeeze().cpu().numpy(), sample_rate=16000, global_step=self.current_epoch)
            self.logger.experiment.add_audio(f'Clean WAV - {idx}', gen_wav.cpu().numpy(), sample_rate=16000, global_step=self.current_epoch)


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