import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
import pytorch_lightning as pl
from src.utils.preprocessing import TorchSignalToFrames

SAMPLE_RATE = 16000
N_FFT = 1022
HOP_LENGTH = 256

class MSELoss:
    def __call__(self, outputs, labels, loss_mask):
        masked_outputs = outputs * loss_mask
        masked_labels = labels * loss_mask
        loss = torch.sum((masked_outputs - masked_labels)**2.0) / torch.sum(loss_mask)
        return loss

class STFTMLoss(pl.LightningModule):
    def __init__(self, frame_size=512, frame_shift=256, loss_type='mae'):
        super(STFTMLoss, self).__init__()

        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.loss_type = loss_type
        self.frame = TorchSignalToFrames(frame_size=self.frame_size,
                                         frame_shift=self.frame_shift)
        D = linalg.dft(frame_size)
        W = np.hamming(self.frame_size)
        DR = np.real(D)
        DI = np.imag(D)
        self.DR = torch.from_numpy(DR).float().cuda()  # to(self.device)
        self.DR = self.DR.contiguous().transpose(0, 1)
        self.DI = torch.from_numpy(DI).float().cuda()  # to(self.device)
        self.DI = self.DI.contiguous().transpose(0, 1)
        self.W = torch.from_numpy(W).float().cuda()  # to(self.device)

    def get_stftm(self, frames):
        frames = frames * self.W
        stft_R = torch.matmul(frames, self.DR)
        stft_I = torch.matmul(frames, self.DI)
        stftm = torch.abs(stft_R) + torch.abs(stft_I)
        return stftm

    def __call__(self, outputs, labels, loss_mask):
        outputs = self.frame(outputs)
        labels = self.frame(labels)
        loss_mask = self.frame(loss_mask)
        outputs = self.get_stftm(outputs)
        labels = self.get_stftm(labels)

        masked_outputs = outputs * loss_mask
        masked_labels = labels * loss_mask
        if self.loss_type == 'mse':
            loss = torch.sum((masked_outputs - masked_labels)**2) / torch.sum(loss_mask)
        elif self.loss_type == 'mae':
            loss = torch.sum(torch.abs(masked_outputs - masked_labels)) / torch.sum(loss_mask)

        return loss

class REGLoss(object):
    def __call__(self, fg1, g2, g1fx, g2fx):
        return torch.mean((fg1-g2-g1fx+g2fx)**2)

def compLossMask(inp, nframes):
    loss_mask = torch.zeros_like(inp).requires_grad_(False)
    for j, seq_len in enumerate(nframes):
        loss_mask.data[j, :, 0:seq_len] += 1.0   # loss_mask.shape: torch.Size([2, 1, 32512])
    return loss_mask

class RegularizedLoss(pl.LightningModule):
    def __init__(self, gamma=1):
        super(RegularizedLoss, self).__init__()

        self.gamma = gamma
        self.time_loss = MSELoss()
        self.freq_loss = STFTMLoss()
        self.reg_loss = REGLoss()

    def wsdr_fn(self, x_, y_pred_, y_true_, eps=1e-8):  # g1_wav, fg1_wav, g2_wav
        y_pred = y_pred_.flatten(1)
        y_true = y_true_.flatten(1)
        x = x_.flatten(1)

        def sdr_fn(true, pred, eps=1e-8):
            num = torch.sum(true * pred, dim=1)
            den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
            return -(num / (den + eps))

        # true and estimated noise
        z_true = x - y_true
        z_pred = x - y_pred

        a = torch.sum(y_true ** 2, dim=1) / (torch.sum(y_true ** 2, dim=1) + torch.sum(z_true ** 2, dim=1) + eps)
        wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
        return torch.mean(wSDR)

    def regloss(self, g1, g2, G1, G2):
        return torch.mean((g1-g2-G1+G2)**2)

    def forward(self, g1_wav, fg1_wav, g2_wav, g1fx, g2fx):

        if(g2_wav.shape[0] == 2):
            nframes = [g2_wav.shape[2],g2_wav.shape[2]]   # nframes: [32512, 32512]
        else:
            nframes = [g2_wav.shape[2]]

        loss_mask = compLossMask(g2_wav, nframes)   
        loss_mask = loss_mask.float()
        loss_time = self.time_loss(fg1_wav, g2_wav, loss_mask)
        loss_freq = self.freq_loss(fg1_wav, g2_wav, loss_mask)
        loss1 = (0.8 * loss_time + 0.2 * loss_freq)/600

        return loss1 + self.wsdr_fn(g1_wav, fg1_wav, g2_wav) + self.gamma * self.regloss(fg1_wav, g2_wav, g1fx, g2fx)