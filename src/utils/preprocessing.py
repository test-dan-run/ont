import math
import scipy
import numpy as np

import torch
import torch.nn as nn

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, x):
        return torch.from_numpy(x).float()

class SignalToFrames:
    """Chunks a signal into frames
        required input shape is [1, 1, -1]
        input params:    (frame_size: window_size,  frame_shift: overlap(samples))
        output:   [1, 1, num_frames, frame_size]
    """

    def __init__(self, frame_size=512, frame_shift=256):
        self.frame_size = frame_size
        self.frame_shift = frame_shift

    def __call__(self, in_sig):

        sig_len = in_sig.shape[-1]
        nframes = math.ceil((sig_len - self.frame_size) / self.frame_shift + 1)
        a = np.zeros(list(in_sig.shape[:-1]) + [nframes, self.frame_size])
        start = 0
        end = start + self.frame_size
        k = 0
        for i in range(nframes):
            if end < sig_len:
                a[..., i, :] = in_sig[..., start:end]
                k += 1
            else:
                tail_size = sig_len - start
                a[..., i, :tail_size] = in_sig[..., start:]

            start = start + self.frame_shift
            end = start + self.frame_size
        return a


class TorchSignalToFrames(object):
    """
    it is for torch tensor
    """
    def __init__(self, frame_size=512, frame_shift=256):
        super(TorchSignalToFrames, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift

    def __call__(self, in_sig):
        sig_len = in_sig.shape[-1]
        nframes = math.ceil((sig_len - self.frame_size) / self.frame_shift + 1)
        a = torch.zeros(tuple(in_sig.shape[:-1]) + (nframes, self.frame_size), device=in_sig.device)
        start = 0
        end = start + self.frame_size
        k = 0
        for i in range(nframes):
            if end < sig_len:
                a[..., i, :] = in_sig[..., start:end]
                k += 1
            else:
                tail_size = sig_len - start
                a[..., i, :tail_size] = in_sig[..., start:]

            start = start + self.frame_shift
            end = start + self.frame_size
        return a


class OLA:
    """Performs overlap-and-add
        required input is ndarray
        performs frames into signal
    """
    def __init__(self, frame_shift=256):
        self.frame_shift = frame_shift

    def __call__(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = np.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype)
        ones = np.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones


class TorchOLA(nn.Module):
    """Overlap and add on gpu using torch tensor
        required input is tensor
        perform frames into signal
        used in the output of network
    """
    # Expects signal at last dimension
    def __init__(self, frame_shift=256):
        super(TorchOLA, self).__init__()
        self.frame_shift = frame_shift

    def forward(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = torch.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype, device=inputs.device, requires_grad=False)
        ones = torch.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones


class STFT:
    """Computes STFT of a signal
    input is ndarray
    required input shape is [1, 1, -1]
    """
    def __init__(self, frame_size=512, frame_shift=256):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.win = scipy.hamming(frame_size)
        self.get_frames = SignalToFrames(self.frame_size, self.frame_shift)
    def __call__(self, signal):
        frames = self.get_frames(signal)
        frames = frames*self.win
        feature = np.fft.fft(frames)[..., 0:(self.frame_size//2+1)]
        feat_R = np.real(feature)
        feat_I = np.imag(feature)
        feature = np.stack([feat_R, feat_I], axis=0)
        return feature


class ISTFT:
    r"""Computes inverse STFT"""
    # includes overlap-and-add
    def __init__(self, frame_size=512, frame_shift=256):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.win = scipy.hamming(frame_size)
        self.ola = OLA(self.frame_shift)
    def __call__(self, stft):
        R = stft[0:1, ...]
        I = stft[1:2, ...]
        cstft = R + 1j*I
        fullFFT = np.concatenate((cstft, np.conj(cstft[..., -2:0:-1])), axis=-1)
        T = np.fft.ifft(fullFFT)
        T = np.real(T)
        T = T / self.win
        signal = self.ola(T)
        return signal.astype(np.float32)
