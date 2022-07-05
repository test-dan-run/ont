import os
import json
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

from typing import Tuple, Optional

class SubSample(torch.nn.Module):
    def __init__(self, k: int):
        assert k == 2 or k == 4, 'value of k can only be 2 or 4'

        super(SubSample, self).__init__()
        self.k = k
        self.k_correction = 192 if k == 4 else 128

    def forward(self, wav: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        channels, length = wav.shape
        sampled_length = length // self.k - self.k_correction

        # get the starting indices of the values to be sampled from the original wav
        sampled_indices = torch.arange(sampled_length) * self.k

        # randomise the starting indices by the value of k (input samples)
        wav1_rand_nums = torch.randint(high=self.k, size=(sampled_length,))
        wav1_idx = sampled_indices + wav1_rand_nums

        # get indices adjacent to the input indices (target samples)
        wav2_rand_nums = (wav1_rand_nums+1)%self.k
        wav2_idx = sampled_indices + wav2_rand_nums

        # generate input+target samples based on indices
        wav1, wav2 = wav[:,wav1_idx], wav[:,wav2_idx]

        return wav1, wav2

class SpeechDataset(Dataset):

    def __init__(
        self, 
        noisy_data_dir: str, noisy_manifest_path: str, 
        clean_data_dir: Optional[str] = None, clean_manifest_path: Optional[str] = None,
        n_fft: int = 1022, hop_length: int = 256, max_len: int = 65280
        ) -> None:
        super(SpeechDataset, self).__init__()

        # loads noisy files
        with open(noisy_manifest_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        self.noisy_files = [
            os.path.join(
                noisy_data_dir, 
                json.loads(item.strip('\r\n'))['audio_filepath'],
                ) for item in lines]

        # loads clean files
        self.clean_files = None
        if clean_data_dir is not None and clean_manifest_path is not None:
            with open(clean_manifest_path, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
            self.clean_files = [
                os.path.join(
                    clean_data_dir, 
                    json.loads(item.strip('\r\n'))['audio_filepath'],
                    ) for item in lines]       

        # stft parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.len_ = len(self.noisy_files)

        # fixed len
        self.max_len = max_len

        self.subsample = SubSample(k=2)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        waveform, _ = torchaudio.load(file)
        return waveform

    def _prepare_sample(self, waveform):
        current_len = waveform.shape[1]
        if current_len > self.max_len:
            return waveform[:, :self.max_len]
        else:
            return F.pad(waveform, (0, self.max_len-current_len))

    def __getitem__(self, index):
        # load to tensors and normalization
        # followed by padding/cutting
        # followed by STFT
        x_clean_stft = []
        if self.clean_files is not None:
            x_clean = self.load_sample(self.clean_files[index])   #list[n]
            x_clean = self._prepare_sample(x_clean)
            x_clean_stft = torch.stft(input=x_clean, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)

        x_noisy = self.load_sample(self.noisy_files[index])
        x_noisy = self._prepare_sample(x_noisy)
        x_noisy_stft = torch.stft(input=x_noisy, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)

        g1_wav, g2_wav = self.subsample(x_noisy)
        g1_stft = torch.stft(input=g1_wav, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)
        
        return x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft, x_noisy