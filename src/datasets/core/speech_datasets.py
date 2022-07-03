import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset
import pytorch_lightning as pl

from typing import Tuple, Optional

class SubSample(pl.LightningModule):
    def __init__(self, k: int):
        assert k == 2 or k == 4, 'value of k can only be 2 or 4'

        super(SubSample, self).__init__()
        self.k = k
        self.k_correction = 192 if k == 4 else 128

    def forward(self, wav: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        channels, dims = wav.shape
        sampled_dims = dims // self.k - self.k_correction
        
        # official implementation
        # for channel in range(channels):
        #     for i in range(sampled_dims):
        #         s_i = i * self.k
        #         num = np.random.choice(self.k_range)
        #         wav1[channel, i], wav2[channel, i] = wav[channel, s_i+num%self.k], wav[channel, s_i+(num+1)%self.k]
        
        # return wav1, wav2

        # speed-up implementation
        k_values = torch.arange(sampled_dims) * self.k
        wav1_rand_nums = torch.randint(high=self.k, size=(sampled_dims,))
        wav2_rand_nums = (wav1_rand_nums+1)%self.k
        wav1_idx = k_values + wav1_rand_nums
        wav2_idx = k_values + wav2_rand_nums

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

        output = torch.zeros((1, self.max_len), dtype=torch.float32)
        output[0, -current_len:] = waveform[0, :self.max_len]

        return output

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
        
        return x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft