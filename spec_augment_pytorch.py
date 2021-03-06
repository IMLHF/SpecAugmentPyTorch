"""SpecAugment Implementation for pytorch.
Related paper : https://arxiv.org/pdf/1904.08779.pdf
In this paper, show summarized parameters by each open datasets in Tabel 1.
-----------------------------------------
Policy | W  | F  | m_F |  T  |  p  | m_T
-----------------------------------------
None   |  0 |  0 |  -  |  0  |  -  |  -
-----------------------------------------
LB     | 80 | 27 |  1  | 100 | 1.0 | 1
-----------------------------------------
LD     | 80 | 27 |  2  | 100 | 1.0 | 2
-----------------------------------------
SM     | 40 | 15 |  2  |  70 | 0.2 | 2
-----------------------------------------
SS     | 40 | 27 |  2  |  70 | 0.2 | 2
-----------------------------------------
LB : LibriSpeech basic
LD : LibriSpeech double
SM : Switchboard mild
SS : Switchboard strong

reference:
[1] https://github.com/DemisEom/SpecAugment
[2] https://github.com/zcaceres/spec_augment/issues/17
[3] https://arxiv.org/pdf/1904.08779.pdf
"""

import torch
import random
import librosa
import matplotlib
import numpy as np
import librosa.display
import matplotlib.pyplot as plt


def h_poly(t):
    tt = t.unsqueeze(-2)**torch.arange(4, device=t.device).view(-1,1)
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=t.dtype, device=t.device)
    return A @ tt

def hspline_interpolate_1D(x, y, xs):
    '''
    Input x and y must be of shape (batch, n) or (n)
    '''
    m = (y[..., 1:] - y[..., :-1]) / (x[..., 1:] - x[..., :-1])
    m = torch.cat([m[...,[0]], (m[...,1:] + m[...,:-1]) / 2, m[...,[-1]]], -1)
    idxs = torch.searchsorted(x[..., 1:], xs)
    # print(torch.abs(x.take_along_dim(idxs+1, dim=-1) - x.gather(dim=-1, index=idxs+1)))
    dx = (x.gather(dim=-1, index=idxs+1) - x.gather(dim=-1, index=idxs))
    hh = h_poly((xs - x.gather(dim=-1, index=idxs)) / dx)
    return hh[...,0,:] * y.gather(dim=-1, index=idxs) \
        + hh[...,1,:] * m.gather(dim=-1, index=idxs) * dx \
        + hh[...,2,:] * y.gather(dim=-1, index=idxs+1) \
        + hh[...,3,:] * m.gather(dim=-1, index=idxs+1) * dx
    # dx = (x.take_along_dim(idxs+1, dim=-1) - x.take_along_dim(idxs, dim=-1))
    # hh = h_poly((xs - x.take_along_dim(idxs, dim=-1)) / dx)
    # return hh[...,0,:] * y.take_along_dim(idxs, dim=-1) \
    #     + hh[...,1,:] * m.take_along_dim(idxs, dim=-1) * dx \
    #     + hh[...,2,:] * y.take_along_dim(idxs+1, dim=-1) \
    #     + hh[...,3,:] * m.take_along_dim(idxs+1, dim=-1) * dx

def time_warp(specs, W=50):
  '''
  Timewarp augmentation

  param:
    specs: spectrogram of size (batch, channel, freq_bin, length)
    W: strength of warp
  '''
  device = specs.device
  batch_size, _, num_rows, spec_len = specs.shape

  warp_p = torch.randint(W, spec_len - W, (batch_size,), device=device)

  # Uniform distribution from (0,W) with chance to be up to W negative
  # warp_d = torch.randn(1)*W # Not using this since the paper author make random number with uniform distribution
  warp_d = torch.randint(-W, W, (batch_size,), device=device)
  # print("warp_d", warp_d)
  x = torch.stack([torch.tensor([0], device=device).expand(batch_size),
                   warp_p, torch.tensor([spec_len-1], device=device).expand(batch_size)], 1)
  y = torch.stack([torch.tensor([-1.], device=device).expand(batch_size),
                   (warp_p-warp_d)*2/(spec_len-1.)-1., torch.tensor([1.], device=device).expand(batch_size)], 1)
  # print((warp_p-warp_d)*2/(spec_len-1.)-1.)

  # Interpolate from 3 points to spec_len
  xs = torch.linspace(0, spec_len-1, spec_len, device=device).unsqueeze(0).expand(batch_size, -1)
  ys = hspline_interpolate_1D(x, y, xs)

  grid = torch.cat(
      (ys.view(batch_size,1,-1,1).expand(-1,num_rows,-1,-1),
       torch.linspace(-1, 1, num_rows, device=device).view(-1,1,1).expand(batch_size,-1,spec_len,-1)), -1)

  return torch.nn.functional.grid_sample(specs, grid, align_corners=True)

def spec_augment(mel_spectrogram, time_warping_para=80, frequency_masking_para=27,
                 time_masking_para=100, frequency_mask_num=1, time_mask_num=1):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): [B, C, F, T] audio file path of you want to warping and masking. C=1 for magnitude, C=2 for STFT complex output
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    assert len(mel_spectrogram.shape) == 4, "input spectra as [Batch, Channel, Frequency_dim, N_frame]"
    v = mel_spectrogram.shape[2]
    tau = mel_spectrogram.shape[3]

    # Step 1 : Time warping
    warped_mel_spectrogram = time_warp(mel_spectrogram, W=time_warping_para)

    # Step 2 : Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, v-f)
        warped_mel_spectrogram[:, :, f0:f0+f, :] = 0

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau-t)
        warped_mel_spectrogram[:, :, :, t0:t0+t] = 0

    return warped_mel_spectrogram


def visualization_spectrogram(mel_spectrogram, title):
    """visualizing result of SpecAugment
    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 3))
    # plt_d = librosa.power_to_db(mel_spectrogram[ :, :], ref=np.max)
    plt_d = np.log(mel_spectrogram.numpy()+0.05)
    plt_d = plt_d - plt_d.min()
    librosa.display.specshow(plt_d, y_axis='mel', x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig("./examples/"+title)
    plt.show()
    plt.close()

class SpecAugmentTorch(torch.nn.Module):

  def __init__(self, W, F, mF, T, p, mT, batch):
      super().__init__()
      self.W = W
      self.F = F
      self.mF = mF
      self.T = T
      self.p = p
      self.mT = mT
      self.batch = batch

  def forward(self, spec_batch):
    '''
    spec_batch: [b, F, T]
    '''
    if self.batch:
      return spec_augment(spec_batch, self.W, self.F, self.T, self.mF, self.mT)
    else:
      specaug_lst = []
      for i in range(spec_batch.shape[0]):
        spec_aug = spec_augment(spec_batch[i].unsqueeze(0), self.W, self.F, self.T, self.mF, self.mT)
        specaug_lst.append(spec_aug)
      specaug_batch = torch.cat(specaug_lst, dim=0)
      return specaug_batch

class N2FT_TO_MAG(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, n2ft):
    # n2ft: [N, 2, F, T]
    real, imag = n2ft[:, 0, :, :], n2ft[:, 1, :, :]
    mags = torch.sqrt(real**2+imag**2)
    return mags #[N, F, T]

if __name__ == "__main__":
  p = {'W':40, 'F':19, 'mF':2, 'T':100, 'p':1.0, 'mT':2, 'batch':False}
  aug_fn = SpecAugmentTorch(**p)
  n2ft_to_mag = N2FT_TO_MAG()

  import soundfile as sf
  wav1, sr = sf.read("./examples/1089-0001.flac")
  wav2, sr = sf.read("./examples/1089-0002.flac")
  # wav1 = wav1[:len(wav2)]
  # sf.write("./examples/1089-0001.flac", wav1, sr)
  wav1 = wav1.astype(np.float32)
  wav2 = wav2.astype(np.float32)
  print(wav1.dtype)
  print(wav1.shape, wav2.shape)
  wav = torch.from_numpy(np.stack([wav1, wav2]))
  spec = torch.stft(wav, 512, 160, 512, torch.hann_window(512)).permute(0, 3, 1, 2) # [N, 2, F, T]
  print(spec.shape) # [N, 2, F, T]

  spec_aug = aug_fn(spec)

  wav_aug = torch.istft(spec_aug.permute(0, 2, 3, 1), 512, 160, 512, torch.hann_window(512), length=wav.shape[-1])
  sf.write("./examples/1089-0001-SpecAug.flac", wav_aug[0], sr)
  sf.write("./examples/1089-0002-SpecAug.flac", wav_aug[1], sr)

  mag = n2ft_to_mag(spec)
  mag_aug = n2ft_to_mag(spec_aug)

  visualization_spectrogram(mag[0],"1089-0001")
  visualization_spectrogram(mag_aug[0],"1089-0001-SpecAug")
  visualization_spectrogram(mag[1],"1089-0002")
  visualization_spectrogram(mag_aug[1],"1089-0002-SpecAug")


