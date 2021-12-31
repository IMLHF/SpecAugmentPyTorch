# this stft&istft is consistent with torch.stft&torch.istft

import torch as th
import torch.nn.functional as F
from scipy.signal import check_COLA, get_window


support_clp_op = None # if support complex
if th.__version__ >= '1.7.0':
  from torch.fft import rfft as fft
  support_clp_op = True
else:
  from torch import rfft as fft

class ConvSTFT(th.nn.Module):
  def __init__(self, win_len=1024, win_hop=512, fft_len=1024,
               enframe_mode='continue', win_type='hann',
               win_sqrt=False, pad_center=True, **kwargs):
    """
    Implement of STFT using 1D convolution and 1D transpose convolutions.
    Implement of framing the signal in 2 ways, `break` and `continue`.
    `break` method is a kaldi-like framing.
    `continue` method is a librosa-like framing.
    More information about `perfect reconstruction`:
    1. https://ww2.mathworks.cn/help/signal/ref/stft.html
    2. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html
    Args:
      win_len (int): Number of points in one frame.  Defaults to 1024.
      win_hop (int): Number of framing stride. Defaults to 512.
      fft_len (int): Number of DFT points. Defaults to 1024.
      enframe_mode (str, optional): `break` and `continue`. Defaults to 'continue'.
      win_type (str, optional): The type of window to create. Defaults to 'hann'.
      win_sqrt (bool, optional): using square root window. Defaults to False.
      pad_center (bool, optional): `perfect reconstruction` opts. Defaults to True.
    """
    super(ConvSTFT, self).__init__()
    assert enframe_mode in ['break', 'continue']
    assert fft_len >= win_len
    self.win_len = win_len
    self.win_hop = win_hop
    self.fft_len = fft_len
    self.mode = enframe_mode
    self.win_type = win_type
    self.win_sqrt = win_sqrt
    self.pad_center = pad_center
    self.pad_amount = self.fft_len // 2

    en_k, fft_k = self.__init_kernel__()
    self.register_buffer('en_k', en_k)
    self.register_buffer('fft_k', fft_k)
    # print([tmp[0] for tmp in list(self.named_parameters())])

  def __init_kernel__(self):
    """
    Generate enframe_kernel and fft_kernel kernel.
    ** enframe_kernel: Using conv1d layer and identity matrix.
    ** fft_kernel: Using linear layer for matrix multiplication. In fact,
    enframe_kernel and fft_kernel can be combined, But for the sake of
    readability, I took the two apart.

    Returns:
      tuple: two stft kernels.
    """
    enframed_kernel = th.eye(self.fft_len)[:, None, :]
    if support_clp_op:
      tmp = fft(th.eye(self.fft_len))
      fft_kernel = th.stack([tmp.real, tmp.imag], dim=2)
    else:
      fft_kernel = fft(th.eye(self.fft_len), 1)
    if self.mode == 'break':
      enframed_kernel = th.eye(self.win_len)[:, None, :]
      fft_kernel = fft_kernel[:self.win_len]
    fft_kernel = th.cat(
      (fft_kernel[:, :, 0], fft_kernel[:, :, 1]), dim=1)
    window = get_window(self.win_type, self.win_len)

    self.perfect_reconstruct = check_COLA(
      window,
      self.win_len,
      self.win_len-self.win_hop)
    window = th.FloatTensor(window)
    if self.mode == 'continue':
      left_pad = (self.fft_len - self.win_len)//2
      right_pad = left_pad + (self.fft_len - self.win_len) % 2
      window = F.pad(window, (left_pad, right_pad))
    if self.win_sqrt:
      self.padded_window = window
      window = th.sqrt(window)
    else:
      self.padded_window = window**2

    fft_kernel = fft_kernel.T * window
    return enframed_kernel, fft_kernel

  def is_perfect(self):
    """
    Whether the parameters win_len, win_hop and win_sqrt
    obey constants overlap-add(COLA)
    Returns:
        bool: Return true if parameters obey COLA.
    """
    return self.perfect_reconstruct and self.pad_center

  def forward(self, inputs, return_type='N2FT'):
    """Take input data (audio) to STFT domain.
    Args:
      inputs (tensor): Tensor of floats, with shape (num_batch, num_samples)
      return_type (str, optional): return (mag, phase) when `magphase`,
      return (real, imag) when `realimag`, complex(real, imag) when `complex`,
      cat([real.unsqueeze(1), imag.unsqueeze(1)], 1) when `N2FT`.
      Defaults to 'N2FT'.
    Returns:
      tuple: (mag, phase) when `magphase`, return (real, imag) when
      `realimag`, each elements with shape [num_batch, num_frequencies, num_frames].
      Defaults to 'N2FT', with shape [num_batch, 2, num_frequencies, num_frames]
    """
    assert return_type in ['magphase', 'realimag', 'complex', 'N2FT']
    if inputs.dim() == 2:
      inputs = th.unsqueeze(inputs, 1)
    self.num_samples = inputs.size(-1)
    if self.pad_center:
      inputs = F.pad(
        inputs, (self.pad_amount, self.pad_amount), mode='reflect')
    enframe_inputs = F.conv1d(inputs, self.en_k, stride=self.win_hop)
    outputs = th.transpose(enframe_inputs, 1, 2)
    outputs = F.linear(outputs, self.fft_k)
    outputs = th.transpose(outputs, 1, 2)
    dim = self.fft_len//2+1
    real = outputs[:, :dim, :]
    imag = outputs[:, dim:, :]
    if return_type == 'realimag':
      return real, imag
    elif return_type == 'complex':
      assert support_clp_op
      return th.complex(real, imag)
    elif return_type == 'magphase':
      mags = th.sqrt(real**2+imag**2)
      phase = th.atan2(imag, real)
      return mags, phase
    elif return_type == 'N2FT':
      return th.cat([real.unsqueeze(1), imag.unsqueeze(1)], 1) # [N, 2, F, T]
    else:
      raise NotImplementedError(return_type)

class ConvISTFT(th.nn.Module):
  def __init__(self, win_len=1024, win_hop=512, fft_len=1024,
               enframe_mode='continue', win_type='hann',
               win_sqrt=False, pad_center=True, **kwargs):
    """
    Implement of STFT using 1D convolution and 1D transpose convolutions.
    Implement of framing the signal in 2 ways, `break` and `continue`.
    `break` method is a kaldi-like framing.
    `continue` method is a librosa-like framing.
    More information about `perfect reconstruction`:
    1. https://ww2.mathworks.cn/help/signal/ref/stft.html
    2. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html
    Args:
      win_len (int): Number of points in one frame.  Defaults to 1024.
      win_hop (int): Number of framing stride. Defaults to 512.
      fft_len (int): Number of DFT points. Defaults to 1024.
      enframe_mode (str, optional): `break` and `continue`. Defaults to 'continue'.
      win_type (str, optional): The type of window to create. Defaults to 'hann'.
      win_sqrt (bool, optional): using square root window. Defaults to False.
      pad_center (bool, optional): `perfect reconstruction` opts. Defaults to True.
    """
    super(ConvISTFT, self).__init__()
    assert enframe_mode in ['break', 'continue']
    assert fft_len >= win_len
    self.win_len = win_len
    self.win_hop = win_hop
    self.fft_len = fft_len
    self.mode = enframe_mode
    self.win_type = win_type
    self.win_sqrt = win_sqrt
    self.pad_center = pad_center
    self.pad_amount = self.fft_len // 2

    ifft_k, ola_k = self.__init_kernel__()
    self.register_buffer('ifft_k', ifft_k)
    self.register_buffer('ola_k', ola_k)

  def __init_kernel__(self):
    """
    Generate ifft_kernel and overlap-add kernel.
    ** ifft_kernel, pinv of fft_kernel. fft_kernel: Using linear layer for
    matrix multiplication. In fact, enframe_kernel and fft_kernel can be
    combined, But for the sake of readability, I took the two apart.
    ** overlap-add kernel, just like enframe_kernel, but transposed.

    Returns:
        tuple: two istft kernels.
    """
    if support_clp_op:
      tmp = fft(th.eye(self.fft_len))
      fft_kernel = th.stack([tmp.real, tmp.imag], dim=2)
    else:
      fft_kernel = fft(th.eye(self.fft_len), 1)
    if self.mode == 'break':
      fft_kernel = fft_kernel[:self.win_len]
    fft_kernel = th.cat(
      (fft_kernel[:, :, 0], fft_kernel[:, :, 1]), dim=1)
    ifft_kernel = th.pinverse(fft_kernel)[:, None, :]
    window = get_window(self.win_type, self.win_len)

    self.perfect_reconstruct = check_COLA(
      window,
      self.win_len,
      self.win_len-self.win_hop)
    window = th.FloatTensor(window)
    if self.mode == 'continue':
      left_pad = (self.fft_len - self.win_len)//2
      right_pad = left_pad + (self.fft_len - self.win_len) % 2
      window = F.pad(window, (left_pad, right_pad))
    if self.win_sqrt:
      self.padded_window = window
      window = th.sqrt(window)
    else:
      self.padded_window = window**2

    ifft_kernel = ifft_kernel * window
    ola_kernel = th.eye(self.fft_len)[:self.win_len, None, :]
    if self.mode == 'continue':
      ola_kernel = th.eye(self.fft_len)[:, None, :self.fft_len]
    return ifft_kernel, ola_kernel

  def is_perfect(self):
    """
    Whether the parameters win_len, win_hop and win_sqrt
    obey constants overlap-add(COLA)
    Returns:
        bool: Return true if parameters obey COLA.
    """
    return self.perfect_reconstruct and self.pad_center

  def forward(self, input1, L, input2=None, input_type='N2FT'):
    """Call the inverse STFT (iSTFT), given tensors produced
    by the `transform` function.
    Args:
      L (int): wav length.
      input1 (tensors): Magnitude/Real-part/complex/N2FT of STFT with shape
      [num_batch, num_frequencies, num_frames]/~/~/[num_batch, 2, num_frequencies, num_frames]
      input2 (tensors): Phase/Imag-part/None/None of STFT with shape
      [num_batch, num_frequencies, num_frames]/~/-/-
      input_type (str, optional): Mathematical meaning of input tensor's.
      Defaults to 'N2FT', refer to ConvSTFT inputs
    Returns:
      tensors: Reconstructed audio given magnitude and phase. Of
      shape [num_batch, num_samples]
    """
    assert input_type in ['realimag', 'complex', 'magphase', 'N2FT']
    if input_type == 'realimag':
      real, imag = input1, input2
    elif input_type == 'complex':
      assert support_clp_op and th.is_complex(input1)
      real, imag = input1.real, input1.imag
    elif input_type == 'magphase':
      real = input1*th.cos(input2)
      imag = input1*th.sin(input2)
    elif input_type == 'N2FT':
      real, imag = input1[:, 0, :, :], input1[:, 1, :, :]
    else:
      raise NotImplementedError(input_type)
    inputs = th.cat([real, imag], dim=1)
    outputs = F.conv_transpose1d(inputs, self.ifft_k, stride=self.win_hop)
    t = (self.padded_window[None, :, None]).repeat(1, 1, inputs.size(-1))
    t = t.to(inputs.device)
    coff = F.conv_transpose1d(t, self.ola_k, stride=self.win_hop)
    if self.pad_center:
      rm_start, rm_end = self.pad_amount, self.pad_amount+L
      outputs = outputs[..., rm_start:rm_end]
      coff = coff[..., rm_start:rm_end]
    coffidx = th.where(coff > 1e-8)
    outputs[coffidx] = outputs[coffidx]/(coff[coffidx])
    return outputs.squeeze(dim=1)


class N2FT_TO_MAG(th.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, n2ft):
    real, imag = n2ft[:, 0, :, :], n2ft[:, 1, :, :]
    mags = th.sqrt(real**2+imag**2)
    return mags

if __name__ == "__main__":
  """Take input data (audio) to STFT domain and then back to audio.
  Args:
      inputs (tensor): Tensor of floats, with shape [num_batch, num_samples]
  Returns:
      tensor: Reconstructed audio given magnitude and phase.
      Of shape [num_batch, num_samples]
  """
  inputs = th.rand(1,16004)
  stft = ConvSTFT(400,160,512)

  # p = th.nn.Sequential(TorchSafeLog(1e-8), stft)
  # n2ft = p(inputs)
  n2ft = stft(inputs)

  istft = ConvISTFT(400,160,512)
  rec_wav = istft(n2ft, 16004)
  print(th.mean(th.abs(inputs-rec_wav)))
  print(th.sum(th.abs(inputs-rec_wav)))
