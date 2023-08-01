import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from .d3net import D3_block
from torch.nn import Parameter
from asteroid.models.base_models import BaseModel

class MultiHeadAttention(nn.Module):
    __constants__ = [
        'out_channels',
        'd_model',
        'n_heads',
        'query_shape',
        'memory_flange'
    ]

    max_bins: int
    d_model: int
    n_heads: int
    query_shape: int
    memory_flange: int

    def __init__(self, nb_channels, out_channels, d_model=32, n_heads=8, query_shape=24, memory_flange=8):
        super().__init__()

        self.out_channels = out_channels
        self.d_model = d_model
        self.n_heads = n_heads
        self.query_shape = query_shape
        self.memory_flange = memory_flange

        self.qkv_conv = nn.Conv2d(nb_channels, d_model * 3, 3, padding=1)
        self.out_conv = nn.Conv2d(
            d_model, out_channels, 3, padding=1, bias=False)

    def _pad_to_multiple_2d(self, x: torch.Tensor, query_shape: int):
        t = x.shape[-1]
        offset = t % query_shape
        if offset != 0:
            offset = query_shape - offset

        if offset > 0:
            return F.pad(x, [0, offset])
        return x

    def forward(self, x):
        qkv = self._pad_to_multiple_2d(self.qkv_conv(x), self.query_shape)
        qkv = qkv.view((qkv.shape[0], self.n_heads, -1) + qkv.shape[2:])
        q, k, v = qkv.chunk(3, 2)

        k_depth_per_head = self.d_model // self.n_heads
        q = q * k_depth_per_head ** -0.5

        k = F.pad(k, [self.memory_flange] * 2)
        v = F.pad(v, [self.memory_flange] * 2)

        unfold_q = q.reshape(
            q.shape[:4] + (q.shape[4] // self.query_shape, self.query_shape))
        unfold_k = k.unfold(-1, self.query_shape +
                            self.memory_flange * 2, self.query_shape)
        unfold_v = v.unfold(-1, self.query_shape +
                            self.memory_flange * 2, self.query_shape)

        unfold_q = unfold_q.permute(0, 1, 4, 3, 5, 2)
        tmp = unfold_q.shape
        unfold_q = unfold_q.reshape(
            -1, unfold_q.shape[-2] * unfold_q.shape[-3], k_depth_per_head)
        unfold_k = unfold_k.permute(0, 1, 4, 2, 3, 5).reshape(
            unfold_q.shape[0], k_depth_per_head, -1)
        unfold_v = unfold_v.permute(0, 1, 4, 3, 5, 2).reshape(
            unfold_q.shape[0], -1, k_depth_per_head)

        bias = (unfold_k.abs().sum(-2, keepdim=True)
                == 0).to(unfold_k.dtype) * -1e-9
        # correct value should be -1e9, we type this by accident and use it during the whole competition
        # so just leave it what it was :)

        logits = unfold_q @ unfold_k + bias
        weights = logits.softmax(-1)
        out = weights @ unfold_v

        out = out.view(tmp).permute(0, 1, 5, 3, 2, 4)
        out = out.reshape(out.shape[0], out.shape[1] *
                          out.shape[2], out.shape[3], -1)
        out = out[..., :x.shape[2], :x.shape[3]]

        return self.out_conv(out)


class UNetAttn(BaseModel):
    def __init__(self,
            window_length=4096,
            in_chan=4096,
            n_hop=1024,
            nb_channels=1,
            sample_rate=44100,
            max_bins=None,
            k=12,
            spec_power=1,
            n_src=2,
            mask_logit=False,
            eps=1e-10,
            hidden_size=128,
            device='cuda'):
        super(UNetAttn, self).__init__(sample_rate=sample_rate)

        if device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.max_bins = max_bins
        self.window_length = window_length
        self.in_chan = in_chan
        self.n_hop = n_hop
        self.nb_output_bins = in_chan // 2 + 1
        if max_bins:
            self.max_bins = max_bins
        else:
            self.max_bins = self.nb_output_bins
        self.spec_power = spec_power
        self.n_src = n_src
        self.nb_channels=nb_channels
        self.mask_logit=mask_logit
        self.eps=eps
        self.hidden_size=hidden_size
        self.k=k

        layers = 6
        self.conv_n_filters = []

        #nb_channels = 2
        self.down_convs = nn.ModuleList()

        # Define spectral encoder
        stft = _STFT(window_length=window_length, n_fft=in_chan, n_hop=n_hop, center=True)
        stft.window.to(self.device)
        spec = _Spectrogram(spec_power=spec_power, mono=(nb_channels == 1))
        self.encoder = nn.Sequential(stft, spec)  # Return: Spec, Angle
        # Define spectral decoder
        self.decoder = _ISTFT(window=stft.window, n_fft=in_chan, hop_length=n_hop, center=True)

        for i in range(layers):
            L = max(2, layers - i - 1)
            tmp = D3_block(
                nb_channels, M=2,
                k=k, L=L, last_n_layers=L
            )
            out_channels = tmp.get_output_channels()
            self.down_convs.append(
                nn.Sequential(
                    tmp,
                    nn.AvgPool2d(2, 2)
                )
            )
            nb_channels = out_channels
            self.conv_n_filters.append(out_channels)

        self.attn = nn.Sequential(
            MultiHeadAttention(nb_channels, self.hidden_size*2, self.hidden_size,
                               query_shape=2, memory_flange=3),
            nn.BatchNorm2d(self.hidden_size*2),
            nn.ReLU(inplace=True),
            MultiHeadAttention(self.hidden_size*2, self.hidden_size*2, self.hidden_size,
                               query_shape=2, memory_flange=3),
            nn.BatchNorm2d(self.hidden_size*2),
            nn.ReLU(inplace=True)
        )

        self.up_convs = nn.ModuleList()
        concat_channels = 256
        n_droupout = 3
        for i in range(len(self.conv_n_filters) - 2, -1, -1):
            L = max(2, layers - i - 1)
            d3 = D3_block(
                self.conv_n_filters[i + 1] + concat_channels, M=2,
                k=k, L=L, last_n_layers=L
            )
            tmp = [
                d3,
                nn.ConvTranspose2d(d3.get_output_channels(
                ), self.conv_n_filters[i], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.conv_n_filters[i]),
                nn.ReLU(inplace=True)
            ]
            if n_droupout > 0:
                tmp.append(nn.Dropout2d(0.4))
                n_droupout -= 1
            self.up_convs.append(
                nn.Sequential(
                    *tmp
                )
            )
            concat_channels = self.conv_n_filters[i]

        self.end = nn.Sequential(
            nn.ConvTranspose2d(concat_channels * 2,
                               concat_channels, 5, 2, 2, bias=False),
            nn.BatchNorm2d(concat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(concat_channels, self.nb_channels*self.n_src, 3, padding=2, dilation=2)
        )

    def forward(self, wav):
        mean = torch.mean(wav)
        std = torch.std(wav)
        wav = (wav - mean) / (1e-5 + std)
        wav = wav.to(device=self.device)
        while wav.dim()<3:
            wav = wav.unsqueeze(1)

        # Transform
        mixture, ang = self.encoder(wav)
        mixture = mixture.permute(1,2,0,3)

        x = mixture[..., :self.max_bins]
        skips = []
        for conv in self.down_convs:
            x = conv(x)
            skips.append(x)
        skips.pop()

        x = torch.cat([x, self.attn(x)], 1)

        i = len(skips) - 1
        for conv in self.up_convs:
            sk = skips[i]
            x = conv(x)
            if x.shape[2] < sk.shape[2] or x.shape[3] < sk.shape[3]:
                x = F.pad(x, [0, sk.shape[3] - x.shape[3],
                              0, sk.shape[2] - x.shape[2]])
            x = torch.cat([x, sk], 1)
            i -= 1
        x = self.end(x)
        x = x.sigmoid()

        x = F.pad(x, [0, mixture.shape[3] - x.shape[3], 0,
                      mixture.shape[2] - x.shape[2]], value=0.25)
        x = x.view(x.shape[0], self.n_src, self.nb_channels, x.shape[2], x.shape[3])

        # Apply masks to mixture
        masked_mixture = self.apply_masks(mixture, x)

        # Inverse Transform
        #[2, 64, 1, 513, 141]
        spec = masked_mixture.permute(0, 1, 2, 4, 3)
        time_signals = self.decoder(spec, ang)
        time_signals = time_signals.permute(1,0,2,3)
        time_signals=time_signals.squeeze(2)
        if time_signals.shape[-1]<wav.shape[-1]:
            # pad(left, right, top, bottom)
            time_signals=torch.nn.functional.pad(input=time_signals, pad=(0, wav.shape[-1]-time_signals.shape[-1], 0, 0), mode='constant', value=0)

        return time_signals * std + mean

    def apply_masks(self, mixture, est_masks):
        if self.mask_logit:
            masked_tf_rep = torch.stack([mixture * est_masks[:,i,...] for i in range(self.n_src)])
        else:
            masked_tf_rep = torch.stack([mixture * est_masks[:,i,...]/(torch.sum(est_masks,axis=1)+self.eps) for i in range(self.n_src)])

        return masked_tf_rep

    def get_model_args(self):
        """ Arguments needed to re-instantiate the model. """
        fb_config = {
            "window_length": self.window_length,
            "in_chan": self.in_chan,
            "n_hop": self.n_hop,
            "sample_rate": self.sample_rate,
        }

        net_config = {
            "n_src": self.n_src,
            "hidden_size": self.hidden_size,
            "nb_channels": self.nb_channels,
            "max_bins": self.max_bins,
            "spec_power": self.spec_power,
            "mask_logit": self.mask_logit,
            "k": self.k,
        }

        # Merge all args under model_args.
        model_args = {
            **fb_config,
            **net_config,
        }
        return model_args

class _STFT(nn.Module):
    def __init__(self, window_length, n_fft=4096, n_hop=1024, center=True):
        super(_STFT, self).__init__()
        self.window = Parameter(torch.hann_window(window_length), requires_grad=False)
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """

        nb_samples, nb_channels, nb_timesteps = x.size()

        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_samples * nb_channels, -1)
        # compute stft with parameters as close as possible scipy settings
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            pad_mode="reflect",
            return_complex=False,
        )

        # reshape back to channel dimension
        stft_f = stft_f.contiguous().view(nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2)
        return stft_f


class _Spectrogram(nn.Module):
    def __init__(self, spec_power=1, mono=True):
        super(_Spectrogram, self).__init__()
        self.spec_power = spec_power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram and the corresponding phase
            (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        phase = stft_f.detach().clone()
        phase = torch.atan2(phase[Ellipsis, 1], phase[Ellipsis, 0])

        stft_f = stft_f.transpose(2, 3)

        # take the magnitude
        stft_f = stft_f.pow(2).sum(-1).pow(self.spec_power / 2.0)

        # downmix in the mag domain
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)
            phase = torch.mean(phase, 1, keepdim=True)

        # permute output for LSTM convenience
        return [stft_f.permute(2, 0, 1, 3), phase]


class _ISTFT(nn.Module):
    def __init__(self, window, n_fft=4096, hop_length=1024, center=True):
        super(_ISTFT, self).__init__()
        self.window = window
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center

    def forward(self, spec, ang):
        sources, bsize, channels, fbins, frames = spec.shape
        x_r = spec * torch.cos(ang)
        x_i = spec * torch.sin(ang)
        x = torch.stack([x_r, x_i], dim=-1)
        x = x.view(sources * bsize * channels, fbins, frames, 2)
        wav = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=self.center
        )
        wav = wav.view(sources, bsize, channels, wav.shape[-1])

        return wav

if __name__ == "__main__":
    net = UNetAttn()
    # net = torch.jit.script(net)
    print(net)
    print(sum(p.numel() for p in net.parameters()
              if p.requires_grad), net.conv_n_filters)
    x = torch.rand(1, 2, 2049, 512)
    y = net(x)
    print(y.shape)