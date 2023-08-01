import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid.models import BaseModel

class UNet(BaseModel):
    def __init__(self, sample_rate, fft_size, hop_size, window_size, kernel_size, stride, device='cuda', eps=1e-8, mask_logit=False):
        super(UNet, self).__init__(sample_rate=sample_rate)
        # self.save_hyperparameters()
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.mask_logit = mask_logit
        if device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.eps=eps

        #### declare layers

        # input batch normalization
        self.input_layer_speech = input_layer(1)
        self.input_layer_music = input_layer(1)

        # down blocks for speech
        self.down_speech_1 = down(1, 16, self.kernel_size, self.stride,(1,1))
        self.down_speech_2 = down(16, 32, self.kernel_size, self.stride,(2,2))
        self.down_speech_3 = down(32, 64, self.kernel_size, self.stride,(2,2))
        self.down_speech_4 = down(64, 128, self.kernel_size, self.stride,(2,2))
        self.down_speech_5 = down(128, 256, self.kernel_size, self.stride,(2,2))
        self.down_speech_6 = down(256, 512, self.kernel_size, self.stride,(2,2))

        # down blocks for music
        self.down_music_1 = down(1, 16, self.kernel_size, self.stride,(1,1))
        self.down_music_2 = down(16, 32, self.kernel_size, self.stride,(2,2))
        self.down_music_3 = down(32, 64, self.kernel_size, self.stride,(2,2))
        self.down_music_4 = down(64, 128, self.kernel_size, self.stride,(2,2))
        self.down_music_5 = down(128, 256, self.kernel_size, self.stride,(2,2))
        self.down_music_6 = down(256, 512, self.kernel_size, self.stride,(2,2))

        # up blocks for speech
        self.up_speech_1 = up(512, 256, self.kernel_size, self.stride, (1,1),(2,2), 1)
        self.up_speech_2 = up(512, 128, self.kernel_size, self.stride, (1,1),(2,2), 2)
        self.up_speech_3 = up(256, 64, self.kernel_size, self.stride, (1,1),(2,2), 3)
        self.up_speech_4 = up(128, 32, self.kernel_size, self.stride, (1,1),(2,2), 4)
        self.up_speech_5 = up(64, 16, self.kernel_size, self.stride, (1,1),(2,2), 5)

        # up blocks for music
        self.up_music_1 = up(512, 256, self.kernel_size, self.stride, (1,1),(2,2), 1)
        self.up_music_2 = up(512, 128, self.kernel_size, self.stride, (1,1),(2,2), 2)
        self.up_music_3 = up(256, 64, self.kernel_size, self.stride, (1,1),(2,2), 3)
        self.up_music_4 = up(128, 32, self.kernel_size, self.stride, (1,1),(2,2), 4)
        self.up_music_5 = up(64, 16, self.kernel_size, self.stride, (1,1),(2,2), 5)

        # last layer sigmoid
        self.last_layer_speech = last_layer(32, 1, self.kernel_size, self.stride, (0, 1),(1,1))
        self.last_layer_music = last_layer(32, 1, self.kernel_size, self.stride, (0, 1),(1,1))



    def forward(self, x_in):
        # normalize audio
        mean = torch.mean(x_in)
        std = torch.std(x_in)
        x_in = (x_in - mean) / (1e-5 + std)
        x_in = x_in.to(device=self.device)

        # compute normalized spectrogram
        window = torch.hamming_window(self.window_size, device=self.device)
        X_in = torch.stft(x_in, self.fft_size, self.hop_size, window=window, normalized=True, center=True, pad_mode='constant', onesided=True)
        real, imag = X_in.unbind(-1)
        complex_n = torch.cat((real.unsqueeze(1), imag.unsqueeze(1)), dim=1).permute(0,2,3,1).contiguous()
        r_i = torch.view_as_complex(complex_n)
        phase = torch.angle(r_i)
        X_in = torch.sqrt(real**2 + imag**2)

        # add channels dimension
        X = X_in.unsqueeze(1)

        # input layer
        X_speech = self.input_layer_speech(X)
        X_music = self.input_layer_music(X)
        # print("X", X_speech.size())

        # first down layer
        X1_speech = self.down_speech_1(X_speech)
        X1_music = self.down_music_1(X_music)
        # print("X1", X1_speech.size())

        # second down layer
        X2_speech = self.down_speech_2(X1_speech)
        X2_music = self.down_music_2(X1_music)
        # print("X2", X2_speech.size())

        # third down layer
        X3_speech = self.down_speech_3(X2_speech)
        X3_music = self.down_music_3(X2_music)
        # print("X3", X3_speech.size())

        # # fourth down layer
        X4_speech = self.down_speech_4(X3_speech)
        X4_music = self.down_music_4(X3_music)
        # print("X4", X4_speech.size())

        # # 5 down layer
        X5_speech = self.down_speech_5(X4_speech)
        X5_music = self.down_music_5(X4_music)
        # print("X5", X5_speech.size())

        # # 6 down layer
        X6_speech = self.down_speech_6(X5_speech)
        X6_music = self.down_music_6(X5_music)
        # print("x6:", X6_speech.shape)


        # # first up layer
        # print("X5", X5_speech.shape, X6_speech.shape)
        uX5_speech = self.up_speech_1(X6_speech)
        uX5_music = self.up_music_1(X6_music)
        # print("x5 before cat:", uX5_speech.shape, X5_speech.shape)
        X5_speech = torch.cat([uX5_speech, X5_speech], dim=1)
        X5_music = torch.cat([uX5_music, X5_music], dim=1)

        # # 2 up layer
        uX4_speech = self.up_speech_2(X5_speech)
        uX4_music = self.up_music_2(X5_music)
        # print("x4 before cat:", uX4_speech.shape, X4_speech.shape)
        X4_speech = torch.cat([uX4_speech, X4_speech], dim=1)
        X4_music = torch.cat([uX4_music, X4_music], dim=1)

        # # 3 up layer
        uX3_speech = self.up_speech_3(X4_speech)
        uX3_music = self.up_music_3(X4_music)
        # print("x3 before cat:", uX3_speech.shape, X3_speech.shape)
        X3_speech = torch.cat([uX3_speech, X3_speech], dim=1)
        X3_music = torch.cat([uX3_music, X3_music], dim=1)

        # # 4 up layer
        uX2_speech = self.up_speech_4(X3_speech)
        uX2_music = self.up_music_4(X3_music)
        # print("x2 before cat:", uX2_speech.shape, X2_speech.shape)
        X2_speech = torch.cat([uX2_speech, X2_speech], dim=1)
        X2_music = torch.cat([uX2_music, X2_music], dim=1)

        # # 5 up layer
        uX1_speech = self.up_speech_5(X2_speech)
        uX1_music = self.up_music_5(X2_music)
        # print("x1 before cat:", uX1_speech.shape, X1_speech.shape)
        X1_speech = torch.cat([uX1_speech, X1_speech], dim=1)
        X1_music = torch.cat([uX1_music, X1_music], dim=1)

        # last up layer (no concat after transposed conv)
        X_speech = self.last_layer_speech(X1_speech)
        X_music = self.last_layer_music(X1_music)
        # print(X_speech.size())

        # remove channels dimension:
        X_speech = X_speech.squeeze(1)
        X_music = X_music.squeeze(1)


        if self.mask_logit:
            speech = X_in * torch.sigmoid(X_speech)
            music = X_in * torch.sigmoid(X_music)
        else:
            # create the mask
            X_mask_speech = X_speech / (X_speech + X_music + self.eps)
            # use mask to separate speech from mix
            speech = X_in * X_mask_speech
            # and music
            music = X_in * (1 - X_mask_speech)
        # istft
        polar_speech = speech * torch.cos(phase) + speech * torch.sin(phase) * 1j
        polar_music = music * torch.cos(phase) + music * torch.sin(phase) * 1j
        speech_out = torch.istft(polar_speech, self.fft_size, length=x_in.shape[1] ,hop_length=self.hop_size, window=window, return_complex=False, onesided=True, center=True, normalized=True)
        music_out = torch.istft(polar_music, self.fft_size, length=x_in.shape[1], hop_length=self.hop_size, window=window, return_complex=False, onesided=True, center=True, normalized=True)
        # print(speech_out.size())
        # print(x_in.size())
        # remove additional dimention
        speech_out = speech_out.squeeze(1)
        music_out = music_out.squeeze(1)

        # unnormalize
        speech_out = speech_out * std + mean
        music_out = music_out * std + mean

        # add both sources to a tensor to return them
        T_data = torch.stack([speech_out, music_out], dim=1)

        return T_data

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        model_args = {
            "fft_size": self.fft_size,
            "hop_size": self.hop_size,
            "window_size": self.window_size,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "sample_rate": self.sample_rate,
            "mask_logit": self.mask_logit,
            }
        return model_args

# subparts of the U-Net model
class input_layer(nn.Module):
    '''(conv => BN => LeackyReLU)'''
    def __init__(self, out_ch):
        super(input_layer, self).__init__()
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.bn(x)
        return x

class down(nn.Module):
    '''(conv => BN => LeakyReLU)'''
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding_mode='zeros', padding=padding), # padding=stride,
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.conv(x)

        return x

class up(nn.Module):
    '''
        (concatenation => deconv => BN => ReLU => Dropout )
        as proposed in SINGING VOICE SEPARATION WITH DEEP
        U-NET CONVOLUTIONAL NETWORK
    '''
    def __init__(self, in_ch, out_ch, kernel_size, stride, output_padding, padding, index):
        super(up, self).__init__()
        #self.up_conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, output_padding=output_padding, padding=2) #, padding=(2,2)
        if index > 3:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, output_padding=output_padding, padding=padding),
                nn.ReLU(),
                nn.BatchNorm2d(out_ch),
            )
        else:
            # 50% dropout for the first 3 layers
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, output_padding=output_padding, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(out_ch),
                nn.Dropout(p=0.5)
            )

    def forward(self, x):
        x = self.up_conv(x)
        return x


class last_layer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, output_padding, padding):
        super(last_layer, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, output_padding=output_padding, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.deconv(x)

        return x