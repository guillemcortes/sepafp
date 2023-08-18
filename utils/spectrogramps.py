"""A simple example of creating a figure with text rendered in LaTeX."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc,rcParams

from librosa import load, stft
import librosa
import librosa.display

from skimage.feature import peak_local_max

def set_matplotlib_style():
    # Using seaborn's style
    plt.style.use('seaborn-whitegrid')

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Bold text
        "font.weight": "bold",
        "axes.labelweight": "bold",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 18,
        "font.size": 18,
        # Make the legend/label fonts a little smaller
        "legend.frameon": True,
        "legend.framealpha": 0.6,
        "legend.fontsize": 14,
        #"xtick.labelsize": 16,
        #"ytick.labelsize": 16,
        # colormap
        "image.cmap": 'viridis'
    }

    # matplotlib.rc('font', family='serif', serif='cm10')
    # matplotlib.rc('text', usetex=True)
    #rcParams['text.latex.preamble'] = [r'\boldmath']
    plt.rcParams.update(tex_fonts)

def plot_spectrogram(audio_path, title=None, start=None, end=None, linear=False):
    SAMPLE_RATE = 8000
    MONO = True
    WINDOW = 'hann'
    FFT_WINDOW_SIZE = 1024
    FFT_HOP_SIZE = 128  # 1 frame = 128 / 8000 = 0.016s
    PEAKS_DISTANCE=20
    COLORS = ['red', 'orange']

    audio_ts, _ = load(audio_path, sr=SAMPLE_RATE, mono=MONO)
    spectrogram = np.abs(stft(audio_ts,
                              window=WINDOW,
                              n_fft=FFT_WINDOW_SIZE,
                              hop_length=FFT_HOP_SIZE))
    s_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    peaks = peak_local_max(spectrogram, min_distance=PEAKS_DISTANCE)

    fig, ax = plt.subplots()
    
    if linear:
        yaxis = 'linear'
    else:
        yaxis = 'log'
    img = librosa.display.specshow(s_db, ax=ax,
                                   x_axis='time', y_axis=yaxis,
                                   cmap='viridis', sr=SAMPLE_RATE,
                                   hop_length=FFT_HOP_SIZE, n_fft=FFT_WINDOW_SIZE,
                                   auto_aspect=False)
    if start:
        if len(start) > 1:
            for i, (s,e) in enumerate(zip(start, end)):
                plt.axvline(x=s, color=COLORS[i])
                plt.axvline(x=e, color=COLORS[i])

    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(audio_path)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Plot spectrograms',
                    description='A pyhton script to plot spectrograms')
    parser.add_argument('audio_path', help='Path of the audio file.')
    parser.add_argument('outpath', help='Path where to save the figure.') 
    parser.add_argument('-t', '--title', help='Plot title.')
    parser.add_argument('-s', '--start', help='Match/es start timestamp (list).')
    parser.add_argument('-e', '--end', help='Match/es end timestamp (list).')
    parser.add_argument('-l', '--linear', action='store_true',
                        help='Y-axis scale. set --linear to make it linear, Logarithmic otherwise.')
    args = parser.parse_args()

    set_matplotlib_style()
    figure = plot_spectrogram(args.audio_path, title=args.title,
                              start=args.start, end=args.end, linear=args.linear)
    figure.savefig(args.outpath, format='jpg')