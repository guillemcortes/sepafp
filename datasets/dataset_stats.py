import copy
import sklearn
from sklearn import preprocessing
import torch
import tqdm 
import numpy as np
import asteroid
from asteroid.models.x_umx import _STFT, _Spectrogram

def get_statistics(conf, dataset):
    scaler = sklearn.preprocessing.StandardScaler()

    spec = torch.nn.Sequential(
        _STFT(window_length=conf["model"]["window_length"], n_fft=conf["model"]["in_chan"], n_hop=conf["model"]["nhop"]),
        _Spectrogram(spec_power=conf["model"]["spec_power"], mono=True),
    )

    dataset_scaler = copy.deepcopy(dataset)
    pbar = tqdm.tqdm(range(len(dataset_scaler)))
    for ind in pbar:
        x, _ = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        X = spec(x[None, None, ...])[0]
        scaler.partial_fit(np.squeeze(X))

    # set inital input scaler values
    std = np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_))
    return scaler.mean_, std
