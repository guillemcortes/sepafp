import os, sys
import torch
import yaml
import argparse
import numpy as np
import multiprocessing
import itertools
import torchaudio
import asteroid
import utils
from utils import my_import
import functools
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path",
    type=str,
    required=True,
    help="Input audio file or path"
)
parser.add_argument(
    "--model",
    type=str,
    default='tmp/cfg/UNet_config/best_model.pth',
    required=True,
    help="The path to the serialized model to use"
)
parser.add_argument(
    "--out_dir",
    type=str,
    default='',
    help="Directory where the eval results will be stored. Leave empty to use the same as input.",
)
parser.add_argument(
    "--file_filter",
    type=str,
    default='',
    help="Select only the files containing this string. Leave empty to select all the files.",
)
parser.add_argument(
    "--sample_rate",
    type=int,
    default=8000,
    help="Resample at this sample rate"
)
parser.add_argument(
    "--normalize",
    default=False,
    action="store_true",
    help="Normalize the output audios")

torchaudio.set_audio_backend(backend='sox_io')


def process_file(pair,**kwargs):
    psubdir,pfilename = pair
    with torch.no_grad():
        # # Forward the network on the mixture.
        #print(os.path.join(psubdir,pfilename))
        with open(os.path.join(psubdir,pfilename), 'rb') as wf:
            audio_signal,sr = torchaudio.load(wf)

        if len(audio_signal.shape)>1:
            audio_signal = audio_signal.mean(dim=0,keepdim=True)

        if sr != kwargs["sample_rate"]:
            resampler = utils.Resampler(
                input_sr=sr,
                output_sr=kwargs["sample_rate"],
                dtype=torch.float32,
                filter='hann'
            )
            #import pdb;pdb.set_trace()
            audio_signal = resampler.forward(audio_signal)

        if kwargs["target_model"] == "Waveunet" or kwargs["target_model"] == "UNet" or kwargs["target_model"] == "OUMX" or kwargs["target_model"] == "UNetAttn":
            audio_signal = audio_signal.unsqueeze(0)
        if kwargs["target_model"] == "UMX":
            audio_signal = audio_signal.unsqueeze(0)
            audio_signal = audio_signal.unsqueeze(0)

        audio_signal.to(torch.device(kwargs['device']))
        est_sources = kwargs["continuous_nnet"].forward(audio_signal)
        if isinstance(est_sources, tuple) and kwargs["target_model"] == "UMX":
            est_sources = est_sources[1]

        elif kwargs["target_model"] == "UMX":
            est_sources = est_sources.squeeze()

        if kwargs["normalize"]:
            norm, _ = torch.abs(est_sources).max(dim=2,keepdim=True)
            est_sources =  est_sources / norm

        fext = pfilename.split('.')[-1]
        fname = pfilename.replace(fext,'')[:-1]

        if kwargs["singlefile"] is not None:
            eval_save_dir_idx = kwargs["eval_save_dir"]
        else:
            eval_save_dir_idx = psubdir.replace(kwargs["input_path"],kwargs["eval_save_dir"])
        suffix = '-music'
        if kwargs["out_dir"] != '':
            os.makedirs(eval_save_dir_idx, exist_ok=True)
        else:
            suffix += '-'+kwargs["experiment_name"]
        filename_music = os.path.join(eval_save_dir_idx,fname+suffix+'.'+fext)
        torchaudio.save(filepath=filename_music, src=est_sources[:,1,:].to(device=torch.device('cpu')), sample_rate=kwargs["sample_rate"])


def main(conf):
    model_path = conf["model"]
    conf["experiment_name"] = os.path.basename(conf["train_conf"]["main_args"]["config"]).split(".yml")[0]
    conf["target_model"] = conf["train_conf"]["model"]["name"]
    AsteroidModelModule = utils.my_import("models."+conf["target_model"])
    model = AsteroidModelModule.from_pretrained(model_path, sample_rate=conf["sample_rate"])
    model._return_time_signals = True

    # Handle device placement
    print(conf["device"])
    device = torch.device(conf["device"])
    model.device = torch.device(conf["device"])
    model.to(device=device)
    model.eval()

    conf["continuous_nnet"] = asteroid.dsp.overlap_add.LambdaOverlapAdd(
        nnet=model,  # function to apply to each segment.
        n_src=2,  # number of sources in the output of nnet
        window_size=conf["sample_rate"]*conf["segment"],  # Size of segmenting window
        hop_size=None,  # segmentation hop size
        window="hann",  # Type of the window (see scipy.signal.get_window
        reorder_chunks=True,  # Whether to reorder each consecutive segment.
        enable_grad=False,  # Set gradient calculation on of off (see torch.set_grad_enabled)
    )
    conf["continuous_nnet"].window = conf["continuous_nnet"].window.to(device=device)

    #import pdb;pdb.set_trace()
    if os.path.isfile(conf["input_path"]):
        inlistpath = os.path.dirname(conf["input_path"])
        conf["singlefile"] = conf["input_path"]
    elif os.path.isdir(conf["input_path"]):
        inlistpath = conf["input_path"]
        conf["singlefile"] = None
    else:
        print("input_path is not valid.\n")
        sys.exit(1)

    if conf["out_dir"] != '':
        conf["eval_save_dir"] = os.path.join(conf["out_dir"], conf["experiment_name"])
    else:
        conf["eval_save_dir"] = inlistpath

    filelist = []
    for subdir, dirs, files in os.walk(inlistpath):
        if conf["singlefile"] is not None:
            filelist = [(subdir,filename) for filename in files if filename in conf["singlefile"]]
        else:
            filelist.extend([(subdir,filename) for filename in files if conf["file_filter"] in filename])

    if conf["singlefile"] is not None:
        process_file(filelist[0],**conf)
    else:

        for pair in tqdm.tqdm(filelist):
            process_file(pair,**conf)


if __name__ == "__main__":
    parser.add_argument('--disable_cuda', action='store_true',help='Disable CUDA')
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    assert os.path.isfile(args.model)
    config_model = os.path.join(os.path.dirname(args.model), "conf.yml")
    assert os.path.isfile(config_model)
    # Load training config
    with open(config_model) as f:
        train_conf = yaml.safe_load(f)

    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["segment"] = train_conf["data"]["segment"]
    arg_dic["train_conf"] = train_conf
    arg_dic["device"] = {}
    if not args.disable_cuda and torch.cuda.is_available():
        arg_dic["device"] = 'cuda'
    else:
        arg_dic["device"] = 'cpu'

    main(arg_dic)

 # python predict.py --input_path /path/to/input/audios/ --model /path/to/best_model.pth --out_dir /path/to/output/dir/



