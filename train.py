import os, sys
import argparse
import json
import yaml

import torch
import pytorch_lightning as pl
#from neptune.new.integrations.pytorch_lightning import NeptuneLogger
from pytorch_lightning.loggers import WandbLogger
import asteroid
from asteroid import engine
import numpy as np
import pandas as pd
import datasets
import models
import losses
import utils



pl.seed_everything(1, workers=True)

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_dir",
    default="tmp",
    help="Full path to save experiment results"
)

def main(conf):

    # neptune_logger = NeptuneLogger(name=conf["model"]["name"],
    #     project=conf["neptune"]["project"],
    #     api_key=conf["neptune"]["api_key"],
    # )  # your credentials
    wandb_logger = WandbLogger(name=conf["neptune"]["expname"],project=conf["neptune"]["project"])

    dataloader_kwargs = (
        {"num_workers": conf["training"]["num_workers"], "pin_memory": True} if "gpu" in arg_dic["main_args"]["device"] else {}
    )

    train_set = datasets.CombineDatasets(
        speech_dirs=conf["data"]["speech_train_dir"],
        music_dirs=conf["data"]["music_train_dir"],
        sound_dirs=conf["data"]["sound_train_dir"] if "sound_train_dir" in conf["data"] else None,
        sample_rate=conf["data"]["sample_rate"],
        original_sample_rate=conf["data"]["original_sample_rate"],
        segment=conf["data"]["segment"],
        shuffle_tracks=True,
        multi_speakers=conf["training"]["multi_speakers"],
        multi_speakers_frequency=conf["training"]["multi_speakers_frequency"],
        data_ratio=conf["training"]["data_ratio"] if "data_ratio" in conf["training"] else 1.,
        new_data=conf["training"]["new_data"] if "new_data" in conf["training"] else False,
        sound_probability=conf["data"]["sound_probability"] if "sound_probability" in conf["data"] else 0.,
        mixwithspeech=conf["data"]["mixwithspeech"] if "mixwithspeech" in conf["data"] else True,
    )
    val_set = datasets.CombineDatasets(
        speech_dirs=conf["data"]["speech_valid_dir"],
        music_dirs=conf["data"]["music_valid_dir"],
        sound_dirs=conf["data"]["sound_valid_dir"] if "sound_valid_dir" in conf["data"] else None,
        sample_rate=conf["data"]["sample_rate"],
        original_sample_rate=conf["data"]["original_sample_rate"],
        segment=conf["data"]["segment"],
        shuffle_tracks=False,
        multi_speakers=conf["training"]["multi_speakers"],
        multi_speakers_frequency=conf["training"]["multi_speakers_frequency"],
        data_ratio=1.,
        new_data=False,
        sound_probability=conf["data"]["sound_probability"] if "sound_probability" in conf["data"] else 0.,
        mixwithspeech=conf["data"]["mixwithspeech"] if "mixwithspeech" in conf["data"] else True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        drop_last=True,
        **dataloader_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        drop_last=True,
        **dataloader_kwargs
    )
    
    ###Models
    if(conf["model"]["name"] == "ConvTasNet"):
        conf["masknet"].update({"n_src": conf["data"]["n_src"]})
        model = models.ConvTasNetNorm(
            conf["filterbank"],
            conf["masknet"],
            sample_rate=conf["data"]["sample_rate"],
            device=arg_dic["main_args"]["device"]
        )
        #loss_func = losses.LogL2Time()
        plugins = pl.plugins.DDPPlugin(find_unused_parameters=False)
    elif (conf["model"]["name"] == "UNet"):
        # UNet with logl2 time loss and normalization inside model
        model = models.UNet(
            conf["data"]["sample_rate"],
            conf["data"]["fft_size"],
            conf["data"]["hop_size"],
            conf["data"]["window_size"],
            conf["model"]["kernel_size"],
            conf["model"]["stride"],
            #sources=conf["data"]["sources"],
            device=arg_dic["main_args"]["device"],
            mask_logit=conf["model"]["mask_logit"]
        )
        #loss_func = losses.LogL2Time_weighted(weights=[0.1,0.9])
        plugins = pl.plugins.DDPPlugin(find_unused_parameters=False)
    elif(conf["model"]["name"] == "OUMX"):
        scaler_mean, scaler_std = datasets.get_statistics(conf, val_set)
        max_bin = utils.bandwidth_to_max_bin(conf["data"]["sample_rate"], conf["model"]["in_chan"], conf["model"]["bandwidth"])
        #scaler_mean, scaler_std = np.array([0]*max_bin), np.array([0.5]*max_bin)
        model = models.OUMX(
            window_length=conf["model"]["window_length"],
            input_mean=scaler_mean,
            input_scale=scaler_std,
            nb_channels=conf["model"]["nb_channels"],
            hidden_size=conf["model"]["hidden_size"],
            in_chan=conf["model"]["in_chan"],
            n_hop=conf["model"]["nhop"],
            sources=conf["data"]["sources"],
            max_bin=max_bin,
            bidirectional=conf["model"]["bidirectional"],
            sample_rate=conf["data"]["sample_rate"],
            spec_power=conf["model"]["spec_power"],
        )
        plugins=None
    elif(conf["model"]["name"] == "UNetAttn"):
        #scaler_mean, scaler_std = datasets.get_statistics(conf, val_set)
        max_bins = utils.bandwidth_to_max_bin(conf["data"]["sample_rate"], conf["model"]["in_chan"], conf["model"]["bandwidth"])
        #scaler_mean, scaler_std = np.array([0]*max_bin), np.array([0.5]*max_bin)
        model = models.UNetAttn(
            window_length=conf["model"]["window_length"],
            nb_channels=conf["model"]["nb_channels"],
            in_chan=conf["model"]["in_chan"],
            n_hop=conf["model"]["nhop"],
            max_bins=max_bins,
            sample_rate=conf["data"]["sample_rate"],
            spec_power=conf["model"]["spec_power"],
            n_src=conf["data"]["n_src"],
            mask_logit=conf["model"]["mask_logit"],
            k=conf["model"]["k"],
            hidden_size=conf["model"]["hidden_size"],
            device=arg_dic["main_args"]["device"],
        )
        plugins = pl.plugins.DDPPlugin(find_unused_parameters=False)
    elif (conf["model"]["name"] == "Waveunet"):
        num_features = [conf["model"]["features"]*i for i in range(1, conf["model"]["levels"]+1)] if conf["model"]["feature_growth"]== "add" else \
                        [conf["model"]["features"]*2**i for i in range(0, conf["model"]["levels"])]
        target_outputs = int(conf["data"]["segment"] * conf["data"]["sample_rate"])
        model = models.Waveunet(
            num_inputs=conf["model"]["channels"],
            num_channels=num_features,
            num_outputs=conf["model"]["channels"],
            instruments=conf["data"]["sources"],
            kernel_size=conf["model"]["kernel_size"],
            target_output_size=target_outputs,
            depth=conf["model"]["depth"],
            strides=conf["model"]["stride"],
            conv_type=conf["model"]["conv_type"],
            res=conf["model"]["res"],
            separate=conf["model"]["separate"],
            device=arg_dic["main_args"]["device"],
            sample_rate=conf["data"]["sample_rate"]
        )
    elif (conf["model"]["name"] == "Demucsv2"):
        model = models.Demucsv2(
            sources=conf["data"]["sources"],
            audio_channels=conf["model"]["audio_channels"],
            channels=conf["model"]["channels"],
            depth=conf["model"]["depth"],
            rewrite=True,
            glu=conf["model"]["glu"],
            rescale=conf["model"]["rescale"],
            resample=False,
            kernel_size=conf["model"]["kernel_size"],
            stride=conf["model"]["stride"],
            growth=conf["model"]["growth"],
            lstm_layers=conf["model"]["lstm_layers"],
            context=conf["model"]["context"],
            normalize=True,
            samplerate=conf["data"]["sample_rate"],
            segment_length=len(conf["data"]["sources"]) * conf["data"]["segment"] * conf["data"]["sample_rate"],
            device=arg_dic["main_args"]["device"])


    ###Losses
    loss_module = utils.my_import("losses."+conf["training"]['loss'])
    if 'weighted' in conf["training"]['loss']:
        loss_func = loss_module(weights=np.array(conf["training"]['class_weights'].split(';'),dtype=np.float32))
    elif 'MultiDomain' in conf["training"]['loss']:
        loss_func = losses.MultiDomainLoss(
            window_length=conf["model"]["window_length"],
            in_chan=conf["model"]["in_chan"],
            n_hop=conf["model"]["nhop"],
            spec_power=conf["model"]["spec_power"],
            nb_channels=conf["model"]["nb_channels"],
            loss_combine_sources=conf["training"]['loss_combine_sources'],
            loss_use_multidomain=conf["training"]['loss_use_multidomain'],
            mix_coef=conf["training"]['mix_coef'],
        )
    else:
        loss_func = loss_module()

    ###Optimizer
    optimizer = asteroid.engine.optimizers.make_optimizer(model.parameters(), lr=conf["optim"]["lr"], weight_decay=conf["optim"]["weight_decay"])
    if (conf["model"]["name"] == "Waveunet"):
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer, cycle_momentum=False, base_lr=conf["optim"]["min_lr"], max_lr=conf["optim"]["lr"],step_size_up=conf["optim"]["cycles"],
        )
    elif conf["training"]["half_lr"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=conf["optim"]["lr_decay_gamma"], patience=conf["optim"]["lr_decay_patience"], cooldown=10
        )

    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = os.path.join(conf["main_args"]["exp_dir"],conf["main_args"]['config'].split('.yml')[0])
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    ### delete paths key which raises error
    if "speech_train_dir" in conf["data"]: del conf["data"]["speech_train_dir"]
    if "speech_valid_dir" in conf["data"]: del conf["data"]["speech_valid_dir"]
    if "music_train_dir" in conf["data"]: del conf["data"]["music_train_dir"]
    if "music_valid_dir" in conf["data"]: del conf["data"]["music_valid_dir"]
    if "sound_train_dir" in conf["data"]: del conf["data"]["sound_train_dir"]
    if "sound_valid_dir" in conf["data"]: del conf["data"]["sound_valid_dir"]
    if 'sources' in conf["data"]: conf["data"].pop("sources")

    ###Systems
    # if "system" in conf["training"]:
    #     system_module = utils.my_import("systems."+conf["training"]['system'])
    #     system = system_module(
    #         model=model,
    #         loss_func=loss_func,
    #         optimizer=optimizer,
    #         train_loader=train_loader,
    #         val_loader=val_loader,
    #         scheduler=scheduler,
    #         config=conf,
    #         val_dur=conf["training"]['val_dur'],
    # )
    # else:
    system = asteroid.engine.system.System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf
    )

    # Define callbacks
    class shuffleData(pl.callbacks.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            trainer.train_dataloader.dataset.datasets.data_shuffle()

    class saveSNR(pl.callbacks.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            list_music_gain = trainer.train_dataloader.dataset.datasets.list_music_gain
            list_mix_snr = trainer.train_dataloader.dataset.datasets.list_mix_snr
            list_speech_snr = trainer.train_dataloader.dataset.datasets.list_speech_snr
            df = pd.DataFrame(list(zip(list_music_gain, list_mix_snr, list_speech_snr)),columns =['music_gain', 'mix_snr', 'speech_snr'])
            df.to_csv(os.path.join(exp_dir,'music_gains.csv'), "w")

    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    checkpoint = pl.callbacks.ModelCheckpoint(
        checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=conf["optim"]["patience"],
            verbose=True
        ))
    callbacks.append(saveSNR())
    if "data_ratio" in conf["training"] and conf["training"]["data_ratio"]<1:
        callbacks.append(shuffleData())
    if conf["main_args"]["load"]:
        checkpoint_files = [os.path.join(checkpoint_dir, x) for x in os.listdir(checkpoint_dir) if x.endswith(".ckpt")]
        if len(checkpoint_files)>0:
            newest_checkpoint = max(checkpoint_files, key = os.path.getctime)
        else:
            newest_checkpoint = None
    else:
        newest_checkpoint = None

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() and not conf["main_args"]["disable_cuda"] else None
    #distributed_backend = "ddp" if torch.cuda.is_available() and not conf["main_args"]["disable_cuda"] else None
    distributed_backend = "ddp" if torch.cuda.is_available() and not conf["main_args"]["disable_cuda"] else None
    accelerator = "gpu" if torch.cuda.is_available() and not conf["main_args"]["disable_cuda"] else 'cpu'
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        gpus=gpus,
        gradient_clip_val=5.0,
        resume_from_checkpoint=newest_checkpoint,
        precision=32,
        logger= wandb_logger,
        #plugins=plugins,
        #limit_train_batches=1.,  # Useful for fast experiment
        #strategies=distributed_backend,
        #accelerator=distributed_backend,
        reload_dataloaders_every_n_epochs=1 if "new_data" in conf["training"] and conf["training"]["new_data"] else 0
    )#logger=neptune_logger
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        print(best_k,f)
        json.dump(best_k, f, indent=0)
    print(checkpoint.best_model_path)
    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))

    #train_set.list_music_gain
    #train_set.list_mix_snr


if __name__ == "__main__":

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    parser.add_argument(
        "--config", type=str, required=True, help="the config file for the experiments"
    )
    parser.add_argument('--disable-cuda', action='store_true',help='Disable CUDA')
    parser.add_argument(
        "--load",
        type=bool,
        default=False,
        help="restore the most recent checkpoint with .ckpt extension"
    )
    config_model = sys.argv[2]
    with open(config_model) as f:
        def_conf = yaml.safe_load(f)
    parser = asteroid.utils.prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = asteroid.utils.parse_args_as_dict(parser, return_plain_args=True)
    arg_dic["main_args"]["device"] = {}
    if not arg_dic["main_args"]["disable_cuda"] and torch.cuda.is_available():
        arg_dic["main_args"]["device"] = 'cuda'
    else:
        arg_dic["main_args"]["device"] = 'cpu'

    main(arg_dic)

## CUDA_VISIBLE_DEVICES=0,1 python train.py --config cfg/UNet_config.yml