# Source Separation Models for Audio Fingerprinting

This repository contains the code described in the publication "Enhanced TV Broadcast Monitoring with Source Separation-Assisted Audio Fingerprinting". Here you can find the code of the source separation models developed for the publication. The model checkpoints are hosted in a [GDrive folder](https://drive.google.com/drive/folders/1RFwT0moQcMSMxvI9bz6hR9tj6Gr6UTBL?usp=sharing) (about 600MB).

To fully reproduce the work in the publication, other repositories might be needed:
* Audfprint: https://github.com/dpwe/audfprint
* Panako / Olaf: https://github.com/JorenSix/panako (at its [2.1 version release](https://github.com/JorenSix/Panako/commit/20d94a2444677c9e4917dbb1a881a2599b657ba0))
* NeuralFP: https://github.com/guillemcortes/neural-audio-fp (forked from https://github.com/mimbres/neural-audio-fp)

And Datasets:
* BAF dataset: https://github.com/guillemcortes/baf-dataset
* LibriSpeech
* MillionSongDataset (MSD)
* FSD50K

🎶 👂 This repository also comes with a [webpage](https://guillemcort.es/sepafp/) where readers can listen to some examples of audios separated by the separation models.

# Installation
The authors recommend the use of virtual environments.

**Requirements:**
* Python 3.6+
* Create virtual environment and install requirements

```bash
git clone https://github.com/guillemcortes/ssafp.git
cd ssafp
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Usage
**Training**
```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py --config /path/to/cfg.yml
```
**Inference**
```bash
python predict.py --input_path /path/to/input/audios/ --model /path/to/best_model.pth --out_dir /path/to/output/dir/
```

# License
* The code in this repository is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

# Citation
Please cite the following [publication](https://doi.org/10.5281/zenodo.7372162) when using the dataset:

> TBC

Bibtex version:

```
TBC
```

# Acknowledgements

This research is part of *NextCore – New generation of music monitoring technology (RTC2019-007248-7)*, funded by the Spanish Ministerio de Ciencia e Innovación and the Agencia Estatal de Investigación and part of *resCUE – Smart system for automatic usage reporting of musical works in audiovisual productions (SAV-20221147)* funded by CDTI and the Ministerio de Asuntos Económicos y Transformación Digital. Also, it has received support from Industrial Doctorates plan of the Secretaria d’universitats i Recerca, Departament d’Empresa i Coneixement de la Generalitat de Catalunya, grant agreement No. DI46-2020.

<p align="center">
  <img src="https://user-images.githubusercontent.com/25322884/186637988-7bb1f775-2eac-4110-9961-ad7bbf8cb520.png" height="50" hspace="10" vspace="10" />
  <img src="https://user-images.githubusercontent.com/25322884/186637802-f9c8bbb9-bcf2-4b5e-9407-1fdd49c9aa9b.jpg" height="50" hspace="10" vspace="10"/>
</p>
<p align="center">
  <img src="https://github.com/guillemcortes/ssafp/assets/25322884/8970673f-4501-4cc9-9fef-d6ccd147d56a" height="50" hspace="10" vspace="10" />
  <img src="https://github.com/guillemcortes/ssafp/assets/25322884/7794037b-3bb4-4679-926c-b3a45250af60" height="50" hspace="10" vspace="10"/>
  <img src="https://github.com/guillemcortes/ssafp/assets/25322884/4463ef7a-320c-4cfd-b334-9cd140a0ba04" height="50" hspace="10" vspace="10"/>
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/25322884/186637854-50e06004-9dc6-40ee-8ec9-701899136a6e.png" height="50" hspace="10" vspace="10"/>
  <img src="https://user-images.githubusercontent.com/25322884/186637746-e18c4517-250c-4474-b11e-58df1e1f0787.jpeg" height="50" hspace="10" vspace="10"/>
  <img src="https://user-images.githubusercontent.com/25322884/186637861-24a64957-f82b-4faa-be34-5b1221bbd05c.png" height="50" hspace="10" vspace="10"/>
</p>
