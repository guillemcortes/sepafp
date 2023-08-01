import torch
import pandas as pd
import torchaudio
import os, re
import numpy as np
import scipy
import random
import utils

class CombineDatasets(torch.utils.data.Dataset):
    """
    TODO
    """

    dataset_name = "SpeechMusicMix"

    def __init__(self, speech_dirs, music_dirs, sound_dirs=None, sample_rate=8000, original_sample_rate=8000, segment=9,
                 shuffle_tracks=False, multi_speakers=False,multi_speakers_frequency=5,data_ratio=1.,new_data=False,
                 sound_probability=0., mixwithspeech=True):
        self.eps = 1e-12
        self.min_file_size = 100000
        self.segment = segment
        # sample_rate of the original files
        self.original_sample_rate = original_sample_rate
        # destination sample_rate for resample
        self.sample_rate = sample_rate
        self.shuffle_tracks = shuffle_tracks
        self.multi_speakers = multi_speakers
        self.multi_speakers_frequency = multi_speakers_frequency
        self.data_ratio = data_ratio
        self.new_data = new_data
        self.mixwithspeech = mixwithspeech

        self.speech_dirs = [speech_dir for speech_dir in speech_dirs if os.path.isdir(speech_dir)]
        self.music_dirs = [music_dir for music_dir in music_dirs if os.path.isdir(music_dir)]
        assert len(self.speech_dirs)>0
        assert len(self.music_dirs)>0
        self.all_speech_files = [os.path.join(path,file.name) for path in self.speech_dirs for file in os.scandir(path)  if file.name[-4:]=='.wav' and file.stat().st_size>self.min_file_size and not file.name.startswith('.') and not file.is_dir()]
        self.all_music_files = [os.path.join(path,file.name) for path in self.music_dirs for file in os.scandir(path) if file.name[-4:]=='.wav' and file.stat().st_size>self.min_file_size and not file.name.startswith('.') and not file.is_dir()]
        print("Found {} speech and {} music files".format(len(self.all_speech_files),len(self.all_music_files)))
        self.n_speech = int(self.data_ratio*len(self.all_speech_files))
        self.n_music = int(self.data_ratio*len(self.all_music_files))
        self.speech_files = random.sample(self.all_speech_files,self.n_speech)
        self.music_files = random.sample(self.all_music_files,self.n_music)
        self.len_speech = len(self.speech_files)
        self.len_music = len(self.music_files)

        #other sounds
        if sound_dirs is not None:
            self.sound_dirs = [sound_dir for sound_dir in sound_dirs if os.path.isdir(sound_dir)]
            self.sound_probability = np.minimum(1., np.maximum(0., sound_probability))
            self.all_sound_files = [os.path.join(path,file.name) for path in self.sound_dirs for file in os.scandir(path) if file.name[-4:]=='.wav' and file.stat().st_size>self.min_file_size and not file.name.startswith('.') and not file.is_dir()]
            self.n_sound = int(self.data_ratio*len(self.all_sound_files))
            self.sound_files = random.sample(self.all_sound_files,self.n_sound)
            self.len_sounds = len(self.sound_files)
            print("Found {} sound files".format(len(self.all_sound_files)))
        else:
            self.sound_dirs = []
            self.sound_probability = 0.

        #for librispeech and vctk we can get a list of speakers from filenames
        self.set_speaker_ids()

        # initialize indexes
        self.speech_inxs = list(range(self.len_speech))
        self.music_inxs = list(range(self.len_music))
        self.sound_inxs = [] if self.sound_probability==0 else list(range(self.len_sounds))
        random.shuffle(self.music_inxs)
        random.shuffle(self.speech_inxs)
        random.shuffle(self.sound_inxs)

        # declare the resolution of the reduction factor.
        # this will create N different gain values max
        # 1/denominator_gain to multiply the music gain
        self.denominator_gain = 20
        self.gain_ramp = np.array(range(1, self.denominator_gain, 1))/self.denominator_gain

        self.list_music_gain = []
        self.list_mix_snr = []
        self.list_speech_snr = []

        # shuffle the static random gain to use it in testing
        np.random.shuffle(self.gain_ramp)

        # use soundfile as backend
        torchaudio.set_audio_backend(backend='sox_io')

    def set_speaker_ids(self):
        speakers = [re.split('_|-',sf.split(os.sep)[-1])[0] for sf in self.speech_files]
        speakers_set = list(set(speakers))
        print("Found {} speakers".format(len(speakers_set)))
        self.speakers = {speaker:[] for speaker in speakers_set}
        for i, x in enumerate(speakers):
            self.speakers[x].append(i)

    def __len__(self):
        return self.len_music

    def load_random_music(self, music_idx):
        """ Randomly selects a non_silent part of the audio given by audio_path

        Parameters:
        - audio_path (str) : path to the audio file

        Returns:
        - audio_signal (torchaudio) : waveform of the
        """
        seq_duration_samples = int(self.segment * self.original_sample_rate)
        info = torchaudio.info(self.music_files[music_idx])
        length_music = info.num_frames
        # path of music
        music_path = self.music_files[music_idx]
        music_signal = torch.zeros(seq_duration_samples)

        # take random segment
        if length_music > (seq_duration_samples+self.original_sample_rate//2):
            offset = random.randrange(self.original_sample_rate//2,length_music-seq_duration_samples)
            num_frames = seq_duration_samples
        else:
            offset = 0
            num_frames = np.minimum(length_music,seq_duration_samples)

        audio_signal,_ = torchaudio.load(
                self.music_files[music_idx]
            )
        audio_signal = audio_signal.squeeze()

        if length_music > (seq_duration_samples+self.original_sample_rate//2) and torch.count_nonzero(audio_signal)<length_music:
            patience = 10
            while torch.count_nonzero(audio_signal[offset:offset+num_frames])<num_frames and patience>0:
                offset = random.randrange(self.original_sample_rate//2,length_music-seq_duration_samples)
                patience = patience - 1
        music_signal[:num_frames] = audio_signal[offset:offset+num_frames] + torch.rand(num_frames)*self.eps


        return music_signal

    def rms(self, audio):
        """ computes the RMS of an audio signal
        """
        return torch.sqrt(torch.mean(audio ** 2))

    def load_speechs(self, speech_idx):
        """
        concatenates random speech files from the same speaker as speech_idx until
        obtaining a buffer with a length of at least the lenght of the
        input segment.
        If multispeaker is used, a single audio file from a different
        speaker is overlapped in a random position of the buffer, to emulate
        speakers interruptions once every 10 items.
        The buffer is shifted in a random position to prevent always getting
        buffers that starts with the beginning of a speech.
        Returns the shifted buffer with a length equal to segment.
        """

        array_size = self.original_sample_rate * self.segment
        speech_mix = torch.zeros(array_size)

        speech_counter = 0
        ### librispeech and vctk speaker id from filename
        speaker_id = self.speech_files[speech_idx].split(os.sep)[-1].split('-')[0]
        while speech_counter < array_size:
            # file is shorter than segment, concatenate with more until
            # is at least the same length
            if speech_counter == 0:
                new_speech_idx = speech_idx
            else:
                if self.multi_speakers and speech_idx % self.multi_speakers_frequency == 0:
                    new_speech_idx = random.sample(self.speech_inxs,1)[0]
                else:
                    new_speech_idx = random.sample(self.speakers[speaker_id],1)[0]

            audio_path =  os.path.join(self.speech_files[new_speech_idx])
            speech_signal, _ = torchaudio.load(
                audio_path
            )
            speech_length = speech_signal.shape[-1]

            ### the offset for the speech inside the mix is drawn from a poisson distribution with u=1
            offset = speech_counter+int(np.floor((array_size-speech_counter)*(utils.signals.gen_poisson(1) - 1)/10.))
            if offset+speech_length>array_size:
                # add the speech to the buffer
                #import pdb;pdb.set_trace()
                speech_mix[offset:] = speech_signal.squeeze()[:array_size-offset]
                speech_counter = array_size
            else:
                # add the speech to the buffer
                speech_mix[offset:offset+speech_length] = speech_signal.squeeze()
                speech_counter += offset+speech_length

        # we have a segment with the two speakers, the second in a random start.
        # now we randomly shift the array to pick the start
        offset = random.randint(0, array_size)
        zeros_aux = torch.zeros(len(speech_mix))
        aux = speech_mix[:offset]
        
        zeros_aux[:len(speech_mix) - offset] = speech_mix[offset:len(speech_mix)]
        zeros_aux[len(speech_mix) - offset:] = aux
        #import pdb;pdb.set_trace()
        return zeros_aux[:array_size]

    def load_sounds(self, sound_idx):
        """
        Returns the shifted buffer with a length equal to segment.
        """

        seq_duration_samples = int(self.segment * self.original_sample_rate)
        info = torchaudio.info(self.sound_files[sound_idx])
        length_sound = info.num_frames
        sound_signal = torch.zeros(seq_duration_samples)

        # take random segment
        if length_sound > (seq_duration_samples+self.original_sample_rate//2):
            offset = random.randrange(self.original_sample_rate//2,length_sound-seq_duration_samples)
            num_frames = seq_duration_samples
        else:
            offset = 0
            num_frames = np.minimum(length_sound,seq_duration_samples)

        audio_signal,_ = torchaudio.load(
                self.sound_files[sound_idx]
            )
        audio_signal = audio_signal.squeeze()

        if length_sound > (seq_duration_samples+self.original_sample_rate//2) and torch.count_nonzero(audio_signal)<length_sound:
            patience = 10
            while torch.count_nonzero(audio_signal[offset:offset+num_frames])<num_frames and patience>0:
                offset = random.randrange(self.original_sample_rate//2,length_sound-seq_duration_samples)
                patience = patience - 1
        sound_signal[:num_frames] = audio_signal[offset:offset+num_frames] + torch.rand(num_frames)*self.eps

        return sound_signal

    def data_shuffle(self):
        print("\nData shuffle")
        if self.data_ratio<1 and self.new_data:
            self.speech_files = random.sample(self.all_speech_files,self.n_speech)
            self.music_files = random.sample(self.all_music_files,self.n_music)
            self.len_speech = len(self.speech_files)
            self.len_music = len(self.music_files)
            #for librispeech and vctk we can get a list of speakers from filenames
            self.set_speaker_ids()

        # shuffle on first epochs of training and validation. Not testing
        random.shuffle(self.music_inxs)
        random.shuffle(self.speech_inxs)

    def __getitem__(self, idx):
        # get corresponding index from the list
        if self.len_speech<self.len_music:
            speech_idx = self.speech_inxs[idx%self.len_speech]
        else:
            speech_idx = self.speech_inxs[idx]
        if self.len_speech>self.len_music:
            music_idx = self.music_inxs[idx%self.len_music]
        else:
            music_idx = self.music_inxs[idx]

        sources_list = []

        # We want to cleanly separate Speech, so its the first source
        # in the sources_list
        music_signal = self.load_random_music(music_idx)
        speech_signal = self.load_speechs(speech_idx)
        # the third source, other sounds
        if random.random() < self.sound_probability:
            if self.len_sounds<self.len_music:
                sound_idx = self.sound_inxs[idx%self.len_sounds]
            else:
                sound_idx = self.sound_inxs[idx]
            sound_signal = self.load_sounds(sound_idx)
            # gain based on RMS in order to have RMS(speech_signal) >= RMS(sound_singal)
            reduction_factor = self.rms(speech_signal) / self.rms(sound_signal)
            # now we know that rms(r * music_signal) == rms(speech_signal)
            sound_gain = random.uniform(0.1, 0.5) * reduction_factor

            # multiply the music by the gain factor and add to the sources_list
            sound_signal = sound_gain * sound_signal

            if self.mixwithspeech:
                speech_signal = speech_signal + sound_signal
        else:
            sound_signal = None


        # gain based on RMS in order to have RMS(speech_signal) >= RMS(music_singal)
        reduction_factor = self.rms(speech_signal) / self.rms(music_signal)
        # now we know that rms(r * music_signal) == rms(speech_signal)
        music_gain = random.uniform(0.1, 0.5) * reduction_factor

        # multiply the music by the gain factor and add to the sources_list
        music_signal = music_gain * music_signal


        # append sources:
        if self.mixwithspeech:
            sources_list.append(0.5 * speech_signal)
            sources_list.append(0.5 * music_signal)
        else:
            sources_list.append(0.5 * speech_signal)
            sources_list.append(0.5 * music_signal)
            if self.sound_probability>0:
                if sound_signal is None:
                    sources_list.append(torch.zeros_like(speech_signal))
                else:
                    sources_list.append(0.5 * sound_signal)

        mixture = sum(sources_list)
        mixture = torch.squeeze(mixture)

        self.list_music_gain.append(music_gain.tolist())
        self.list_mix_snr.append(utils.signals.snr(mixture,0.5*music_signal).tolist())
        self.list_speech_snr.append(utils.signals.snr(speech_signal,music_signal).tolist())

        # Stack sources
        sources = torch.vstack(sources_list)

        ### save the training examples
        #self.save_example(idx, mixture, 'tmp'+os.sep+'tmp')

        return mixture, sources

    def get_infos(self):
        """Get dataset infos (for publishing models).
        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        return infos

    def save_example(self, idx, mixture, save_dir):
        outfile = os.path.join(save_dir,self.music_files[idx].split(os.sep)[-1])
        torchaudio.save(outfile,mixture[None,...],self.sample_rate)

