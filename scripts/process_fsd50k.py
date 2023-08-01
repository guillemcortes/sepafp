import soundata
import subprocess
import argparse
import os

class FSD50K():

    def __init__(self, out_path):
        assert os.path.exists(out_path)
        self.out_path = out_path
        self.dataset = soundata.initialize('fsd50k')
        self.allowed_tags_as = ["Human_locomotion","Hands","Human_group_actions","Surface_contact","Deformable_shell",
            "Onomatopoeia","Animal","Vehicle","Engine","Domestic sounds, home sounds","Mechanisms","Tools",
            "Explosion","Wood","Glass","Liquid","Specific_impact_sounds","Wind","Thunderstorm","Water","Fire",
            ] #"Generic impact sounds","Bell","Alarm","Miscellaneous sources",
        self.all_tags = ['Animal', 'Domestic_sounds_and_home_sounds', 'Music', 'Vehicle', 'Tearing', 'Human_voice',
            'Liquid', 'Explosion', 'Bell', 'Engine', 'Tap', 'Thump_and_thud', 'Burping_and_eructation',
            'Crumpling_and_crinkling', 'Walk_and_footsteps', 'Thunderstorm', 'Water', 'Fart', 'Squeak',
            'Respiratory_sounds', 'Human_group_actions', 'Crushing', 'Fire', 'Alarm', 'Glass', 'Chewing_and_mastication',
            'Whoosh_and_swoosh_and_swish', 'Mechanisms', 'Hands', 'Tools', 'Crack', 'Motor_vehicle_(road)',
            'Livestock_and_farm_animals_and_working_animals', 'Wind', 'Bird_vocalization_and_bird_call_and_bird_song',
            'Rattle', 'Speech', 'Hiss', 'Singing', 'Keyboard_(musical)', 'Screech', 'Wood', 'Shout', 'Run', 'Drum',
            'Crackle', 'Tick', 'Percussion', 'Plucked_string_instrument', 'Domestic_animals_and_pets', 'Insect',
            'Wild_animals', 'Cymbal', 'Pour', 'Laughter', 'Guitar', 'Door', 'Car', 'Bird', 'Rail_transport',
            'Aircraft', 'Rain', 'Power_tool', 'Ocean', 'Cough', 'Bowed_string_instrument', 'Clock', 'Drum_kit',
            'Mallet_percussion', 'Breathing', 'Cat', 'Typing', 'Brass_instrument', 'Telephone']
        self.allowed_tags = ['Animal', 'Domestic_sounds_and_home_sounds', 'Vehicle', 'Tearing', 'Human_voice',
            'Liquid', 'Bell', 'Engine', 'Tap', 'Thump_and_thud',  'Domestic_animals_and_pets',
            'Crumpling_and_crinkling', 'Walk_and_footsteps', 'Thunderstorm', 'Water', 'Squeak', 'Insect',
            'Respiratory_sounds', 'Human_group_actions', 'Crushing', 'Fire', 'Alarm', 'Glass', 'Chewing_and_mastication',
            'Whoosh_and_swoosh_and_swish', 'Mechanisms', 'Hands', 'Tools', 'Crack', 'Motor_vehicle_(road)',
            'Livestock_and_farm_animals_and_working_animals', 'Wind', 'Bird_vocalization_and_bird_call_and_bird_song',
            'Rattle', 'Speech', 'Hiss', 'Screech', 'Wood', 'Shout', 'Run', 'Crackle', 'Tick',
            'Wild_animals', 'Pour', 'Laughter', 'Door', 'Car', 'Bird', 'Rail_transport',
            'Aircraft', 'Rain', 'Power_tool', 'Ocean', 'Cough', 'Clock', 'Breathing', 'Cat', 'Typing', 'Telephone']


    def download(self):
        self.dataset.download()
        self.dataset.validate()

    def export(self):
        tags = {t:0 for t in self.allowed_tags}
        clip_dict = self.dataset.load_clips()
        os.makedirs(os.path.join(self.out_path,'train'),exist_ok=True)
        os.makedirs(os.path.join(self.out_path,'test'),exist_ok=True)
        os.makedirs(os.path.join(self.out_path,'validation'),exist_ok=True)
        for i in self.dataset.clip_ids:
            #if clip_dict[i].split == 'test':
            if clip_dict[i].tags.labels[-1] in self.allowed_tags:
                tags[clip_dict[i].tags.labels[-1]] += 1
                command_name = ['ffmpeg','-v','error','-y','-i'] #fpmatcher identify -q mixture.wav.fp2v2 -r music.fp2 -c fp2v2
                command_name.append(clip_dict[i].audio_path)
                command_name.extend(['-ar','8000','-ac','1','-acodec','pcm_s16le','-af','aresample=async=1'])
                command_name.append(os.path.join(self.out_path,clip_dict[i].split,clip_dict[i].tags.labels[-1]+'-'+str(i)+'.wav'))
                #import pdb;pdb.set_trace()
                result = subprocess.run(command_name, stdout=subprocess.PIPE, universal_newlines=True)
                #ffmpeg -v error -y -i $f -ar 8000 -ac 1 -acodec pcm_s16le -af aresample=async=1 "$OUT_PATH/$s/$filename.wav"
            # if clip_sdict[i].tags.labels[-1] in tags:
            #     tags[clip_dict[i].tags.labels[-1]] += 1
            # else:
            #     tags[clip_dict[i].tags.labels[-1]] = 0
            # #print(clip_dict[i].tags.labels[-1])
        import pdb;pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", type=str, required=True, help="the path where to write the parsed dataset"
    )
    args = parser.parse_args()
    fsd = FSD50K(args.out_path)
    #fsd.download()
    fsd.export()

