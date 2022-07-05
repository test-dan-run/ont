import json
import librosa
import os

TRAIN_DIR = '/dataset/28spk/noisy_train'
DEV_DIR = '/dataset/28spk/noisy_dev'

with open('/dataset/28spk/train_manifest.json', mode='w', encoding='utf-8') as f:
    for filename in os.listdir(TRAIN_DIR):
        item = {'audio_filepath': os.path.join('noisy_train', filename), 'duration': librosa.get_duration(filename=os.path.join(TRAIN_DIR, filename))}
        f.write(json.dumps(item)+'\n')

with open('/dataset/28spk/dev_manifest.json', mode='w', encoding='utf-8') as f:
    for filename in os.listdir(DEV_DIR):
        item = {'audio_filepath': os.path.join('noisy_dev', filename), 'duration': librosa.get_duration(filename=os.path.join(DEV_DIR, filename))}
        f.write(json.dumps(item)+'\n')