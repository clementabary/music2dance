from subprocess import call
import os
from tqdm import tqdm


def resample(audio_in, audio_out, rate):
    call("sox " + audio_in + " -r " + str(rate) + " " + audio_out, shell=True)
    return True


name = './Music-to-Dance-Motion-Synthesis-master'
rate = 16000

for directory in tqdm(os.listdir('{}'.format(name))):
    directory = '{}/{}'.format(name, directory)
    if os.path.isdir(directory) and os.path.basename(directory)[0:5] == 'DANCE':
        with open(directory + '/config.json') as f:
            resample(directory + '/audio_extract.wav', directory +
                     '/resampled_audio_extract.wav', rate)
