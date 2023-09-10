import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import re
import os

dicts = {'01' : 'neutral', '02' : 'calm', '03' : 'happy', '04' : 'sad', '05' : 'angry', '06' : 'fearful', '07' : 'disgust', '08' : 'surprised'}

path_main = 'Audio_Speech_Actors_01-24'

folders_main = os.listdir(path_main)

counter = 0
for folders in folders_main : 
    path_in = 'Audio_Speech_Actors_01-24/{0}'.format(folders)
    files_sub = os.listdir(path_in)
    for file in files_sub :
        numbers = re.findall('\d+', file)
        emotion = dicts[numbers[2]]
        print(numbers[6], emotion)
        path_save = 'sorted_data/{0}/{1}.jpeg'.format(emotion, file)
        path_load = '{0}/{1}'.format(path_in, file)
        y, sr = librosa.load(path_load)
        yt, _ = librosa.effects.trim(y)
        y = yt
        mel_spect = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = 1024, hop_length = 100)
        mel_spect = librosa.power_to_db(mel_spect, ref = np.max)
        librosa.display.specshow(mel_spect, y_axis = 'mel', fmax = 20000, x_axis = 'time')
        plt.savefig(path_save)
