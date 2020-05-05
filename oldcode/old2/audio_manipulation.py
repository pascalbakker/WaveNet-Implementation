import tensorflow as tf
import librosa
import random
import os
import sys
import time
import numpy as np
from keras.callbacks import Callback
from scipy.io.wavfile import write
import fnmatch
import re

FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'

def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def load_generic_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    id_reg_exp = re.compile(FILE_PATTERN)
    print("files length: {}".format(len(files)))
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        ids = id_reg_exp.findall(filename)
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            category_id = int(ids[0][0])
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, filename, category_id

def frame_generator(sr, audio, frame_size, frame_shift, minibatch_size=20):
    audio_len = len(audio)
    X = []
    y = []
    while 1:
        for i in range(0, audio_len - frame_size - 1, frame_shift):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                break
            if i + frame_size >= audio_len:
                break
            temp = audio[i + frame_size]
            target_val = int((np.sign(temp) * (np.log(1 + 256 * abs(temp)) / (
                np.log(1 + 256))) + 1) / 2.0 * 255)
            X.append(frame.reshape(frame_size, 1))
            y.append((np.eye(256)[target_val]))
            if len(X) == minibatch_size:
                yield np.array(X), np.array(y)
                X = []
                y = []


class AudioManipulation():
    def __init__(self, path_to_source_folder, path_to_output_folder):
        super().__init__()
        self.path_to_source_folder = path_to_source_folder
        self.path_to_output_folder = path_to_output_folder

    def readMp3(self, numSamples=2):

        sr = 8000
        random_sampled = random.sample(os.listdir("C:\\Users\\pasca\Desktop\\audioProject\\data\\clips\\"), numSamples)
        audio = []

        for file_name in random_sampled:
            full_path_name = self.path_to_source_folder + "\\" + file_name
            data = None
            if sr == None:
                data, sr = librosa.load(full_path_name)
            else:
                data, sr = librosa.load(full_path_name, sr=sr)
            audio = audio + data.tolist()
            #audio.append(data)

        audio = np.asarray(audio)
        audio = audio.astype(float)
        audio = audio - audio.min()
        audio = audio / (audio.max() - audio.min())
        audio = (audio - 0.5) * 2

        print("Sample Audio Length: ", len(audio))

        return sr, audio


def get_audio_from_model(model, sr, duration, seed_audio, frame_size):
    print('Generating audio...')
    new_audio = np.zeros((sr * duration))
    curr_sample_idx = 0
    print("\t Generating Prediction of shape",new_audio.shape[0])
    while curr_sample_idx < new_audio.shape[0]:
        prediction = model.predict(seed_audio.reshape(1, frame_size, 1))
        distribution = np.array(prediction, dtype=float).reshape(256)
        distribution /= distribution.sum().astype(float)
        predicted_val = np.random.choice(range(256), p=distribution)
        ampl_val_8 = ((((predicted_val) / 255.0) - 0.5) * 2.0)
        ampl_val_16 = (np.sign(ampl_val_8) * (1 / 256.0) * ((1 + 256.0) ** abs(
            ampl_val_8) - 1)) * 2 ** 15
        new_audio[curr_sample_idx] = ampl_val_16
        seed_audio[:-1] = seed_audio[1:]
        seed_audio[-1] = ampl_val_16
        pc_str = str(round(100 * curr_sample_idx / float(new_audio.shape[0]), 2))
        if curr_sample_idx%(new_audio.shape[0]/100)==0:
            print('\tPercent complete: ' + pc_str + '\r')
        curr_sample_idx += 1
    print('\tAudio generated.')
    str_timestamp = str(int(time.time()))
    np.save("C:\\Users\\pasca\\Desktop\\audioProject\\output\\generated\\numpy\\" + str_timestamp, distribution)
    print('\tAudio Saved.')

    return new_audio.astype(np.int16)


class SaveAudioCallback(Callback):
    def __init__(self, ckpt_freq, sr, seed_audio):
        super(SaveAudioCallback, self).__init__()
        self.ckpt_freq = ckpt_freq
        self.sr = sr
        self.seed_audio = seed_audio

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.ckpt_freq == 0:
            ts = str(int(time.time()))
            filepath = os.path.join('output/', 'ckpt_' + ts + '.wav')
            audio = get_audio_from_model(self.model, self.sr, 0.5, self.seed_audio)
            write(filepath, self.sr, audio)
