import librosa
import random
import os
import numpy as np
import fnmatch
import tqdm
import tensorflow as tf

"""
This file contains function necessary for working with audio data and input and outputting audio from Wavenet.
"""

LJ_DIRECTORY = "C:\\Users\\pasca\\Desktop\\audioProject\\data\\ljdataset"


# Gets all names of files within a directory
def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def load_train_valid_filenames(directory, num_samples=None, percent_training=0.9):
    randomized_files = find_files(directory)
    random.shuffle(randomized_files)
    if num_samples is not None:
        randomized_files = randomized_files[:num_samples]
    number_of_training_samples = int(round(percent_training * len(randomized_files)))
    training_files, validation_files = randomized_files[:number_of_training_samples], randomized_files[
                                                                                      number_of_training_samples:]
    return training_files, validation_files

# Reads the training/validation audio and concats it into a single array for the NN
def load_generic_audio(training_files, validation_files, sample_rate=16000):
    '''Generator that yields audio waveforms from the directory.'''

    # Concat training data
    training_data = []
    for training_filename in training_files:
        audio, _ = librosa.load(training_filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        training_data = training_data + audio.tolist()

    # Concat validation data
    validation_data = []
    for validation_filename in validation_files:
        audio, _ = librosa.load(validation_filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        validation_data = validation_data + audio.tolist()

    return np.array(training_data), np.array(validation_data)

# Generator
# splits an audio sample up into pieces for the neural network.
def frame_generator(audio, frame_size, frame_shift, minibatch_size=20):
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

# Generator
# splits an audio sample up into pieces for the neural network.
def dataset_generator(audio, frame_size, frame_shift, minibatch_size=20):
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
                yield np.array(X), np.array(y)

def validation_generator(audio, frame_size, frame_shift):
    audio_len = len(audio)
    X = []
    y = []
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
    return (np.asarray(X),np.asarray(y))



# Generates an audio clip from the NN. After each sample is collected, the inverse of the softmax is taken to normalize the sound
def get_audio_from_model(model, sr, duration, seed_audio, frame_size):
    new_audio = np.zeros((sr * duration))
    for curr_sample_idx in tqdm.tqdm(range(new_audio.shape[0])):
        distribution = np.array(model.predict(seed_audio.reshape(1, frame_size, 1)), dtype=float).reshape(256)
        distribution /= distribution.sum().astype(float)
        predicted_val = np.random.choice(range(256), p=distribution)
        ampl_val_8 = predicted_val / 255.0
        ampl_val_16 = (np.sign(ampl_val_8) * (1/255.0) * ((1 + 256.0)**abs(
            ampl_val_8) - 1)) * 2**15
        new_audio[curr_sample_idx] = ampl_val_16
        seed_audio[:-1] = seed_audio[1:]
        seed_audio[-1] = ampl_val_16
    return new_audio.astype(np.int16)

