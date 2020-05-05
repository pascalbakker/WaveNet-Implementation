from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from AudioWaveGen import load_generic_audio, frame_generator, get_audio_from_model, load_train_valid_filenames, frame_generator2
import time
from scipy.io.wavfile import write
import tqdm
import tensorflow as tf
import os
import pickle
from pathlib import Path
#from oldcode.old2.wavenet_model import WaveNet
from wavenet_model import WaveNet
# PATHS
#LJ_DIRECTORY = "C:\\Users\\pasca\\Desktop\\audioProject\\data\\ljdata"  # Dataset Directory
LJ_DIRECTORY = Path("./data/ljdata/wavs/")  # Dataset Directory
import numpy as np
#GENERATED_AUDIO_OUTPUT_DIRECTORY = 'C:\\Users\\pasca\Desktop\\audioProject\\output\\generated\\'
GENERATED_AUDIO_OUTPUT_DIRECTORY = Path('./output/generated/')

# MODEL_OUTPUT_DIRECTORY = 'C:\\Users\\pasca\Desktop\\audioProject\\output\\model\\'
MODEL_OUTPUT_DIRECTORY = Path('./output/model/')
CHECKPOINTDIRECTORY = "C:\\Users\\pasca\\PycharmProjects\\WaveGen\\checkpoint\\"
LOG_DIRECTORY = Path("./model_logs/")


def generateAudioFromModel(model, model_id, sr=16000, frame_size=256, num_files=1, generated_seconds=1,
                           validation_audio=None):
    audio_context = validation_audio[:frame_size]

    for i in range(num_files):
        new_audio = get_audio_from_model(model, sr, generated_seconds, audio_context, frame_size)
        audio_context = validation_audio[i:i + frame_size]
        outfilepath = "C:\\Users\\pasca\\PycharmProjects\\WaveGen\\output\\generated\\"+ (model_id + "_sample_" + str(i) + '.wav')
        print("Saving File")
        write(outfilepath, sr, new_audio)


def trainModel():
    # Initialize Variables
    hyperparameters = {"frame_size": 256,
                       "frame_shift": 128,
                       "sample_rate": 22050,
                       "batch_size": 128,
                       "epochs": 100,
                       "num_filters": 64,
                       "filter_size": 2,
                       "dilation_rate": 2,
                       "num_layers": 40}



    # Get Audio
    print("Retrieving Audio")
    training_files, validation_files = load_train_valid_filenames(LJ_DIRECTORY, num_samples=1,
                                                                  percent_training=0.9)
    validation_files = training_files
    print("Training files",len(training_files))
    print("Validation files",len(validation_files))
    print("Concatting Audio")
    training_audio, validation_audio = load_generic_audio(training_files, validation_files, sample_rate=hyperparameters["sample_rate"])
    training_audio_length = len(training_audio)
    validation_audio_length = len(validation_audio)
    print("Audio Retrieved")
    print("Training Audio Length:", training_audio_length)
    print("Valdiation Audio Length:", validation_audio_length)

    # Create audio generators for model
    validation_data_gen = frame_generator2(validation_audio,hyperparameters["frame_size"], hyperparameters["frame_shift"])
    training_data_gen = frame_generator(training_audio, hyperparameters["frame_size"], hyperparameters["frame_shift"], minibatch_size=hyperparameters["batch_size"])



    # CALLBACKS
    model_id = str(int(time.time()))
    print("Model ID", model_id)
    Path.mkdir(LOG_DIRECTORY / model_id)
    log_dir = LOG_DIRECTORY / model_id
    checkpoint_filepath = MODEL_OUTPUT_DIRECTORY / model_id / "checkpoint.ckpt"

    tensorboard_callback = TensorBoard(log_dir=log_dir,
                                       histogram_freq=0)
    earlystopping_callback = EarlyStopping(monitor='val_accuracy',
                                           min_delta=0.01,
                                           patience=10,
                                           verbose=0,
                                           restore_best_weights=True)
    outputFolder = './output/model'
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    filepath = outputFolder + "/model"+model_id+"-{epoch:02d}-{val_accuracy:.2f}.hdf5"

    checkpoint_callback = ModelCheckpoint(
        filepath, monitor='val_accuracy', verbose=1,
        save_best_only=False, save_weights_only=False,
        save_frequency=1)


    print("Writing mandatory Hyperparameter logs to {}\n".format(log_dir / "hyperparameters.pkl"))
    # Write hyper parameters to log file
    hyperparamter_filename = log_dir / "hyperparameters.pkl"
    Path.touch(hyperparamter_filename)
    with open(hyperparamter_filename, 'wb') as f:
        pickle.dump(hyperparameters, f, pickle.HIGHEST_PROTOCOL)

    # Write validation file names to file
    validation_filename = log_dir / "validation_files.pkl"
    #Path.touch(validation_filename)
    with open(validation_filename, 'wb') as fp:
        pickle.dump(validation_files, fp)

    # Write training file names to file
    training_filename = log_dir / "training_files.pkl"
    print(training_files)
    with open(training_filename, 'wb') as fp:
        pickle.dump(training_files, fp)

    print("Starting Model Training...\n")
    sub = WaveNet(num_filters=hyperparameters["num_filters"],
                  filter_size=hyperparameters["filter_size"],
                  dilation_rate=hyperparameters["dilation_rate"],
                  num_layers=hyperparameters["num_layers"],
                  input_size=hyperparameters["frame_size"])
    model = sub.model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])


    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    model.fit(training_data_gen,
              epochs=hyperparameters["epochs"],
              steps_per_epoch=training_audio_length // hyperparameters["batch_size"],
              validation_data=validation_data_gen,
              verbose=1,
              callbacks=[tensorboard_callback, earlystopping_callback, checkpoint_callback])
    print('Saving model...')
    model.save(MODEL_OUTPUT_DIRECTORY / ('model_' + model_id + '_' + '.h5'))
    print("Model saved.")

    print("Generating Audio.")
    generateAudioFromModel(model, model_id, sr=hyperparameters["sample_rate"], frame_size=hyperparameters["frame_size"],
                           num_files=1, generated_seconds=3, validation_audio=validation_audio)
    print("Program Complete.")
    return model_id


# trains from checkpint. TODO
def train_from_checkpoint(model_id):
    checkpoint_filepath = './checkpoint/1588553698/1588553698_checkpoint.cpkt'
    hyperparamter_filepath = LOG_DIRECTORY / str(model_id) / "hyperparameters.pkl"
    training_filepath = LOG_DIRECTORY / model_id / "training_files.pkl"
    validation_filepath = LOG_DIRECTORY / model_id / "validation_files.pkl"

    print("Loading Files")
    with open(training_filepath, 'rb') as fp:
        training_filenames = pickle.load(fp)

    with open(validation_filepath, 'rb') as fp:
        valdiation_filenames = pickle.load(fp)

    with open(hyperparamter_filepath, 'rb') as f:
        hyperparameters = pickle.load(f)

    print("Loading Audio")
    # Load audio data:
    training_audio, validation_audio = load_generic_audio(training_filenames, valdiation_filenames,
                                                          sample_rate=hyperparameters["sample_rate"])
    validation_data_gen = frame_generator2(validation_audio, hyperparameters["frame_size"], hyperparameters["frame_shift"])
    training_data_gen = frame_generator(training_audio, hyperparameters["frame_size"], hyperparameters["frame_shift"], minibatch_size=hyperparameters["batch_size"])

    # Create Callbacks
    log_dir = LOG_DIRECTORY / model_id
    tensorboard_callback = TensorBoard(log_dir= log_dir, histogram_freq=1)
    # Early Stopping
    earlystopping_callback = EarlyStopping(
        monitor='val_accuracy', min_delta=0.005, patience=10, verbose=0, restore_best_weights=True
    )
    # Save model after every epoch
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        CHECKPOINTDIRECTORY + "\\" + model_id + "\\" + model_id + "_checkpoint.cpkt",
        save_weights_only=True,
        verbose=1)

    sub = WaveNet(num_filters=hyperparameters["num_filters"], filter_size=hyperparameters["filter_size"],
                  dilation_rate=hyperparameters["dilation_rate"], num_layers=hyperparameters["num_layers"],
                  input_size=hyperparameters["frame_size"])
    model = sub.model()
    print("Loading weights")
    #fullpath = '.\\checkpoint\\1588553698\\1588553698_checkpoint.cpkt'
    model.load_weights(checkpoint_filepath)
    print("Starting Training")
    model.fit(training_data_gen, epochs=100, steps_per_epoch=len(training_audio) / hyperparameters["batch_size"],
              validation_data=validation_data_gen,
              validation_steps=len(validation_audio) / hyperparameters["batch_size"], verbose=2,
              callbacks=[tensorboard_callback, cp_callback, earlystopping_callback])

    print("Generating Audio.")
    generateAudioFromModel(model, model_id, sr=hyperparameters["sample_rate"], frame_size=hyperparameters["frame_size"],
                           num_files=1, generated_seconds=3, validation_audio=validation_audio)
    print("Program Complete.")
    return model_id


# Given a model, generates samples. TODO
def generateData(model_id):
    hyperparamter_filepath = LOG_DIRECTORY / str(model_id) / "hyperparameters.pkl"
    validation_filepath = LOG_DIRECTORY / str(model_id) / "validation_files.pkl"
    with open(validation_filepath, 'rb') as fp:
        valdiation_filenames = pickle.load(fp)
    print(valdiation_filenames)

    with open(hyperparamter_filepath, 'rb') as f:
        hyperparameters = pickle.load(f)

    # Load audio data:
    training_audio, validation_audio = load_generic_audio([], valdiation_filenames,
                                                          sample_rate=hyperparameters["sample_rate"])
    sub = WaveNet(num_filters=hyperparameters["num_filters"], filter_size=hyperparameters["filter_size"],
                  dilation_rate=hyperparameters["dilation_rate"], num_layers=hyperparameters["num_layers"],
                  input_size=hyperparameters["frame_size"])
    model = sub.model()
    model.load_weights('C:\\Users\\pasca\\PycharmProjects\\WaveGen\\output\model\\model1588632404-26-1.00.hdf5')
    generateAudioFromModel(model, model_id, sr=hyperparameters["sample_rate"], frame_size=hyperparameters["frame_size"],
                           num_files=1, generated_seconds=3, validation_audio=validation_audio)

    return 0


if __name__ == '__main__':
    model_id = trainModel()
    #model_id = train_from_checkpoint('1588553698')
    #generateData("1588632404")