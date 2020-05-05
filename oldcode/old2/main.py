from old2.wavenet_model import WaveNet
from old2.audio_manipulation import *
from tensorflow.python.keras.callbacks import TensorBoard
import datetime


def trainModel():
    # Initialize Paths:
    inputPath = "C:\\Users\\pasca\Desktop\\audioProject\\data\\clips"
    outputPath = "C:\\Users\\pasca\Desktop\\audioProject\\data\\"

        # Initialize Variables
    frame_size = 256
    frame_shift = 128


        # Get Audio
    audio = AudioManipulation(inputPath,outputPath)
    sr_train, training_audio = audio.readMp3(1)
    sr_valid, validation_audio = audio.readMp3(1)
    sr = sr_train
    print("training audio shape",training_audio.shape)
    n_training_examples = int((len(training_audio)-frame_size-1) / float(
            frame_shift))
    n_validation_examples = int((len(validation_audio )-frame_size-1) / float(
            frame_shift))

    # num_filters, filter_size, dilation_rate, num_layers, input_size
    sub = WaveNet(num_filters = 64,filter_size = 2,dilation_rate = 2, num_layers= 40, input_size=frame_size)

    audio_context = validation_audio[:frame_size]
    save_audio_clbk = SaveAudioCallback(500, sr_train, audio_context)

    log_dir = "C:\\Users\\pasca\\Desktop\\audioProject\logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    model = sub.model()
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                          metrics=['accuracy'])

    validation_data_gen = frame_generator(sr, validation_audio, frame_size, frame_shift)
    training_data_gen = frame_generator(sr, training_audio, frame_size, frame_shift)


    STEPS_PER_EPOCH = 1000
    EPOCHS = 20

    model.fit_generator(training_data_gen, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_data=validation_data_gen,validation_steps=STEPS_PER_EPOCH, verbose=1,callbacks=[save_audio_clbk, tensorboard_callback])


    print('Saving model...')
    str_timestamp = str(int(time.time()))
    model.save('C:\\Users\\pasca\Desktop\\audioProject\\output\\model\\model_'+str_timestamp+'_'+str(EPOCHS)+'.h5')

    print('Generating audio...')
    for i in range(1,10):
        str_timestamp = str(int(time.time()))
        new_audio = get_audio_from_model(model, sr_train, 3, audio_context,frame_size)
        audio_context = validation_audio[i*sr_train:i*sr_train+frame_size]
        outfilepath = 'C:\\Users\\pasca\Desktop\\audioProject\\output\\generated\\' + str_timestamp + '.wav'
        print('Writing generated audio to:', outfilepath)
        write(outfilepath, sr_train, new_audio)
        print('\nDone!')

    return


def generateData():
    # Initialize Paths:
    inputPath = "C:\\Users\\pasca\Desktop\\audioProject\\data\\clips"
    outputPath = "C:\\Users\\pasca\Desktop\\audioProject\\data\\"

    # Initialize Variables
    frame_size = 256*2
    frame_shift = 128*2

    # Get Audio
    print("Loading Audio")
    audio = AudioManipulation(inputPath, outputPath)
    sr_train, training_audio = audio.readMp3(1)
    sr_valid, validation_audio = audio.readMp3(1)
    sr = sr_train

    n_training_examples = int((len(training_audio) - frame_size - 1) / float(
        frame_shift))
    n_validation_examples = int((len(validation_audio) - frame_size - 1) / float(
        frame_shift))
    sub = WaveNet(64, 2, 2, 20, input_size=frame_size)

    audio_context = validation_audio[:frame_size]
    save_audio_clbk = SaveAudioCallback(500, sr_train, audio_context)
    print("Audio Loaded")
    print("Loading Model")
    model = sub.model()
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['accuracy'])
    model.load_weights('C:\\Users\\pasca\\Desktop\\audioProject\\output\\model\\model_1587854019_1.h5')
    print('Model Loaded')

    str_timestamp = str(int(time.time()))

    print('Generating audio...')
    for i in range(10):
        str_timestamp = str(int(time.time()))
        new_audio = get_audio_from_model(model, sr_train, 5, audio_context,frame_size)
        audio_context = validation_audio[:frame_size]
        outfilepath = 'C:\\Users\\pasca\Desktop\\audioProject\\output\\generated\\' + str_timestamp + '.wav'
        print('Writing generated audio to:', outfilepath)
        write(outfilepath, sr_train, new_audio)
        print('\nDone!')

    return 1


if __name__=='__main__':
    trainModel()
    #generateData()
