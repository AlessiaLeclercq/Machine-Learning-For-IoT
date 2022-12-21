import os
import uuid
import redis
import psutil
import numpy as np
import argparse as ap
import tensorflow as tf 
import sounddevice as sd
import tensorflow_io as tfio

from zipfile import ZipFile
from time import time, sleep


def check_ts_creation(redis_client, name, retention=0, uncompressed=False):
    try:
        redis_client.ts().create(name, retention_msec = retention, uncompressed = uncompressed)
    except redis.ResponseError:
        pass
    return

def get_audio_from_numpy(indata):
    '''transforms a numpy array into an audio tensor'''
    indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    indata = 2*((indata + 32768) / (32767 + 32768)) -1
    indata = tf.squeeze(indata) 
    return indata


def get_spectrogram(audio, frame_length, frame_step):
    '''computes the spectrogram of an audio tensor'''
    zero_padding = tf.zeros(16000 - tf.shape(audio), dtype=tf.float32)
    audio_padded = tf.concat([audio, zero_padding], axis=0)
    stft = tf.signal.stft(
        audio_padded,
        frame_length=frame_length, #length of convolutional window
        frame_step=frame_step, #length of the step the convolutional window makes
        fft_length=frame_length #equal to frame length
    )
    spectrogram = tf.abs(stft)   
    return spectrogram


def is_silence(indata, frame_length, frame_step, dbFSthresh, duration_time):
    '''checks whether the audio is silent or not, hence whether its energy is below a certain threshold'''
    audio = get_audio_from_numpy(indata) 
    spectrogram = get_spectrogram(audio, frame_length, frame_step)

    dbFS = 20* tf.math.log(spectrogram + 1.e-6) 
    energy = tf.math.reduce_mean(dbFS, axis = 1) 
    non_silence = energy > dbFSthresh
    non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
    non_silence_duration = (non_silence_frames + 1)* frame_length

    if non_silence_duration > duration_time:
        return False, spectrogram
    else:
        return True, spectrogram


def  get_mfccs_from_spectrogram(spectrogram, linear_to_mel_weight_matrix):

    mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

    mfccs = tf.expand_dims(mfccs, 0)  # batch axis
    mfccs = tf.expand_dims(mfccs, -1)  # channel axis
    mfccs = tf.image.resize(mfccs, [32,32])

    return mfccs

#******************************************************************************************************************************************
#******************************************************************************************************************************************

ZIP_PATH = os.path.join(os.getcwd(), 'model10.tflite.zip')
MODEL_PATH = os.path.join(os.getcwd(), 'model10.tflite') 
blocksize = samplerate = 16000
channels = 1
resolution = "int16"
collection = False

PREPROCESSING_ARGS = {
'downsampling_rate': 16000,
'frame_length_in_s': .016,
'frame_step_in_s': .016,
'num_mel_bins': 20,
'lower_frequency': 20,
'upper_frequency': 8000,
'duration_time': .04,
'dbFSthresh': -120,
}

#GET WEIGHT MATRIX
frame_length = int(PREPROCESSING_ARGS['downsampling_rate'] * PREPROCESSING_ARGS['frame_length_in_s'])
frame_step = int(PREPROCESSING_ARGS['downsampling_rate'] * PREPROCESSING_ARGS['frame_step_in_s'])
num_spectrogram_bins = frame_length // 2 + 1

linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=PREPROCESSING_ARGS['num_mel_bins'],
    num_spectrogram_bins=num_spectrogram_bins,
    sample_rate=PREPROCESSING_ARGS['downsampling_rate'],
    lower_edge_hertz=PREPROCESSING_ARGS['lower_frequency'],
    upper_edge_hertz=PREPROCESSING_ARGS['upper_frequency']
)


#MODEL 
with ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(os.getcwd())

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


#ARGUMENT PARSING
parser = ap.ArgumentParser()
parser.add_argument("--device", type = int)
parser.add_argument("--host", type = str)
parser.add_argument("--port", type = int)
parser.add_argument("--user", type = str)
parser.add_argument("--password", type = str)
args = parser.parse_args()

device = args.device
REDIS_HOST = args.host
REDIS_PORT = args.port
REDIS_USER = args.user
REDIS_PASSWORD = args.password


#CREATE REDIS CONNECTION
redis_client = redis.Redis(host = REDIS_HOST, port = REDIS_PORT, password = REDIS_PASSWORD, username= REDIS_USER)
print(f"Is active: {redis_client.ping()}")

#CREATE SERIES
MAC_ADDRESS = hex(uuid.getnode())
series1 = f"{MAC_ADDRESS}:battery"
series2 = f"{MAC_ADDRESS}:power"


check_ts_creation(redis_client = redis_client, name = series1) 
check_ts_creation(redis_client = redis_client, name = series2) 


def callback(indata, frames, callback_time, status):
    '''call back function for sd.InputStream which stores the audio if not silent'''
    global PREPROCESSING_ARGS, frame_length, frame_step, interpreter, input_details, output_details, linear_to_mel_weight_matrix, collection
    #check for silence
    silent, spectrogram = is_silence(indata, frame_length, frame_step, PREPROCESSING_ARGS['dbFSthresh'], PREPROCESSING_ARGS['duration_time'])

    #if it is not silent -> test the model
    if not silent:
        mfccs = get_mfccs_from_spectrogram(spectrogram, linear_to_mel_weight_matrix)
        #test the audio
        interpreter.set_tensor(input_details[0]['index'], mfccs)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        max_probability = max(output[0])
        top_index = np.argmax(output[0])

        if max_probability> 0.95 and top_index == 0: #go - start collecting data 
            print("GO predicted: start data collection")
            collection = True 

        if max_probability> 0.95 and top_index == 1: #stop - stop collecting data
            print("STOP predicted: stop data collection")
            collection = False
        
        #if max probability < 0.95 don't do anything 
    #if silent don't do anything
        
    #to store the data 
    if collection:
        new_timestamp = int(time()*1000)
        battery_level = psutil.sensors_battery().percent
        power_plugged = int(psutil.sensors_battery().power_plugged)
        redis_client.ts().add(series1, new_timestamp, battery_level)
        redis_client.ts().add(series2, new_timestamp, power_plugged)    




with sd.InputStream(device = device, channels = channels , samplerate= samplerate, dtype = resolution, callback=callback, blocksize= blocksize):
    # Keeps recording until a ctrl+c keyboard interrupt occurs
    while True:
        continue


         
         