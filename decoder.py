import math

from DEFs import *
import wave
import numpy as np
import matplotlib.pyplot as plt
from variables import *
from bitstring import BitArray
import textwrap
from scipy.signal import butter, filtfilt

###############-----DECODING PART----#####################
def readingPCM(PCMFile):
    with open(PCMFile) as f:
        data = f.read()

    print(type(data))
    array = []
    flag = 0
    i = 0
    for char in data:
        if char == "1":
            if flag == 1:
                flag = 0
                continue
            i += 1
            array.append(1)
        else:
            i += 1
            flag = 1
            array.append(-1)

    # Convert the list of strings to a numpy array of integers
    num_array = np.array(array, dtype=int)
    print('array is ', num_array[:20], 'length is ', len(num_array))
    # Print the resu
    return num_array



def ManchesterSignalingIntoBites():
    bits = ''
    array = []
    arrays = readingPCM("encoded_signal2.txt")
    timar = 0
    BitsSamples = []
    for j in range(0, len(arrays), 2):
        if arrays[j] == 1 and arrays[j + 1] == -1:
            bits += '1'
        elif arrays[j] == -1 and arrays[j + 1] == 1:
            bits += '0'
        if timar < int(math.log2(levels_number)) - 1:
            timar += 1
        else:
            BitsSamples.append(bits)
            bits = ''
            timar = 0

    print('bits samples', BitsSamples[:10])
    sampledSignal = [int(i, 2) for i in np.array(BitsSamples)]
    print('sampled', sampledSignal[:10], 'len is ', len(sampledSignal))
    return sampledSignal


def signal(qtype):
    delta = (2 * peak_level) / levels_number

    # Define the quantization levels
    if qtype == Quantizer_types.MID_RISE:
        levels = np.arange(-peak_level + delta / 2, peak_level, delta)
    elif qtype == Quantizer_types.MID_TREAD:
        levels = np.arange(-peak_level + delta, peak_level, delta)
    else:
        raise ValueError('Invalid quantization type!')
    sampledsignal = ManchesterSignalingIntoBites()
    quantizedsignal = []
    for i in sampledsignal:
        quantizedsignal.append(levels[i] * 2 ** 15)

    print('quantizedsignal', quantizedsignal[:20], 'len is', len(quantizedsignal))
    return quantizedsignal


def demodulateTheSignal(samples):
    # Calculate the time vector
    time = np.arange(len(samples)) / sampling_frequency

    # Construct the signal
    samples = np.array(samples)
    t = np.linspace(0, time[-1], len(samples) * 10)
    signal = np.interp(t, time, samples)
    print('demodelating signal ', signal[:10])
    return signal


def constractingTheVideo(samples):
    framerate = 480000
    nframes = 806488
    nchannels = 1  # Mono audio
    sampwidth = 2  # 16-bit audio

    # Open a new wave file for writing
    with wave.open('output2.wav', 'w') as wavfile:
        wavfile.setnchannels(nchannels)
        wavfile.setsampwidth(sampwidth)
        wavfile.setframerate(framerate)
        wavfile.setnframes(nframes)

        # Convert the numpy array to bytes and write to the wave file
        wavfile.writeframes(samples.tobytes())


def signalploting():
    signal2 = signal(quantizer_type)
    print('signal again ', signal)
    time_vector = np.arange(0, len(signal2)) / sampling_frequency
    plt.plot(time_vector, signal2)
    plt.show()


samples = signal(quantizer_type)
demodulatedSignal = demodulateTheSignal(samples)
constractingTheVideo(demodulatedSignal)
