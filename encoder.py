from DEFs import *

import wave
import numpy as np
import matplotlib.pyplot as plt



#1 The Sampler function
def sampler(audio_file:str, sampling_frequency:int):
    # Open the audio file
    with wave.open(audio_file, 'rb') as wav_file:
        # Get the sample rate, number of channels, and number of frames
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()

        print('Sample rate: ', sample_rate)
        print('Number of frames: ', num_frames)

        # Read all the frames and convert them to a numpy array
        frames = wav_file.readframes(num_frames)
        audio_data = np.frombuffer(frames, dtype=np.int16)

        # Resample the audio data to the specified sampling frequency
        resampled_data = np.zeros(int(len(audio_data) * sampling_frequency / sample_rate), dtype=np.int16)
        for i in range(len(resampled_data)):
            resampled_data[i] = audio_data[int(i * sample_rate / sampling_frequency)]

        # Create the time vector and amplitude vector
        time_vector = np.arange(0, len(resampled_data)) / sampling_frequency
        amplitude_vector = resampled_data # / (2 ** 15) # use it if you want to normalize the data

        return time_vector, amplitude_vector
    

#2 The Quantizer function
def quantizer(time, amplitude, levels_number, peak_level, qtype:Quantizers):
    
    # Determine the step size
    delta = (2*peak_level) / levels_number
    
    # Define the quantization levels
    if qtype == 'mid-rise':
        levels = np.arange(-peak_level + delta/2, peak_level, delta)
    elif qtype == 'mid-tread':
        levels = np.arange(-peak_level + delta, peak_level, delta)
    else:
        raise ValueError('Invalid quantization type!')
    
    # Quantize the amplitude values
    quantized_amplitude = np.zeros_like(amplitude)
    for i in range(len(amplitude)):
        idx = np.argmin(np.abs(amplitude[i] - levels))
        quantized_amplitude[i] = levels[idx]
    
    plot_from_quantizer(time, amplitude, quantized_amplitude)
    
    return quantized_amplitude

def plot_from_quantizer(time, amplitude, quantized_amplitude):
    # Plot the input and quantized signals
    plt.plot(time, amplitude, label='Input Signal')
    plt.plot(time, quantized_amplitude, drawstyle='steps', label='Quantized Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Quantizer Output')
    plt.legend()
    plt.show()



def encoder():
    pass








###############-----MAIN SCRIPT----#####################

def main():
    # Read the audio file and resample it to 8kHz
    time_vector, amplitude_vector = sampler('audio_warda.wav', 4*1000)
    print(time_vector,'\n\n\n\n', amplitude_vector)


    pass



if __name__ == '__main__':
    main()