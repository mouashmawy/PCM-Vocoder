from DEFs import *
import wave
import numpy as np
import matplotlib.pyplot as plt
from variables import *
from binary_converting import *
import textwrap

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
        amplitude_vector = resampled_data  / (2 ** 15) # use it if you want to normalize the data


        # with wave.open('output3.wav', 'w') as wavfile2:

        #     x1 = wav_file.getnchannels()
        #     x2 = wav_file.getsampwidth()
        #     x3 = wav_file.getframerate()
        #     x4 = wav_file.getnframes()


        #     print('x1: ', x1, 'x2: ', x2, 'x3: ', x3, 'x4: ', x4)
        #     wavfile2.setnchannels(x1)
        #     wavfile2.setsampwidth(x2)
        #     wavfile2.setframerate(sampling_frequency)
        #     wavfile2.setnframes(int(x4*sampling_frequency/sample_rate)+1)

        #     # Convert the numpy array to bytes and write to the wave file
        #     wavfile2.writeframes(amplitude_vector.tobytes())




        return time_vector, amplitude_vector
    



#2 The Quantizer function
def quantizer(time, amplitude, levels_number, peak_level, qtype:Quantizer_types):
    
    # Determine the step size
    delta = (2*peak_level) / levels_number
    
    # Define the quantization levels
    if qtype == Quantizer_types.MID_RISE:
        levels = np.arange(-peak_level + delta/2, peak_level, delta)
    elif qtype == Quantizer_types.MID_TREAD:
        levels = np.arange(-peak_level + delta, peak_level, delta)
    else:
        raise ValueError('Invalid quantization type!')
    
    # Quantize the amplitude values
    quantized_amplitude = np.zeros_like(amplitude)
    bits = ''
    mse = 0
    for i in range(len(amplitude)):
        idx = np.argmin(np.abs(amplitude[i] - levels))
        quantized_amplitude[i] = levels[idx]
        bits += f'{idx:08b}'  # convert the quantization index to a 4-bit binary string
        #print(f'{idx:32b}')
        mse += (amplitude[i] - levels[idx])**2
        if i<10:
            print("qqqqqqqqqqqqqqqqqq",idx)
    mse /= len(amplitude)
    
    print(f'Stream of bits: {bits[:200]}')
    print(f'Mean square quantization error: {mse:.4f}')
    #plot_quantizer(time, amplitude, quantized_amplitude)
    

    return bits


#taking the 32-bits and convert it to samples
def convering_from_bits_to_samples(bits):
    print(00)
    BitsSamples = textwrap.wrap(bits, 8)
    print(8)
    print('bits samples', BitsSamples[:10])
    sampledSignal = [int(i, 2) for i in np.array(BitsSamples)]
    print('sampled', sampledSignal[:10], 'len is ', len(sampledSignal))
    return sampledSignal




def plot_quantizer(time, amplitude, quantized_amplitude):
    # Plot the input and quantized signals
    plt.plot(time, amplitude, label='Input Signal')
    plt.plot(time, quantized_amplitude, drawstyle='steps', label='Quantized Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Quantizer Output')
    plt.legend()
    plt.show()



#3 The encoder function
def encoder(bits, pulse_amp, bit_dur, enc_type, bits_plotted=20):
    # Convert the bit stream to a sequence of symbols according to the specified encoding
    if enc_type == Encoder_types.MANCHESTER:
        signal = []
        for b in bits:
            if b == '1':
                signal += [1, -1]
            else:
                signal += [-1, 1]
        signal = np.array(signal) 
        print(len(signal))

    elif enc_type == Encoder_types.ALTERNATE_MARK_INVERSION:
        flag = 0 
        signal = np.zeros(len(bits))
        for ii in range(0,len(bits)):
            if (bits[ii]=='1') and (flag==0):
                signal[ii] = 1
                flag=1;
            elif (bits[ii]=='1') and (flag==1):
                signal[ii]= -1
                flag = 0;
            elif (bits[ii]=='0'):
                pass

        signal *= pulse_amp
        print(type(signal))

    else:
        raise ValueError('Invalid encoding type')

      
    plot_encoder(signal, enc_type, pulse_amp, bit_dur, bits_plotted)

    return signal




def plot_encoder(signal, enc_type, pulse_amp, bit_dur, bits_plotted=20):
    
    signal = np.repeat(signal, int(bit_dur)) 
    signal = signal[:bits_plotted*int(bit_dur)]

    t = np.arange(len(signal) * bit_dur, step=bit_dur)
    plt.plot(t, signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'{enc_type} encoding, Pulse amplitude: {pulse_amp}, Bit duration: {bit_dur}')
    plt.show()


def save_to_file(signal, filename):
    with open(filename, 'w') as file:
        for s in signal:
            file.write(f'{s}')
    print(f'Signal saved to {filename}')


###############-----MAIN SCRIPT----#####################

def main():

    # All the variables are defined in variables.py file

    time_vector, amplitude_vector = sampler(audio_name, sampling_frequency)
    bits = quantizer(time_vector, amplitude_vector, levels_number, peak_level, quantizer_type)
    x = convering_from_bits_to_samples(bits)
    print('x ', x[:10])
    #encoded_signal = encoder(bits, pulse_amp, bit_dur, encoder_type, bits_to_plot)
    #save_to_file(encoded_signal, 'encoded_signal.txt')

    



if __name__ == '__main__':
    main()