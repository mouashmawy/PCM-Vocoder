from DEFs import *
from encoder import *
# Define the input variables----------------------------------
#------- Sampler ---------------#
sampling_frequency = 8*1000
audio_name = 'audio_warda.wav'
#------- Quantizer ---------------#
quantizer_type = Quantizer_types.MID_RISE
levels_number = 256
peak_level = 1
#------- Encoder ---------------#
encoder_type = Encoder_types.MANCHESTER
pulse_amp = 1
bit_dur = 10
bits_to_plot = 50



###############-----MAIN SCRIPT----#####################

def main():

    # All the variables are defined in variables.py file

    time_vector, amplitude_vector = sampler(audio_name, sampling_frequency)
    bits = quantizer(time_vector, amplitude_vector, levels_number, peak_level, quantizer_type)
    encoded_signal = encoder(bits, pulse_amp, bit_dur, encoder_type, bits_to_plot)
    #save_to_file(encoded_signal, 'encoded_signal.txt')

    decoded_bits = decoder(encoded_signal, encoder_type, pulse_amp)
    decoded_samples = convering_from_bits_to_samples(decoded_bits, levels_number, peak_level, quantizer_type)
    #plot_quantizer(time_vector, amplitude_vector, decoded_samples)
    writing_wav_file(amplitude_vector, sampling_frequency, 'decoded_audio.wav')
    



if __name__ == '__main__':
    main()