from DEFs import *

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