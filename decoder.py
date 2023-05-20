from DEFs import *
import wave
import numpy as np
import matplotlib.pyplot as plt
from variables import *



#taking the 32-bits and convert it to samples
def convering_from_bits_to_samples(bits,levels_number, peak_level, qtype:Quantizer_types):
    delta = (2*peak_level) / levels_number
    if qtype == Quantizer_types.MID_RISE:
        levels = np.arange(-peak_level + delta/2, peak_level, delta)
    elif qtype == Quantizer_types.MID_TREAD:
        levels = np.arange(-peak_level + delta, peak_level, delta)
    else:
        raise ValueError('Invalid quantization type!')

    print(00)
    decimals = np.zeros(len(bits)//8+1)
    for i in range(0,len(bits),8):
        idx = int(bits[i:i+8],2)
        dec = levels[idx]
        decimals[int(i//8)] = dec

    print(decimals[:20])

    decimals = decimals * (2 ** 15)
    decimals = decimals.astype(np.int16)

    return decimals



def decoder(signal, enc_type, pulse_amp):
    # Convert the bit stream to a sequence of symbols according to the specified encoding

    #signal /= pulse_amp
    bits=''
    if enc_type == Encoder_types.MANCHESTER:
        for i in range(0,len(signal),2):
            if signal[i]==1:
                bits+='1'
            elif signal[i]==-1:
                bits+='0'
            else:
                pass
    


    elif enc_type == Encoder_types.ALTERNATE_MARK_INVERSION:
        for i in range(0,len(signal)):
            if signal[i]==1:
                bits+='1'
            elif signal[i]==-1:
                bits+='1'
            else:
                bits+='0'

    else:
        raise ValueError('Invalid encoding type')
    
    print(bits[:200])
    return bits


def writing_wav_file(signal, sampling_frequency, filename):
    with wave.open(filename, 'w') as wavfile2:


        wavfile2.setnchannels(2)
        wavfile2.setsampwidth(2)
        wavfile2.setframerate(sampling_frequency)
        wavfile2.setnframes(len(signal))

        # Convert the numpy array to bytes and write to the wave file
        wavfile2.writeframes(signal.tobytes())





