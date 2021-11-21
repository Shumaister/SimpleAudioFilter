import numpy as np
from scipy.signal import butter, lfilter, freqz
from scipy.io import wavfile
import scipy
import scipy.fftpack
import matplotlib.pyplot as plt

#General Definitions
wav_fname = "song-estrellas.wav"

# Filter requirements.
order = 6
#fs = 44100.0        # sample rate, Hz
cutoff = 500.0       # desired cutoff frequency of the filter, Hz

cutoff_high = 5000.0 

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# TODO AGREGAR --- 
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
# TODO AGREGAR --- 

#read the wav file
samplerate, data = wavfile.read(wav_fname)
fs = samplerate
print(f"number of channels = {data.shape[1]}")
samples = data.shape[0]                         # nro de muestras que hay
length = samples / samplerate
print(f"length = {length}s")
time = np.linspace(0., length, data.shape[0])

#Drawing the audio file --- el del entrada
plt.subplot(3, 2, 1)
plt.plot(time, data[:, 0], label="Left channel", color='blue')
#plt.plot(time, data[:, 1], label="Right channel", color='orange')
plt.legend()
plt.ylim(-30000, +30000)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()

#input audio FFT (Spectrum) -- Tranfformada de fourier
left_ch_fft = scipy.fftpack.fft(data[:, 0])
left_ch_fft_frec = scipy.fftpack.fftfreq(len(left_ch_fft), 1.0 / fs)

plt.subplot(3, 2, 2)
plt.xscale('log')
#x_scale = np.linspace(0.0 , samplerate / samples, int(samples /2))
signal_psd = np.abs(left_ch_fft[:samples//2]) ** 2   # se divide por dos porqiue es simetrica respeto a Y
plt.plot(left_ch_fft_frec[:samples//2], 10 * np.log10(signal_psd), label="Left channel FFT", color='red') #al hacerle log 10 lo pasas a db
plt.legend()
plt.xlim(10, 0.5*fs)
plt.xlabel("Frecuency [Hz]")
plt.ylabel("Power spectral density [dB]")
plt.grid(which='both')


# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)
print(a) # polinomio del numerador - normalizado - CEROS
print(b) # polinomio del denominador - POLOS

# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.subplot(3, 2, 4)
plt.xscale('log')
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(10, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid(which='both')

# Filter the data, and plot both the original and filtered signals.
outch1 = butter_lowpass_filter(data[:,0], cutoff, fs, order)
outch2 = butter_lowpass_filter(data[:,1], cutoff, fs, order)

plt.subplot(3, 2, 5)
plt.plot(time, outch1, label='Filtered Left channel', color='blue')
#plt.plot(time, outch2, label='Filtered Right channel', color='orange')
plt.ylim(-30000, +30000)
plt.ylabel("Amplitude")
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

#output audio FFT (Spectrum) ACA SE DIBUJA UN SOLO CANAL en este caso el izquierzo
left_ch_o_fft = scipy.fftpack.fft(outch1)
left_ch_o_fft_frec = scipy.fftpack.fftfreq(len(left_ch_o_fft), 1.0 / fs)
plt.subplot(3, 2, 6)
plt.xscale('log')
signal_o_psd = np.abs(left_ch_o_fft[:samples//2]) ** 2
#x_scale = np.linspace(0.0 , samples / 1.0, int(samples /2))
plt.plot(left_ch_o_fft_frec[:samples//2], 10 * np.log10(signal_o_psd), label="Left channel Filtered FFT", color='red')
#plt.plot(x_scale, 2.0 / samples * np.abs(left_ch_fft[:samples//2]) ** 2, label="Left channel Filtered FFT", color='red')
plt.legend()
plt.xlim(10, 0.5*fs)
plt.xlabel("Frecuency [Hz]")
plt.ylabel("Power Spectral Density")
plt.grid(which='both')


#write the wav file, asuming signed int 16 
outdata = data
outdata[:,0] = outch1
outdata[:,1] = outch2
wav_fname = "out_filtered.wav"
wavfile.write(wav_fname, int(fs), outdata.astype(np.int16))

#Draws the plot
plt.subplots_adjust(hspace=0.35)
plt.show()
