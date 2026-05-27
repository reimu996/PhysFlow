import numpy as np
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.signal.windows import hann


def butter_bandpass(sig, lowcut, highcut, fs, order=2):
    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, sig)
    return y


def hr_fft(sig, fs, harmonics_removal=True):
    sig_orig = np.reshape(sig, -1)
    try:
        sig = sig_orig.copy()
        sig = sig * signal.windows.hann(sig.shape[0])

        sig_f = np.abs(fft(sig))

        low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
        high_idx = np.round(4 / fs * sig.shape[0]).astype('int')

        sig_f[:low_idx] = 0
        sig_f[high_idx:] = 0

        peak_idx, _ = signal.find_peaks(sig_f)
        sort_idx = np.argsort(sig_f[peak_idx])[::-1]

        peak_idx1 = peak_idx[sort_idx[0]]
        f_hr1 = peak_idx1 / sig.shape[0] * fs
        hr1 = f_hr1 * 60

        if len(sort_idx) > 1:
            peak_idx2 = peak_idx[sort_idx[1]]
            f_hr2 = peak_idx2 / sig.shape[0] * fs
            hr2 = f_hr2 * 60

            if harmonics_removal:
                if np.abs(hr1 - 2 * hr2) < 5 and sig_f[peak_idx2] > 0.9 * sig_f[peak_idx1]:
                    hr = hr2
                else:
                    hr = hr1
            else:
                hr = hr1
        else:
            hr = hr1

        return hr

    except Exception:
        return calculate_hr_fft(sig_orig, fs)


def calculate_hr_fft(sig, fs):
    sig = sig * hann(len(sig))
    sig_f = np.fft.fft(sig)
    freqs = np.fft.fftfreq(len(sig), 1 / fs)

    sig_f = sig_f[:len(sig) // 2]
    freqs = freqs[:len(sig) // 2]

    mag = np.abs(sig_f)

    valid_idx = np.where((freqs >= 0.6) & (freqs <= 4.0))[0]
    mag_valid = mag[valid_idx]
    freqs_valid = freqs[valid_idx]

    peak_idx = np.argmax(mag_valid)
    peak_freq = freqs_valid[peak_idx]

    hr = peak_freq * 60
    return hr
