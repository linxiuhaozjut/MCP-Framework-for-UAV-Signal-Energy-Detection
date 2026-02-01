from scipy.signal import firwin, lfilter


def design_lowpass_filter(fs, cutoff_freq, num_taps=1024):
    nyquist = fs / 2
    cutoff_norm = cutoff_freq / nyquist
    taps = firwin(num_taps, cutoff_norm, pass_zero=True, window='hann')
    return taps


def filter_iq_signal(iq_data, filter_taps):
    i_filtered = lfilter(filter_taps, 1.0, iq_data.real)
    q_filtered = lfilter(filter_taps, 1.0, iq_data.imag)
    iq_filtered = i_filtered + 1j * q_filtered
    delay = len(filter_taps) // 2
    return iq_filtered[delay:]


def extract_noise_base(iq_filtered, fs, noise_duration=1, block_overlap_ratio=0.1):
    noise_samples = int(fs * noise_duration)
    if noise_samples >= len(iq_filtered):
        raise ValueError("error")

    block_length = noise_samples
    block_overlap = int(block_length * block_overlap_ratio)
    block_step = block_length - block_overlap
    block_starts = np.arange(0, len(iq_filtered) - block_length + 1, block_step)

    blocks = np.array([iq_filtered[start:start + block_length] for start in block_starts])
    block_energies = np.mean(np.abs(blocks) ** 2, axis=1)
    min_energy_idx = np.argmin(block_energies)
    min_energy_start = block_starts[min_energy_idx]

    noise_base = iq_filtered[min_energy_start: min_energy_start + block_length]
    return noise_base
