import numpy as np
import os


def load_iq_data(file_path, dtype=np.int16):
    raw_data = np.fromfile(file_path, dtype=dtype)
    iq_data = raw_data[::2] + 1j * raw_data[1::2]
    return iq_data


def slice_iq_data(iq_data, fs, slice_duration=1, overlap_ratio=0):
    samples_per_slice = int(fs * slice_duration)
    overlap_samples = int(samples_per_slice * overlap_ratio)
    step_samples = samples_per_slice - overlap_samples

    slices = []
    start_idx = 0
    while start_idx + samples_per_slice <= len(iq_data):
        end_idx = start_idx + samples_per_slice
        slices.append(iq_data[start_idx:end_idx])
        start_idx += step_samples

    return slices


def save_iq_slices(iq_slices, original_name, output_dir, dtype=np.int16):
    os.makedirs(output_dir, exist_ok=True)
    for i, slice_data in enumerate(iq_slices):
        i_data = slice_data.real.astype(dtype)
        q_data = slice_data.imag.astype(dtype)
        combined = np.empty(2 * len(slice_data), dtype=dtype)
        combined[::2] = i_data
        combined[1::2] = q_data

        file_name = f"{original_name}-{i}.bin"
        file_path = os.path.join(output_dir, file_name)
        combined.tofile(file_path)
    return len(iq_slices)


def save_single_iq(iq_data, output_path, dtype=np.int16):
    i_data = iq_data.real.astype(dtype)
    q_data = iq_data.imag.astype(dtype)
    combined = np.empty(2 * len(iq_data), dtype=dtype)
    combined[::2] = i_data
    combined[1::2] = q_data
    combined.tofile(output_path)
