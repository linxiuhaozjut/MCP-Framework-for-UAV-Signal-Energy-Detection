import cv2
from scipy.signal import stft
import numpy as np
import os
from PIL import Image

def iq_to_stft_image(iq_data, fs, win_size=2 ** 17, overlap_ratio=0.5, nfft=2 ** 17):

    overlap_samples = int(win_size * overlap_ratio)
    f, t, Zxx = stft(
        iq_data,
        fs=fs,
        window='hann',
        nperseg=win_size,
        noverlap=overlap_samples,
        nfft=nfft,
        return_onesided=False
    )

    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)
    pos_freq_idx = f >= 0
    f = f[pos_freq_idx]
    Zxx = Zxx[pos_freq_idx, :]

    lowpass_cutoff = 10e6
    valid_freq_idx = f <= lowpass_cutoff
    f = f[valid_freq_idx]
    Zxx = Zxx[valid_freq_idx, :]

    stft_mag = np.abs(Zxx)
    stft_db = 20 * np.log10(stft_mag + 1e-10)

    n_smooth = 16
    n_frames_new = stft_mag.shape[1] // n_smooth
    remainder = stft_mag.shape[1] % n_smooth
    stft_smoothed = stft_db[:, :n_frames_new * n_smooth].reshape(stft_db.shape[0], n_frames_new, n_smooth).mean(axis=2)
    if remainder > 0:
        remainder_mean = stft_db[:, n_frames_new * n_smooth:].mean(axis=1, keepdims=True)
        stft_smoothed = np.hstack([stft_smoothed, remainder_mean])

    max_db, min_db = 40, -80
    stft_db_clamped = np.clip(stft_smoothed, min_db, max_db)
    stft_norm = ((stft_db_clamped - min_db) / (max_db - min_db) * 255).astype(np.uint8)

    stft_cv = cv2.cvtColor(stft_norm, cv2.COLOR_GRAY2RGB)
    stft_cv = cv2.applyColorMap(stft_cv, cv2.COLORMAP_PLASMA)
    return stft_cv





def slice_stft_image_to_squares(stft_img, output_dir, base_name="stft_img", overlap_ratio=0.1):

    os.makedirs(output_dir, exist_ok=True)


    if isinstance(stft_img, np.ndarray):
        if stft_img.shape[2] == 3:
            # 假设输入是 OpenCV BGR -> 转为 RGB
            stft_img_rgb = cv2.cvtColor(stft_img, cv2.COLOR_BGR2RGB)
        else:
            stft_img_rgb = stft_img
        img = Image.fromarray(stft_img_rgb)
    elif isinstance(stft_img, Image.Image):
        img = stft_img
    else:
        raise TypeError(f"stft_img not surpport: {type(stft_img)}")

    w, h = img.size  # (width, height)
    slice_size = w
    step = int(slice_size * (1 - overlap_ratio))
    if step <= 0:
        step = slice_size

    num_slices = max(1, (h - slice_size + step - 1) // step + 1)

    for i in range(num_slices):
        top = i * step
        bottom = top + slice_size

        if bottom > h:
            bottom = h
            top = h - slice_size
            if top < 0:
                top = 0

        crop_img = img.crop((0, top, w, bottom))

        if (bottom - top) < slice_size:
            crop_img = crop_img.resize((slice_size, slice_size), Image.BILINEAR)

        save_name = f"{base_name}_{i}.jpg"
        save_path = os.path.join(output_dir, save_name)
        crop_img.save(save_path)

    print(f"Cut Done: {num_slices} imgs (overlap={overlap_ratio*100:.0f}%) save to {output_dir}")
