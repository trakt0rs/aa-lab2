import numpy as np
from PIL import Image

def logarithmic_correction(img: Image.Image) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    c = 255.0 / np.log(1 + 255.0)
    arr = c * np.log(1 + arr)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

def compute_histogram(channel: np.ndarray) -> list:
    hist = [0] * 256
    for pixel in channel.flatten():
        hist[int(pixel)] += 1
    return hist

def compute_cdf(hist: list) -> list:
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]
    return cdf

def histogram_equalization(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    result = np.zeros_like(arr)
    total_pixels = arr.shape[0] * arr.shape[1]

    for ch in range(3):
        channel = arr[:, :, ch]

        hist = compute_histogram(channel)
        cdf = compute_cdf(hist)

        cdf_min = 0
        for val in cdf:
            if val > 0:
                cdf_min = val
                break

        lut = [0] * 256
        for i in range(256):
            lut[i] = round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255)

        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                result[y, x, ch] = lut[channel[y, x]]

    return Image.fromarray(result.astype(np.uint8))

if __name__ == "__main__":
    img_low_contrast = Image.open("images/low_contrast.jpeg")
    img_overexposed  = Image.open("images/overexposed.jpeg")
    img_underexposed = Image.open("images/underexposed.jpeg")


    result_low_contrast_log = logarithmic_correction(img_low_contrast)
    result_overexposed_log  = logarithmic_correction(img_overexposed)
    result_underexposed_log = logarithmic_correction(img_underexposed)

    result_low_contrast_log.save("result/corrected_low_contrast_log.jpeg")
    result_overexposed_log.save("result/corrected_overexposed_log.jpeg")
    result_underexposed_log.save("result/corrected_underexposed_log.jpeg")


    result_low_contrast_heq = histogram_equalization(img_low_contrast)
    result_overexposed_heq  = histogram_equalization(img_overexposed)
    result_underexposed_heq = histogram_equalization(img_underexposed)

    result_low_contrast_heq.save("result/corrected_low_contrast_heq.jpeg")
    result_overexposed_heq.save("result/corrected_overexposed_heq.jpeg")
    result_underexposed_heq.save("result/corrected_underexposed_heq.jpeg")
