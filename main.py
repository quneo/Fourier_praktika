import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from PIL import Image


def load_image(file_name: str) -> np.array:
    with Image.open(file_name) as img:
        img = img.convert('L')
        img.load()
    return np.array(img)


def compute_pq(m, n):
    p = 2 * m - 1
    q = 2 * n - 1
    if p % 2 != 0:
        p += 1
    if q % 2 != 0:
        q += 1
    return p, q


def pad_image(image: np.array, p: int, q: int) -> np.array:
    m, n = image.shape
    pad_top = (p - m) // 2
    pad_bottom = p - m - pad_top
    pad_left = (q - n) // 2
    pad_right = q - n - pad_left
    padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    return padded


def apply_fourier_transform(image_np: np.array) -> np.array:
    f_transform = fft2(image_np)
    f_shifted = fftshift(f_transform)
    return f_shifted  # Возвращаем комплексный спектр


def hi_lo_filter(fft_shifted: np.array, low_bound: int, high_bound: int) -> tuple[np.array, np.array]:
    rows, cols = fft_shifted.shape
    crow, ccol = rows // 2, cols // 2

    Y, X = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    band_pass_mask = (dist_from_center >= low_bound) & (dist_from_center <= high_bound)

    filtered_shifted = fft_shifted * band_pass_mask

    f_ishift = ifftshift(filtered_shifted)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)

    filtered_magnitude = np.abs(filtered_shifted)
    return filtered_magnitude, img_back


def main():
    image = load_image("test3.png")
    m, n = image.shape
    p, q = compute_pq(m, n)

    padded_image = pad_image(image, p, q)
    fft_shifted = apply_fourier_transform(padded_image)

    # Полный спектр для исходного изображения
    original_spectrum = np.abs(fft_shifted)

    # Применяем полосовой фильтр
    filtered_spectrum, restored_image = hi_lo_filter(fft_shifted, low_bound=1, high_bound=80)

    # Обрезаем восстановленное изображение до оригинального размера
    pad_top = (p - m) // 2
    pad_left = (q - n) // 2
    restored_cropped = restored_image[pad_top:pad_top + m, pad_left:pad_left + n]

    # Также обрезаем спектр (для сравнения)
    filtered_spectrum_cropped = filtered_spectrum[pad_top:pad_top + m, pad_left:pad_left + n]

    # Визуализация
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Original Spectrum")
    plt.imshow(np.log1p(original_spectrum), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Restored Image")
    plt.imshow(restored_cropped, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Filtered Spectrum")
    plt.imshow(np.log1p(filtered_spectrum_cropped), cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
