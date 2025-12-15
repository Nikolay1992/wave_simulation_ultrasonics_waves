# src/postprocessing.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_results(t_arr, mean_displacement, impulse, meta, config):
    viz = config['visualization']
    output_dir = viz['output_dir']
    save_plots = viz['save_plots']

    if save_plots:
        ensure_dir(output_dir)

    dt = meta['dt']
    Lz = meta['Lz']
    c = meta.get('c', 3600)

    # --- 1. Сигналы во времени ---
    plt.figure(figsize=(9, 4))
    plt.plot(t_arr[1:-1] * 1e6, mean_displacement, label='Выход (z = Lz)', alpha=0.85, linewidth=1.2)
    plt.plot(t_arr * 1e6, impulse, '--', label='Входной импульс', alpha=0.75, linewidth=1.0)
    plt.xlabel('Время, мкс')
    plt.ylabel('Смещение (условные единицы)')
    plt.title('Входной и выходной сигналы')
    plt.grid(True, alpha=0.5)
    plt.legend()
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'time_signals.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # --- 2. Амплитудные спектры (нормированные) ---
    def compute_spectrum(signal, dt):
        N = len(signal)
        freqs = fftfreq(N, dt)
        spec = np.abs(fft(signal))
        return freqs, spec

    freqs_d, spec_d = compute_spectrum(mean_displacement, dt)
    freqs_i, spec_i = compute_spectrum(impulse, dt)

    # Берём только положительные частоты
    mask_pos = freqs_d >= 0
    freqs = freqs_d[mask_pos]          # (N/2+1,)
    spec_d = spec_d[mask_pos]
    spec_i = spec_i[:len(spec_d)]      # обрезаем до длины spec_d

    # Нормировка: максимум = 1
    spec_d_norm = spec_d / np.max(spec_d) if np.max(spec_d) > 0 else spec_d
    spec_i_norm = spec_i / np.max(spec_i) if np.max(spec_i) > 0 else spec_i

    plt.figure(figsize=(8, 4))
    plt.plot(freqs / 1e6, spec_i_norm, label='Вход (нормир.)', linewidth=1.5)
    plt.plot(freqs / 1e6, spec_d_norm, label='Выход (нормир.)', linewidth=1.5)
    plt.xlabel('Частота, МГц')
    plt.ylabel('Нормированная амплитуда')
    plt.title('Амплитудные спектры сигналов')
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.xlim(0, 5.0)  # до 5 МГц (достаточно для f0 = 1 МГц)
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'normalized_spectra.png'), dpi=150, bbox_inches='tight')
    plt.show()