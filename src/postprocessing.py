# src/postprocessing.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.stats import linregress
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
    c = meta.get('c', 3600)  # скорость в матрице (не c_pore!)

    # --- 1. Сигналы во времени ---
    t_shift = Lz / c  # задержка из-за конечной скорости
    t_output_shifted = t_arr - t_shift  # сдвигаем временнУю ось выхода

    plt.figure(figsize=(9, 4))
    plt.plot(t_output_shifted * 1e6, mean_displacement, label='Выход (z = Lz), скорректировано', alpha=0.85, linewidth=1.2)
    plt.plot(t_arr * 1e6, impulse, '--', label='Входной импульс', alpha=0.75, linewidth=1.0)
    plt.xlabel('Время, мкс')
    plt.xlim(-5, 10)
    plt.ylabel('Смещение (условные единицы)')
    plt.title('Входной и выходной сигналы (с компенсацией задержки)')
    plt.grid(True, alpha=0.5)
    plt.legend()
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'time_signals_aligned.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # --- 2. Амплитудные спектры (без нормировки!) ---
    def compute_spectrum(signal, dt):
        N = len(signal)
        freqs = fftfreq(N, dt)
        spec = np.abs(fft(signal))
        return freqs, spec

    freqs_d, spec_d = compute_spectrum(mean_displacement, dt)
    freqs_i, spec_i = compute_spectrum(impulse, dt)

    # Только положительные частоты
    mask_pos = freqs_d >= 0
    freqs = freqs_d[mask_pos]
    spec_d = spec_d[mask_pos]
    spec_i = spec_i[:len(spec_d)]

    # Защита от деления на ноль
    spec_i_safe = np.where(spec_i == 0, 1e-20, spec_i)
    ln_ratio = np.log(spec_d / spec_i_safe)

    # --- 3. ln(A_out / A_in) + оценка Q ---
    plt.figure(figsize=(8, 4))
    plt.plot(freqs / 1e6, ln_ratio, 'o', markersize=3, label=r'$\ln(A_{\text{out}}/A_{\text{in}})$', alpha=0.7)

    # Аппроксимация на [0.8, 1.4] МГц
    f1, f2 = 0.8e6, 1.4e6
    mask_fit = (freqs >= f1) & (freqs <= f2)
    f_fit = freqs[mask_fit]
    ln_fit = ln_ratio[mask_fit]

    Q_est = np.nan
    if len(f_fit) >= 2:
        slope, intercept, r_value, p_value, std_err = linregress(f_fit, ln_fit)
        ln_fit_line = slope * f_fit + intercept

        plt.plot(f_fit / 1e6, ln_fit_line, 'r-', linewidth=2,
                 label=f'Аппроксимация в диапазоне {f1/1e6} - {f2/1e6} МГц')

        # 🔹 Оценка Q (только если slope < 0)
        if slope < 0:
            Q_est = -np.pi * Lz / (c * slope)
            plt.text(f1/1e6, -4,
                     f'$Q_{{\\text{{est}}}} = {Q_est:.1f}$',
                     fontsize=12, bbox=dict(facecolor='lightgreen', alpha=0.8))
            print(f"📈 Оценка добротности по наклону:")
            print(f"    Q_est = -π·L / (c · slope) = -π·{Lz:.4f} / ({c:.0f} · {slope:.3e})")
            print(f"    → Q_est = {Q_est:.2f}")
        else:
            print("⚠️ Наклон ≥ 0 → затухания нет или сильное рассеяние. Q не оценивается.")

        print(f"    R² = {r_value**2:.4f}")
    else:
        print("⚠️ Недостаточно точек для аппроксимации в диапазоне [0.8, 1.4] МГц")

    plt.xlabel('Частота, МГц')
    plt.ylabel(r'$\ln(A_{\text{out}} / A_{\text{in}})$')
    plt.title(r'Логарифмическое отношение спектров с оценкой $Q$')
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.xlim(0, 3.0)
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'ln_ratio_with_Q.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # --- 4. Абсолютные спектры ---
    plt.figure(figsize=(8, 4))
    plt.plot(freqs / 1e6, spec_i, label='|A_in|', linewidth=1.5)
    plt.plot(freqs / 1e6, spec_d, label='|A_out|', linewidth=1.5)
    plt.xlabel('Частота, МГц')
    plt.ylabel('Амплитуда')
    plt.title('Абсолютные спектры')
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.xlim(0, 3.0)
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'absolute_spectra.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ Постобработка завершена.")
    return Q_est  # теперь возвращаем оценку