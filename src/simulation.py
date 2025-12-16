# src/simulation.py
"""
3D-симуляция распространения ультразвуковой волны в пористом цилиндрическом образце
с использованием модели Standard Linear Solid (SLS) для описания вязкоупругости.

📌 Модель SLS: 
   ∂²u/∂t² = c² ∇²u + η ∂/∂t (∇²u),   где η = c² / (ω₀ Q)
→ Учитывает частотно-зависимое затухание без дисперсии (для узкополосных сигналов).

📎 Особенности:
   - Пренебрегаем геометрическим расхождением → квази-одномерное распространение
   - Поры — области с другими c, Q (по умолчанию: жёсткие включения отключены)
   - Явная схема 2-го порядка, с защитой по Куранту и вязкости
"""

import numpy as np
import os
import yaml
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')  # Сохранение без отображения окон
import matplotlib.pyplot as plt


def load_config(config_path: str) -> dict:
    """Загружает конфигурацию из YAML-файла."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_snapshot(n: int, u: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                  cylinder_mask: np.ndarray, pore_mask: np.ndarray,
                  pore_centers: list, pore_radius: float,
                  Lx: float, Ly: float, Lz: float,
                  impulse: np.ndarray, output_dir: str, dt: float):
    """
    Сохраняет 3 снимка на шаге `n`:
      1. x-z срез (y = mid)
      2. x-y срез (z = mid)
      3. Профиль амплитуды по оси z
    """
    os.makedirs(output_dir, exist_ok=True)

    Nx, Ny, Nz = u.shape[:3]
    mid_y = Ny // 2
    mid_z = Nz // 2

    # --- 1. x-z срез (y = mid_y) ---
    plt.figure(figsize=(5, 5))
    mask_slice = cylinder_mask[:, mid_y]                     # (Nx,)
    data = np.where(mask_slice[:, None], u[:, mid_y, :, 1], np.nan).T  # (Nz, Nx)
    im = plt.imshow(data, extent=[0, Lx, 0, Lz], origin='lower',
                    vmin=impulse.min(), vmax=impulse.max(), cmap='seismic')
    plt.colorbar(im, label="Смещение")
    plt.title(f"x-z (y=0, t={n*dt*1e6:.1f} мкс)")
    plt.xlabel("x (м)"); plt.ylabel("z (м)")

    # Наложение пор
    for xc, yc, zc in pore_centers:
        ix = np.argmin(np.abs(x - xc))
        iy = np.argmin(np.abs(y - yc))
        if not cylinder_mask[ix, iy]:
            continue
        circle = plt.Circle((xc, zc), pore_radius, color='gray',
                            fill=False, linestyle='--', linewidth=0.8)
        plt.gca().add_patch(circle)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"xz_step_{n:04d}.png"), dpi=150)
    plt.close()

    # --- 2. x-y срез (z = mid_z) ---
    plt.figure(figsize=(6, 5))
    data = np.where(cylinder_mask, u[:, :, mid_z, 1], np.nan).T
    im = plt.imshow(data, extent=[0, Lx, 0, Ly], origin='lower',
                    vmin=impulse.min(), vmax=impulse.max(), cmap='seismic')
    plt.colorbar(im, label="Смещение")
    plt.title(f"x-y (z=0, t={n*dt*1e6:.1f} мкс)")
    plt.xlabel("x (м)"); plt.ylabel("y (м)")

    for xc, yc, zc in pore_centers:
        ix = np.argmin(np.abs(x - xc))
        iy = np.argmin(np.abs(y - yc))
        if not cylinder_mask[ix, iy]:
            continue
        circle = plt.Circle((xc, yc), pore_radius, color='gray',
                            fill=False, linestyle='--', linewidth=0.8)
        plt.gca().add_patch(circle)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"xy_step_{n:04d}.png"), dpi=150)
    plt.close()

    # --- 3. Профиль амплитуды по z ---
    plt.figure(figsize=(6, 4))
    u_current = u[:, :, :, 1]              # Текущее смещение
    slice_2d = u_current[:, mid_y, :]       # (Nx, Nz)

    # Маска: внутри цилиндра И (не в поре, если поры есть)
    mask_cyl_2d = cylinder_mask[:, mid_y][:, None]  # (Nx, 1)
    if len(pore_centers) > 0:
        mask_valid = mask_cyl_2d & (~pore_mask[:, mid_y, :])  # (Nx, Nz)
    else:
        mask_valid = mask_cyl_2d

    profile_2d = np.nanmean(np.where(mask_valid, slice_2d, np.nan), axis=0)

    plt.plot(z, profile_2d, label='Среднее по срезу y=mid', linewidth=1.5)
    margin = 0.1 * max(abs(impulse.min()), abs(impulse.max()))
    plt.ylim(impulse.min() - margin, impulse.max() + margin)
    plt.title(f"Профиль амплитуды по z (шаг {n})")
    plt.xlabel("z (м)"); plt.ylabel("Смещение")
    plt.grid(True, alpha=0.5); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"profile_z_step_{n:04d}.png"), dpi=150)
    plt.close()


def wave_simulation_3d(
    sample_length: float,
    sample_diameter: float,
    source_diameter: float,
    pore_radius: float = 0.001,
    porosity_percent: float = 0.0,
    Q: float = 10.0,
    Q_pore: float = 5.0,
    c: float = 6320.0,
    c_pore: float = None,
    pulse_type: str = 'berlage',
    Nt: int = 2000,
    plot_interval: int = 0,
    save_plots: bool = False,
    output_dir: str = "output/plots"
) -> tuple:
    """
    Выполняет 3D-симуляцию по модели SLS.
    
    Параметры:
        sample_length, sample_diameter, source_diameter — в мм
        pore_radius — в м
        Q, Q_pore — добротность в матрице и порах
        c, c_pore — скорость звука (м/с)
    
    Возвращает:
        t_arr, mean_displacement, impulse, meta
    """
    # --- Установка параметров по умолчанию ---
    if c_pore is None:
        c_pore = 1500.0  # скорость в воде/воздухе по умолчанию

    # --- Геометрия и сетка ---
    Lx = Ly = sample_diameter / 1000.0   # м
    Lz = sample_length / 1000.0          # м
    Nx, Ny, Nz = 80, 80, 200
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    z = np.linspace(0, Lz, Nz)

    # --- Центральная частота и амплитуда импульса ---
    f0 = 1e6      # 1 МГц
    A = 1e-6      # условные единицы
    omega0 = 2 * np.pi * f0

    # --- Расчёт временного шага: Курант + вязкость (SLS-защита) ---
    c_max = max(c, c_pore)
    dt_cfl = 0.75 / (c_max * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))

    # Вязкость η = c²/(ωQ); берём максимальную из всех областей
    eta_max = max(c**2 / (omega0 * Q), c_pore**2 / (omega0 * Q_pore))
    dt_viscous = min(dx**2, dy**2, dz**2) / (2 * eta_max) * 0.75  # 75% запаса

    dt = min(dt_cfl, dt_viscous)
    t_arr = np.arange(Nt) * dt

    # --- Генерация импульса ---
    def berl(t):
        alpha, n = 30e5, 2
        return A * (t**n) * np.exp(-alpha*t) * np.cos(2*np.pi*f0*t)

    def rect(t, width=2e-6):
        return A * ((t >= 0) & (t <= width))

    def ricker(t, f0=f0):
        t0 = 3/f0
        pi2f2 = (np.pi*f0)**2
        tau = t - t0
        return A * (1 - 2*pi2f2*tau**2) * np.exp(-pi2f2*tau**2)

    if pulse_type == 'berlage':
        impulse = berl(t_arr)
    elif pulse_type == 'rect':
        impulse = rect(t_arr)
    elif pulse_type == 'ricker':
        impulse = ricker(t_arr)
    else:
        raise ValueError(f"pulse_type '{pulse_type}' не поддерживается")

    # --- Генерация пор ---
    pore_centers = []
    if porosity_percent > 0 and pore_radius > 0:
        sample_volume = np.pi * (Lx**2) * Lz / 4.0
        target_volume = (porosity_percent / 100.0) * sample_volume
        single_pore_volume = (4.0/3.0) * np.pi * pore_radius**3
        accumulated = 0.0

        while accumulated < target_volume:
            xc = np.random.uniform(pore_radius, Lx - pore_radius)
            yc = np.random.uniform(pore_radius, Ly - pore_radius)
            zc = np.random.uniform(pore_radius, Lz - pore_radius)

            # Проверка на пересечение
            too_close = False
            for px, py, pz in pore_centers:
                if np.linalg.norm([xc-px, yc-py, zc-pz]) < 2 * pore_radius:
                    too_close = True
                    break
            if not too_close:
                pore_centers.append((xc, yc, zc))
                accumulated += single_pore_volume

    # --- Маски: цилиндр и поры ---
    R = Lx / 2.0
    X, Y = np.meshgrid(x, y, indexing='ij')
    cylinder_mask = (X - Lx/2)**2 + (Y - Ly/2)**2 <= R**2

    pore_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    if pore_centers:
        for xc, yc, zc in pore_centers:
            dx_arr = x[:, None, None] - xc
            dy_arr = y[None, :, None] - yc
            dz_arr = z[None, None, :] - zc
            dist_sq = dx_arr**2 + dy_arr**2 + dz_arr**2
            pore_mask |= dist_sq < pore_radius**2
        # Обрезаем поры вне цилиндра
        pore_mask[~np.repeat(cylinder_mask[:, :, None], Nz, axis=2)] = False

    # --- Карты материальных параметров ---
    c_map = np.full((Nx, Ny, Nz), c, dtype=np.float32)
    Q_map = np.full((Nx, Ny, Nz), Q, dtype=np.float32)
    eta_map = c_map**2 / (omega0 * Q_map)  # η = c²/(ω₀ Q)

    if pore_centers:
        c_map[pore_mask] = c_pore
        Q_map[pore_mask] = Q_pore
        eta_map[pore_mask] = c_pore**2 / (omega0 * Q_pore)

    # --- Инициализация полей ---
    u = np.zeros((Nx, Ny, Nz), dtype=np.float64)   # смещение u(x,y,z)
    v = np.zeros((Nx, Ny, Nz), dtype=np.float64)   # скорость v = ∂u/∂t
    mean_displacement = np.zeros(Nt, dtype=np.float64)  # ← ИСПРАВЛЕНО: Nt (не Nt-1!)

    # --- Переменная для хранения предыдущего лапласиана (∂/∂t ∇²u) ---
    lap_u_prev = np.zeros((Nx-2, Ny-2, Nz-2), dtype=np.float64)

    # ------------------------------------------------------------
    # 🔁 ОСНОВНОЙ ЦИКЛ: SLS-сходимость (Nt шагов)
    # ------------------------------------------------------------
    for n in tqdm(range(Nt), desc="SLS Симуляция", unit="шаг", ncols=80):
        # 1. Вычисляем лапласиан текущего смещения: ∇²u
        lap_u_curr = (
            (u[2:, 1:-1, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]) / dx**2 +
            (u[1:-1, 2:, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, :-2, 1:-1]) / dy**2 +
            (u[1:-1, 1:-1, 2:] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, 1:-1, :-2]) / dz**2
        )  # shape: (Nx-2, Ny-2, Nz-2)

        # 2. Извлекаем локальные параметры (без границ)
        c_local = c_map[1:-1, 1:-1, 1:-1]    # (Nx-2, Ny-2, Nz-2)
        eta_local = eta_map[1:-1, 1:-1, 1:-1]

        # 3. Вычисляем ∂/∂t (∇²u) ≈ (lap_u_curr - lap_u_prev) / dt
        if n == 0:
            d_lap_dt = np.zeros_like(lap_u_curr)
        else:
            d_lap_dt = (lap_u_curr - lap_u_prev) / dt

        # 4. Ускорение по SLS: a = c² ∇²u + η ∂/∂t(∇²u)
        acceleration = c_local**2 * lap_u_curr + eta_local * d_lap_dt

        # 5. Обновляем скорость: v_new = v + dt * a
        v_new = np.zeros_like(v)
        v_new[1:-1, 1:-1, 1:-1] = v[1:-1, 1:-1, 1:-1] + dt * acceleration
        v_new[~cylinder_mask] = 0.0  # Граничное условие: вне цилиндра — покой

        # 6. Обновляем смещение: u_new = u + dt * v_new
        u_new = np.zeros_like(u)
        u_new[1:-1, 1:-1, 1:-1] = u[1:-1, 1:-1, 1:-1] + dt * v_new[1:-1, 1:-1, 1:-1]
        u_new[~cylinder_mask] = 0.0

        # 7. Граничное условие: источник на z=0 (торец)
        source_radius_m = (source_diameter / 2) / 1000.0
        source_mask = (X - Lx/2)**2 + (Y - Ly/2)**2 <= source_radius_m**2
        u_new[:, :, 0] = np.where(source_mask, impulse[n], u_new[:, :, 0])

        # 8. Обновление полей для следующего шага
        u[:] = u_new
        v[:] = v_new
        lap_u_prev[:] = lap_u_curr  # Сохраняем для ∂/∂t на следующем шаге

        # 9. Диагностика: отлов неустойчивости
        if not np.isfinite(u).all():
            print(f"\n⚠️  НЕУСТОЙЧИВОСТЬ на шаге {n}: u содержит inf/nan")
            print(f"    max|u| = {np.nanmax(np.abs(u)):.3e}")
            break

        # 10. Запись сигнала на приёмнике (z = Lz - dz)
        mean_displacement[n] = np.nanmean(u[:, :, -2])

        # 11. Сохранение снимков (если включено)
        if save_plots and plot_interval > 0 and n % plot_interval == 0:
            # save_snapshot ожидает u в формате (Nx, Ny, Nz, 2)
            u_snap = np.zeros((Nx, Ny, Nz, 2), dtype=np.float64)
            u_snap[:, :, :, 1] = u  # текущее u как "current"
            save_snapshot(
                n=n, u=u_snap, x=x, y=y, z=z,
                cylinder_mask=cylinder_mask, pore_mask=pore_mask,
                pore_centers=pore_centers, pore_radius=pore_radius,
                Lx=Lx, Ly=Ly, Lz=Lz, impulse=impulse,
                output_dir=output_dir, dt=dt
            )

    # --- Метаданные для постобработки ---
    meta = {
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
        'dx': dx, 'dy': dy, 'dz': dz,
        'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
        'x': x, 'y': y, 'z': z,
        'cylinder_mask': cylinder_mask,
        'pore_mask': pore_mask,
        'pore_centers': pore_centers,
        'dt': dt, 'c': c, 'c_pore': c_pore,
        'Q': Q, 'Q_pore': Q_pore, 'f0': f0,
        'omega0': omega0,
        'eta_max': eta_max,
        'dt_used': dt,
    }

    return t_arr, mean_displacement, impulse, meta