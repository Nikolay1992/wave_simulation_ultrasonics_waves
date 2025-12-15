# src/simulation.py
import numpy as np
import os
import yaml
from tqdm.auto import tqdm  # авто-выбор: tqdm или tqdm.notebook
import matplotlib
matplotlib.use('Agg')  # не показывать окна, только сохранять
import matplotlib.pyplot as plt


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_snapshot(n, u, x, y, z, cylinder_mask, pore_mask, pore_centers, pore_radius,
                  Lx, Ly, Lz, impulse, output_dir, dt):
    """
    Сохраняет 3 изображения на шаге n:
      1. x-z срез (y = mid)
      2. x-y срез (z = mid)
      3. Профиль амплитуды по z
    """
    os.makedirs(output_dir, exist_ok=True)

    Nx, Ny, Nz = u.shape[:3]
    mid_y = Ny // 2
    mid_z = Nz // 2

    # --- 1. x-z срез (y = mid_y) ---
    plt.figure(figsize=(8, 4))
    mask_slice = cylinder_mask[:, mid_y]  # (Nx,)
    data = np.where(mask_slice[:, None], u[:, mid_y, :, 1], np.nan).T  # (Nz, Nx)
    im = plt.imshow(data, extent=[0, Lx, 0, Lz], origin='lower',
                    vmin=impulse.min(), vmax=impulse.max(), cmap='seismic')
    plt.colorbar(im, label="Смещение")
    plt.title(f"x-z срез (y=0, шаг {n}, t={n*dt*1e6:.1f} мкс)")
    plt.xlabel("x (м)"); plt.ylabel("z (м)")

    for xc, yc, zc in pore_centers:
        ix = np.argmin(np.abs(x - xc))
        iy = np.argmin(np.abs(y - yc))
        if not cylinder_mask[ix, iy]:
            continue
        circle = plt.Circle((xc, zc), pore_radius, color='gray', fill=False, linestyle='--', linewidth=0.8)
        plt.gca().add_patch(circle)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"xz_step_{n:04d}.png"), dpi=150)
    plt.close()

    # --- 2. x-y срез (z = mid_z) ---
    plt.figure(figsize=(6, 5))
    data = np.where(cylinder_mask, u[:, :, mid_z, 1], np.nan).T  # (Ny, Nx)
    im = plt.imshow(data, extent=[0, Lx, 0, Ly], origin='lower',
                    vmin=impulse.min(), vmax=impulse.max(), cmap='seismic')
    plt.colorbar(im, label="Смещение")
    plt.title(f"x-y срез (z=0, шаг {n})")
    plt.xlabel("x (м)"); plt.ylabel("y (м)")

    for xc, yc, zc in pore_centers:
        ix = np.argmin(np.abs(x - xc))
        iy = np.argmin(np.abs(y - yc))
        if not cylinder_mask[ix, iy]:
            continue
        circle = plt.Circle((xc, yc), pore_radius, color='gray', fill=False, linestyle='--', linewidth=0.8)
        plt.gca().add_patch(circle)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"xy_step_{n:04d}.png"), dpi=150)
    plt.close()

    # --- 3. Профиль по z ---
    plt.figure(figsize=(6, 4))
    u_current = u[:, :, :, 1]  # (Nx, Ny, Nz)
    slice_2d = u_current[:, mid_y, :]  # (Nx, Nz)

    mask_cyl_2d = cylinder_mask[:, mid_y][:, None]  # (Nx, 1)
    if len(pore_centers) > 0:
        mask_valid = mask_cyl_2d & (~pore_mask[:, mid_y, :])  # (Nx, Nz)
    else:
        mask_valid = mask_cyl_2d  # broadcasting: (Nx,1) → (Nx,Nz)

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
    sample_length,
    sample_diameter,
    source_diameter,
    pore_radius=0.001,
    porosity_percent=0,
    Q=10,
    Q_pore=5,               # ← новое: добротность в порах
    c=6320,
    c_pore=None,            # ← новое: скорость в порах
    pulse_type='berlage',
    Nt=2000,
    plot_interval=0,
    save_plots=False,
    output_dir="output/plots"
):
    # --- Совместимость: значение по умолчанию для c_pore ---
    if c_pore is None:
        c_pore = 1500.0  # скорость в воде/воздухе

    # --- Параметры модели ---
    Lx = Ly = sample_diameter / 1000.0
    Lz = sample_length / 1000.0
    Nx, Ny, Nz = 80, 80, 200
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    z = np.linspace(0, Lz, Nz)

    # Условие устойчивости: используем МАКСИМАЛЬНУЮ скорость
    c_max = max(c, c_pore)
    dt = 0.95 / (c_max * np.sqrt(1 / dx ** 2 + 1 / dy ** 2 + 1 / dz ** 2))
    t_arr = np.arange(Nt) * dt

    f0 = 1e6
    alpha = 30e5
    A = 1e-6
    n_pow = 2

    def berl(t):
        return A * (t ** n_pow) * np.exp(-alpha * t) * np.cos(2 * np.pi * f0 * t)

    def rect(t, width=2e-6):
        return A * ((t >= 0) & (t <= width))

    def ricker(t, f0=f0):
        t0 = 3 / f0
        pi2f2 = (np.pi * f0) ** 2
        tau = t - t0
        return A * (1 - 2 * pi2f2 * tau ** 2) * np.exp(-pi2f2 * tau ** 2)

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
        sample_volume = np.pi * (Lx ** 2) * Lz / 4.0
        target_pore_volume = (porosity_percent / 100.0) * sample_volume
        single_pore_volume = (4.0 / 3.0) * np.pi * pore_radius ** 3
        accumulated_volume = 0.0

        while accumulated_volume < target_pore_volume:
            xc = np.random.uniform(pore_radius, Lx - pore_radius)
            yc = np.random.uniform(pore_radius, Ly - pore_radius)
            zc = np.random.uniform(pore_radius, Lz - pore_radius)

            too_close = any(
                np.linalg.norm([xc - px, yc - py, zc - pz]) < 2 * pore_radius
                for px, py, pz in pore_centers
            )
            if not too_close:
                pore_centers.append((xc, yc, zc))
                accumulated_volume += single_pore_volume

    # --- Маски и карты свойств ---
    R = Lx / 2.0
    X, Y = np.meshgrid(x, y, indexing='ij')
    cylinder_mask = (X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2 <= R ** 2

    pore_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    if pore_centers:
        for xc, yc, zc in pore_centers:
            dx_arr = x[:, None, None] - xc
            dy_arr = y[None, :, None] - yc
            dz_arr = z[None, None, :] - zc
            dist_sq = dx_arr ** 2 + dy_arr ** 2 + dz_arr ** 2
            pore_mask |= dist_sq < pore_radius ** 2
        pore_mask[~np.repeat(cylinder_mask[:, :, None], Nz, axis=2)] = False

    # ✅ Карта скорости и добротности
    c_map = np.full((Nx, Ny, Nz), c, dtype=np.float32)
    Q_map = np.full((Nx, Ny, Nz), Q, dtype=np.float32)

    if pore_centers:
        c_map[pore_mask] = c_pore
        Q_map[pore_mask] = Q_pore

    # --- Инициализация поля u (только y-компонента) ---
    u = np.zeros((Nx, Ny, Nz, 2))  # [x, y, z, время: 0=prev, 1=curr]
    omega_0 = 2 * np.pi * f0
    mean_displacement = np.zeros(Nt - 2)

    # ------------------------------------------------------------
    # 🔁 ОСНОВНОЙ ЦИКЛ С ПРОГРЕСС-БАРОМ
    # ------------------------------------------------------------
    for n in tqdm(range(1, Nt - 1), desc="Симуляция", unit="шаг", ncols=80):
        # --- Локальные свойства в центральной области ---
        c_local = c_map[1:-1, 1:-1, 1:-1]      # (Nx-2, Ny-2, Nz-2)
        Q_local = Q_map[1:-1, 1:-1, 1:-1]

        # --- Лапласиан по y-компоненте ---
        laplacian = (
            (u[2:, 1:-1, 1:-1, 1] - 2 * u[1:-1, 1:-1, 1:-1, 1] + u[:-2, 1:-1, 1:-1, 1]) / dx ** 2 +
            (u[1:-1, 2:, 1:-1, 1] - 2 * u[1:-1, 1:-1, 1:-1, 1] + u[1:-1, :-2, 1:-1, 1]) / dy ** 2 +
            (u[1:-1, 1:-1, 2:, 1] - 2 * u[1:-1, 1:-1, 1:-1, 1] + u[1:-1, 1:-1, :-2, 1]) / dz ** 2
        )

        du_dt = (u[1:-1, 1:-1, 1:-1, 1] - u[1:-1, 1:-1, 1:-1, 0]) / dt
        attenuation_term = (omega_0 / Q_local) * du_dt

        # 🔹 Волновое обновление с локальной скоростью и затуханием
        u_new_inner = (
            2 * u[1:-1, 1:-1, 1:-1, 1] - u[1:-1, 1:-1, 1:-1, 0] +
            (dt ** 2) * (c_local ** 2) * laplacian -
            (dt ** 2) * attenuation_term
        )

        u_new = np.zeros((Nx, Ny, Nz))
        mask_inner = cylinder_mask[1:-1, 1:-1][:, :, None] & (~pore_mask[1:-1, 1:-1, 1:-1])
        #u_new[1:-1, 1:-1, 1:-1] = np.where(mask_inner, u_new_inner, 0.0)
        u_new[1:-1, 1:-1, 1:-1] = u_new_inner
        u_new[~cylinder_mask] = 0.0

        # Источник на z=0 (только в твёрдой фазе)
        source_radius_m = (source_diameter / 2) / 1000.0
        source_mask = (X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2 <= source_radius_m ** 2 #& (~pore_mask[:, :, 0]
        u_new[:, :, 0] = np.where(source_mask, impulse[n], 0.0)

        # Обновление временного слоя
        u[:, :, :, 0] = u[:, :, :, 1]
        u[:, :, :, 1] = u_new

        # Запись смещения на выходе (z = Lz - dz)
        mean_displacement[n - 1] = np.nanmean(u[:, :, -2, 1])

        # 🔔 Сохранение снимков
        if save_plots and plot_interval > 0 and n % plot_interval == 0:
            save_snapshot(
                n=n,
                u=u,
                x=x,
                y=y,
                z=z,
                cylinder_mask=cylinder_mask,
                pore_mask=pore_mask,
                pore_centers=pore_centers,
                pore_radius=pore_radius,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                impulse=impulse,
                output_dir=output_dir,
                dt=dt
            )

    # --- Метаинформация ---
    meta = {
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
        'dx': dx, 'dy': dy, 'dz': dz,
        'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
        'x': x, 'y': y, 'z': z,
        'cylinder_mask': cylinder_mask,
        'pore_mask': pore_mask,
        'pore_centers': pore_centers,
        'dt': dt,
        'c': c,
        'c_pore': c_pore,
        'Q': Q,
        'Q_pore': Q_pore,
        # 'c_map': c_map,   # можно раскомментировать, если нужна в postprocessing
        # 'Q_map': Q_map,
        'f0': f0,
    }

    return t_arr, mean_displacement, impulse, meta