# main.py
import os
import sys
from src.simulation import wave_simulation_3d, load_config
from src.postprocessing import plot_results

CONFIG_PATH = 'config/config.yaml'

if __name__ == "__main__":
    if not os.path.exists(CONFIG_PATH):
        sys.exit(f"❌ Конфигурационный файл не найден: {CONFIG_PATH}")

    config = load_config(CONFIG_PATH)
    sim = config['simulation']
    viz = config['visualization']

    print("🚀 Запуск 3D акустической симуляции...")
    t_arr, mean_disp, impulse, meta = wave_simulation_3d(
        sample_length=sim['sample_length'],
        sample_diameter=sim['sample_diameter'],
        source_diameter=sim['source_diameter'],
        pore_radius=sim['pore_radius'],
        porosity_percent=sim['porosity_percent'],
        Q=sim['Q'],
        Q_pore=sim.get('Q_pore', 5),                # ← добавлено
        c=sim['wave_speed'],
        c_pore=sim.get('wave_speed_pore', 1500),    # ← добавлено
        pulse_type=sim['pulse_type'],
        Nt=sim['Nt'],
        plot_interval=sim.get('plot_interval', 0),
        save_plots=viz['save_plots'],
        output_dir=viz['output_dir']
    )

    print("📊 Постобработка результатов...")
    Q_est = plot_results(t_arr, mean_disp, impulse, meta, config)