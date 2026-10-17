import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import hydra
from hydra.utils import instantiate

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.plots import (
    plot_bev_frame,
    plot_bev_with_agent,
    plot_cameras_frame,
    plot_cameras_frame_with_annotations,
    plot_cameras_frame_with_lidar,
    configure_bev_ax,
    frame_plot_to_gif,
)
from navsim.visualization.bev import (
    add_annotations_to_bev_ax,
    add_lidar_to_bev_ax,
)
from navsim.agents.constant_velocity_agent import ConstantVelocityAgent

# 0. Создаем папку для сохранения результатов на сервере
output_dir = Path("./navsim_output_images")
output_dir.mkdir(exist_ok=True)

# 1. Инициализация датасета (с вашими путями)
SPLIT = "test"  # ["mini", "test", "trainval"]
FILTER = "navtest"

hydra.initialize(
    config_path="navsim/planning/script/config/common/train_test_split/scene_filter",
    version_base=None,
)
cfg = hydra.compose(config_name=FILTER)
scene_filter: SceneFilter = instantiate(cfg)
os.environ["OPENSCENE_DATA_ROOT"] = "/auto/ws/drivesuprim_workspace/dataset"  # замените на ваш точный путь, если он отличается

openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

scene_loader = SceneLoader(
    openscene_data_root / f"navsim_logs/{SPLIT}",  # data_path
    openscene_data_root / f"sensor_blobs/{SPLIT}",  # original_sensor_path
    scene_filter,
    None,  # synthetic_sensor_path
    None,  # synthetic_scenes_path
    sensor_config=SensorConfig.build_all_sensors(),
)

# 2. Выбираем случайную сцену и токен
token = np.random.choice(scene_loader.tokens)
scene = scene_loader.get_scene_from_token(token)
frame_idx = scene.scene_metadata.num_history_frames - 1  # текущий кадр

print(f"Выбрана сцена с токеном: {token}")

# --- 3. Виды сверху (BEV) ---
fig, ax = plot_bev_frame(scene, frame_idx)
fig.savefig(output_dir / f"{token}_bev_frame.png", bbox_inches="tight", dpi=300)
plt.close(fig)

# --- 4. BEV с агентом ---
agent = ConstantVelocityAgent()
fig, ax = plot_bev_with_agent(scene, agent)
fig.savefig(output_dir / f"{token}_bev_with_agent.png", bbox_inches="tight", dpi=300)
plt.close(fig)

# --- 5. Камеры (сетка 3x3) ---
fig, ax = plot_cameras_frame(scene, frame_idx)
fig.savefig(output_dir / f"{token}_cameras_frame.png", bbox_inches="tight", dpi=300)
plt.close(fig)

# --- 6. Камеры с аннотациями ---
fig, ax = plot_cameras_frame_with_annotations(scene, frame_idx)
fig.savefig(output_dir / f"{token}_cameras_with_annotations.png", bbox_inches="tight", dpi=300)
plt.close(fig)

# --- 7. Камеры с LiDAR ---
fig, ax = plot_cameras_frame_with_lidar(scene, frame_idx)
fig.savefig(output_dir / f"{token}_cameras_with_lidar.png", bbox_inches="tight", dpi=300)
plt.close(fig)

# --- 8. Кастомный график ---
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_title("Custom plot")
add_annotations_to_bev_ax(ax, scene.frames[frame_idx].annotations)
add_lidar_to_bev_ax(ax, scene.frames[frame_idx].lidar)
configure_bev_ax(ax)
fig.savefig(output_dir / f"{token}_custom_plot.png", bbox_inches="tight", dpi=300)
plt.close(fig)

# --- 9. Создание GIF-анимации ---
print("Генерация GIF-анимации (это может занять немного времени)...")
frame_indices = [idx for idx in range(len(scene.frames))]
gif_path = output_dir / f"{token}_animation.gif"
images = frame_plot_to_gif(
    str(gif_path), 
    plot_cameras_frame_with_annotations, 
    scene, 
    frame_indices
)

print(f"\nГотово! Все файлы (картинки и GIF) сохранены на сервере в папке: {output_dir.resolve()}")
