from celluloid import Camera
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_and_save(maps: dict, animation_file: Path, maps_file: Path, **kwargs) -> None:
    """
    Создает анимацию и сохраняет её в файл, а также сохраняет данные карт в файл.

    :param maps: Словарь с данными карт, включая долготы, широты и значения для каждого временного шага.
    :param animation_file: Путь для сохранения анимации.
    :param maps_file: Путь для сохранения данных карт.
    :param kwargs: Дополнительные параметры, такие как `max_tec` (максимальное значение TEC).
    """
    max_tec = kwargs.get('max_tec', 40)
    maps_keys = [k for k in maps if k.startswith('time')]
    maps_keys.sort()
    
    fig = plt.figure()
    camera = Camera(fig)
    levels = np.arange(0, max_tec, 0.5)
    
    for key in maps_keys:
        plt.contourf(maps['lons'], maps['lats'], maps[key], levels, cmap=plt.cm.jet)
        camera.snap()
    
    anim = camera.animate()
    anim.save(animation_file)
    np.savez(maps_file, maps)
