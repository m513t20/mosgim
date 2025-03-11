import argparse
import numpy as np

from datetime import datetime
from pathlib import Path

from mosgim.data import MagneticCoordType
from mosgim.plotter.animation import plot_and_save
from mosgim.mosg.map_creator import calculate_maps



def main() -> None:
    """
    Основная функция для создания карт из данных LCP.
    Загружает данные, вычисляет карты и сохраняет их в файл, а также создает анимацию.
    """
    parser = argparse.ArgumentParser(description='Solve raw TECs to maps')
    parser.add_argument(
        '--in_file', 
        type=Path, 
        default=Path('/tmp/lcp.npz'),
        help='Path to data, after prepare script'
    )
    parser.add_argument(
        '--out_file', 
        type=Path, 
        default=Path('/tmp/map.npz'),
        help='Path to data, after prepare script'
    )
    parser.add_argument(
        '--animation_file', 
        type=Path, 
        default=Path('/tmp/animation.mp4'),
        help='Path to animation file'
    )
    
    args = parser.parse_args()
    input_file = args.in_file
    output_file = args.out_file
    animation_file = args.animation_file
    
    data = np.load(input_file, allow_pickle=True)
    maps = calculate_maps(data['res'], MagneticCoordType.mdip, datetime(2017, 1, 2))
    
    plot_and_save(maps, animation_file, output_file)
