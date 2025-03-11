import argparse
import numpy as np

from pathlib import Path

from mosgim.mosg.map_creator import solve_weights


def main() -> None:
    """
    Основная функция для создания весов MOSGIM из данных TEC.
    Загружает данные из файла, вычисляет веса и сохраняет результат в файл.
    """
    parser = argparse.ArgumentParser(description='Solve raw TECs to MOSGIM weights')
    parser.add_argument(
        '--in_file', 
        type=Path, 
        default=Path('/tmp/prepared_modip.npz'),
        help='Path to data, after prepare script'
    )
    parser.add_argument(
        '--out_file', 
        type=Path, 
        default=Path('/tmp/mosgim_weights.npz'),
        help='Path to data, after prepare script'
    )
    
    args = parser.parse_args()
    input_file = args.in_file
    output_file = args.out_file
    
    data = np.load(input_file, allow_pickle=True)
    weights, N = solve_weights(data)
    
    np.savez(output_file, res=weights, N=N)

