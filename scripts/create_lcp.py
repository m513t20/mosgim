import argparse
import numpy as np

from pathlib import Path
from loguru import logger

from mosgim.mosg.lcp_solver import create_lcp as crelcp


def main() -> None:
    """
    Основная функция для создания LCP (Linear Combination Parameters) из данных TEC.
    Загружает данные из файла, вычисляет LCP и сохраняет результат в файл.
    """
    parser = argparse.ArgumentParser(description='Solve raw TECs to LCP')
    parser.add_argument(
        '--in_file', 
        type=Path, 
        default=Path('/tmp/mosgim_weights.npz'),
        help='Path to data, after prepare script'
    )
    parser.add_argument(
        '--out_file', 
        type=Path, 
        default=Path('/tmp/lcp.npz'),
        help='Path to data, after prepare script'
    )
    
    args = parser.parse_args()
    input_file = args.in_file
    output_file = args.out_file
    
    data = np.load(input_file, allow_pickle=True)
    lcp_result = crelcp(data)
    
    np.savez(output_file, res=lcp_result, N=data['N'])
    logger.success(f"{output_file} saved successfully")

