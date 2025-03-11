import time
import argparse
import numpy as np

from datetime import datetime
from pathlib import Path

from mosgim.data import (LoaderTxt, 
                        LoaderHDF)
from mosgim.data import (process_data,
                        combine_data,
                        calculate_seed_mag_coordinates_parallel,
                        save_data,
                        sites,
                        DataSourceType)



def parse_args() -> argparse.Namespace:
    """
    Парсит аргументы командной строки.

    :return: Объект с аргументами командной строки.
    """
    parser = argparse.ArgumentParser(description='Prepare data from txt, hdf, or RInEx')
    parser.add_argument(
        '--data_path', 
        type=Path, 
        required=True,
        help='Path to data, content depends on format'
    )
    parser.add_argument(
        '--data_source', 
        type=DataSourceType, 
        required=True,
        help='Path to data, content depends on format'
    )
    parser.add_argument(
        '--date',  
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        required=True,
        help='Date of data, example 2017-01-02'
    )
    parser.add_argument(
        '--modip_file',  
        type=Path,
        default=Path('/tmp/prepared_modip.npz'),
        help='Path to file with results, for modip'
    )
    parser.add_argument(
        '--mag_file',  
        type=Path,
        default=Path('/tmp/prepared_mag.npz'),
        help='Path to file with results, for magnetic lat'
    )
    parser.add_argument(
        '--nsite',  
        type=int,
        help='Number of sites to take into calculations'
    )
    return parser.parse_args()


def load_data(data_path: Path, data_source: DataSourceType, selected_sites: list) -> np.ndarray:
    """
    Загружает данные в зависимости от источника.

    :param data_path: Путь к данным.
    :param data_source: Тип источника данных (hdf, txt и т.д.).
    :param selected_sites: Список выбранных сайтов для загрузки.
    :return: Загруженные данные.
    """
    if data_source == DataSourceType.hdf:
        loader = LoaderHDF(data_path)
        data_generator = loader.generate_data(sites=selected_sites)
    elif data_source == DataSourceType.txt:
        loader = LoaderTxt(data_path)
        data_generator = loader.generate_data(sites=selected_sites)
    else:
        raise ValueError(f"Unsupported data source: {data_source}")
    
    return process_data(data_generator)


def calculate_magnetic_coordinates(data: np.ndarray) -> np.ndarray:
    """
    Вычисляет магнитные координаты.

    :param data: Входные данные для расчета.
    :return: Данные с рассчитанными магнитными координатами.
    """
    print('Start magnetic calculations...')
    start_time = time.time()
    data_chunks = combine_data(data, nchunks=1)

    result = calculate_seed_mag_coordinates_parallel(data_chunks)
    print(f'Done, took {time.time() - start_time} seconds')
    return result


def main() -> None:
    """
    Основная функция для загрузки и подготовки данных.
    """
    args = parse_args()
    process_date = args.date
    selected_sites = sites[:args.nsite] if args.nsite else sites[:]

    # Загрузка данных
    data = load_data(args.data_path, args.data_source, selected_sites)

    # Вычисление магнитных координат
    result = calculate_magnetic_coordinates(data)

    # Сохранение результатов
    save_data(result, args.modip_file, args.mag_file, process_date)
    print(f"Data saved to {args.modip_file} and {args.mag_file}")


if __name__ == '__main__':
    main()
