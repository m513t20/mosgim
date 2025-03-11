import os
import argparse
import time
import numpy as np

from pathlib import Path
from datetime import datetime, timedelta

from mosgim.data import (DataSourceType,
                                       MagneticCoordType,
                                       ProcessingType,
                                       process_data,
                                       combine_data,
                                       get_data,
                                       save_data,
                                       sites,
                                       calculate_seed_mag_coordinates_parallel)
from mosgim.data import (LoaderHDF, 
                                LoaderTxt)
from mosgim.mosg.map_creator import (solve_weights,
                                calculate_maps)
from mosgim.mosg.lcp_solver import create_lcp
from mosgim.plotter.animation import plot_and_save 
                                  



def populate_out_path(args: argparse.Namespace) -> None:
    """
    Заполняет пути для выходных файлов на основе аргументов командной строки.

    :param args: Аргументы командной строки.
    """
    date = args.date
    mag_type = args.mag_type
    out_path = args.out_path
    
    if out_path:
        if not args.modip_file:
            args.modip_file = out_path / f'prepared_mdip_{date}.npz'
        if not args.mag_file:
            args.mag_file = out_path / f'prepared_mag_{date}.npz'
        if not args.weight_file:
            args.weight_file = out_path / f'weights_{mag_type}_{date}.npz'
        if not args.lcp_file:
            args.lcp_file = out_path / f'lcp_{mag_type}_{date}.npz'
        if not args.maps_file:
            args.maps_file = out_path / f'maps_{mag_type}_{date}.npz'
        if not args.animation_file:
            args.animation_file = out_path / f'animation_{mag_type}_{date}.mp4'


def parse_args(command: str = '') -> argparse.Namespace:
    """
    Парсит аргументы командной строки.

    :param command: Строка с аргументами командной строки (опционально).
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
        '--out_path', 
        type=Path, 
        default=Path('/tmp/'),
        help='Path where results are stored'
    )
    parser.add_argument(
        '--process_type', 
        type=ProcessingType, 
        required=True,
        help='Type of processing [single | ranged]'
    )
    parser.add_argument(
        '--data_source', 
        type=DataSourceType, 
        required=True,
        help='Path to data, content depends on format, [hdf | txt | rinex]'
    )
    parser.add_argument(
        '--date',  
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        required=True,
        help='Date of data, example 2017-01-02'
    )
    parser.add_argument(
        '--ndays', 
        type=int, 
        required=True,
        help='Number of days to process in "ranged" processing'
    )
    parser.add_argument(
        '--modip_file',  
        type=Path,
        help='Path to file with results, for modip'
    )
    parser.add_argument(
        '--mag_file',  
        type=Path,
        help='Path to file with results, for magnetic lat'
    )
    parser.add_argument(
        '--weight_file',  
        type=Path,
        help='Path to file with solved weights'
    )
    parser.add_argument(
        '--lcp_file',  
        type=Path,
        help='LCP file'
    )
    parser.add_argument(
        '--mag_type',  
        type=MagneticCoordType,
        required=True,
        help='Type of magnetic coords [mag | mdip]'
    )
    parser.add_argument(
        '--nsite',  
        type=int,
        help='Number of sites to take into calculations'
    )
    parser.add_argument(
        '--nworkers',  
        type=int,
        default=1,
        help='Number of threads for parallel processing'
    )
    parser.add_argument(
        '--memory_per_worker',  
        type=int,
        default=2,
        help='Number of Gb per worker'
    )
    parser.add_argument(
        '--skip_prepare',
        action='store_true',
        help='Skip data reading use existing files'
    )
    parser.add_argument(
        '--animation_file',  
        type=Path,
        help='Path to animation'
    )
    parser.add_argument(
        '--maps_file',  
        type=Path,
        help='Path to map data'
    )
    parser.add_argument(
        '--const',  
        action='store_true',
        help='Defines '
    )
    
    if command:
        args = parser.parse_args(command.split())
    else:
        args = parser.parse_args()
    
    os.makedirs(args.out_path, exist_ok=True)
    
    if args.process_type == ProcessingType.ranged and args.ndays is None:
        parser.error("Ranged processing requires --ndays")
    
    if args.process_type == ProcessingType.ranged:
        base_time = args.date
        for day in range(args.ndays):
            current_args = argparse.Namespace(**vars(args))
            current_date = base_time + timedelta(day)
            doy = str(current_date.timetuple().tm_yday).zfill(3)
            current_args.date = current_date
            current_args.data_path = f"{args.data_path}/{current_date.year}/{doy}"
            populate_out_path(current_args)
            yield current_args
    elif args.process_type == ProcessingType.single:
        populate_out_path(args)
        yield args


def process(args: argparse.Namespace) -> None:
    """
    Основная функция для обработки данных.
    Загружает данные, вычисляет магнитные координаты, веса, LCP, карты и анимацию.

    :param args: Аргументы командной строки.
    """
    print(args)
    process_date = args.date
    start_time = time.time()
    
    if not args.skip_prepare:
        selected_sites = sites[:args.nsite] if args.nsite else sites[:]
        
        if args.data_source == DataSourceType.hdf:
            loader = LoaderHDF(args.data_path)
            data_generator = loader.generate_data(sites=selected_sites)
        elif args.data_source == DataSourceType.txt:
            loader = LoaderTxt(args.data_path)
            data_generator = loader.generate_data_pool(sites=selected_sites, nworkers=args.nworkers)
        else:
            raise ValueError('Define data source')
        
        data = process_data(data_generator)
        print(loader.not_found_sites)
        print(f'Done reading in {time.time() - start_time}')
        
        data_chunks = combine_data(data, nchunks=args.nworkers)
        print('Start magnetic calculations...')
        start_time = time.time()
        result = calculate_seed_mag_coordinates_parallel(data_chunks, nworkers=args.nworkers)
        print(f'Done, took {time.time() - start_time}')
        
        if args.mag_file and args.modip_file:
            save_data(result, args.modip_file, args.mag_file, process_date)
        
        data = get_data(result, args.mag_type, process_date)
    else:
        if args.mag_type == MagneticCoordType.mag:
            data = np.load(args.mag_file, allow_pickle=True)
        elif args.mag_type == MagneticCoordType.mdip:
            data = np.load(args.modip_file, allow_pickle=True)
    
    weights, N = solve_weights(data, nworkers=args.nworkers, gigs=args.memory_per_worker, linear=not args.const)
    
    if args.weight_file:
        np.savez(args.weight_file, res=weights, N=N)
    
    try:
        lcp = create_lcp({'res': weights, 'N': N})
    except Exception as e:
        print(f'Could not finish calculation, LCP is failed: {e}')
        return
    
    if args.lcp_file:
        np.savez(args.lcp_file, res=lcp, N=N)
    
    maps = calculate_maps(lcp, args.mag_type, process_date)
    plot_and_save(maps, args.animation_file, args.maps_file)


if __name__ == '__main__':
    for args in parse_args():
        process(args)
