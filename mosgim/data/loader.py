import os
import time
import numpy as np
import re
import h5py
import concurrent.futures
from datetime import datetime
from warnings import warn
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from numpy import( cos, sin, sqrt, 
arctan, arctan2, rad2deg)

from mosgim.geo.geo import HM
from mosgim.geo.geo import sub_ionospheric


class Loader():
    
    def __init__(self):
        self.FIELDS = ['datetime', 'el', 'ipp_lat', 'ipp_lon', 'tec']
        self.DTYPE = (object, float, float, float, float)
        self.not_found_sites = []

class LoaderTxt(Loader):
    
    def __init__(self, root_dir:Path):
        super().__init__()
        self.dformat = "%Y-%m-%dT%H:%M:%S"
        self.root_dir = root_dir

    def get_files(self, rootdir:Path)->defaultdict[str,list[str]]:
        """
        Root directroy must contain folders with site name 
        Inside subfolders are *.dat files for every satellite
        """
        result = defaultdict(list)
        for subdir, _, files in os.walk(rootdir):
            for filename in files:
                filepath = Path(subdir) / filename
                if str(filepath).endswith(".dat"):
                    site = filename[:4]
                    if site != subdir[-4:]:
                        raise ValueError(f'{site} in {subdir}. wrong site name')
                    result[site].append(filepath)
                else:
                    warn(f'{filepath} in {subdir} is not data file')
        for site in result:
            result[site].sort()
        return result

    def load_data(self, filepath:Path)->tuple[np.array,Path]:
        convert = lambda x: datetime.strptime(x.decode("utf-8"), self.dformat)
        data = np.genfromtxt(filepath, 
                             comments='#', 
                             names=self.FIELDS, 
                             dtype=self.DTYPE,
                             converters={"datetime": convert},  
                             #unpack=True
                             )

        #tt = sec_of_day(data['datetime'])
        #data = append_fields(data, 'sec_of_day', tt, np.float)
        return data, filepath
    

    def generate_data(self, sites:list[str]=[]):
        files = self.get_files(self.root_dir)
        print(f'Collected {len(files)} sites')
        self.not_found_sites = sites[:]
        for site, site_files in files.items():
            if sites and not site in sites:
                continue
            self.not_found_sites.remove(site)
            count = 0
            st = time.time()
            for sat_file in site_files:
                try:
                    data, _ = self.load_data(sat_file)
                    count += 1
                    yield data, sat_file
                except Exception as e:
                    print(f'{sat_file} not processed. Reason: {e}')
            print(f'{site} contribute {count} files, takes {time.time() - st}')
            
    def generate_data_pool(self, sites:list[str]=[], nworkers:int=1):
        files = self.get_files(self.root_dir)
        print(f'Collected {len(files)} sites')
        self.not_found_sites = sites[:]
        with ProcessPoolExecutor(max_workers=nworkers) as executor:
            queue = []
            for site, site_files in files.items():
                if sites and not site in sites:
                    continue
                self.not_found_sites.remove(site)
                count = 0
                st = time.time()
                for sat_file in site_files:
                    try:
                        query = executor.submit(self.load_data, sat_file)
                        queue.append(query)
                    except Exception as e:
                        print(f'{sat_file} not processed. Reason: {e}')
                print(site)
            for cur_future in concurrent.futures.as_completed(queue):
                yield cur_future.result()


class LoaderHDF(Loader):
    
    def __init__(self, hdf_path:Path):
        super().__init__()
        self.hdf_path = hdf_path
        
    def get_files(self)->list[Path]:
        result = []
        for subdir, _, files in os.walk(self.hdf_path):
            for filename in files:
                filepath = Path(subdir) / filename
                if str(filepath).endswith(".h5"):
                    result.append(filepath)
        if len(result) != 1:
            msg = f'Must be exactly one hdf in {self.hdf_path} or subfolders'
            raise ValueError(msg)
        return result
    
    def __get_hdf_file(self)->Path:
        return self.get_files()[0]
    
    def generate_data(self, sites:list[str]=[]):
        hdf_file = h5py.File(self.__get_hdf_file(), 'r')
        self.not_found_sites = sites[:]
        for site in hdf_file:
            if sites and not site in sites:
                continue
            self.not_found_sites.remove(site)
            slat = hdf_file[site].attrs['lat']
            slon = hdf_file[site].attrs['lon']
            st = time.time()
            count = 0
            for sat in hdf_file[site]:
                sat_data = hdf_file[site][sat]
                arr = np.empty((len(sat_data['tec']),), 
                               list(zip(self.FIELDS,self.DTYPE)))
                el = sat_data['elevation'][:]
                az = sat_data['azimuth'][:]
                ts = sat_data['timestamp'][:]
                ipp_lat, ipp_lon = sub_ionospheric(slat, slon, HM, az, el)
                
                arr['datetime'] = np.array([datetime.fromtimestamp(float(t)) for t in ts])
                arr['el'] = np.rad2deg(el)
                arr['ipp_lat'] = np.rad2deg(ipp_lat)
                arr['ipp_lon'] = np.rad2deg(ipp_lon)
                arr['tec'] = sat_data['tec'][:]
                count += 1
                yield arr, sat + '_' + site
            print(f'{site} contribute {count} files, takes {time.time() - st}')


class LoaderRinex(Loader):
    
    def __init__(self, rinex_path:Path, nav_path:Path):
        super().__init__()
        self.rinex_path = rinex_path
        self.nav_path = nav_path
        self.o_regx = re.compile((r'.[1-9][1-9]o'))

    def __is_ofile(self, file:Path)->bool:
        is_ofile = False
        is_ofile = is_ofile or bool(self.o_regx.match(file[-4:])) 
        is_ofile = is_ofile or file.endswith('.rnx')
        is_ofile = is_ofile or file.endswith('.RNX')
        return is_ofile
        
    def get_files(self)->dict[str,Path]:
        result = dict()
        for subdir, _, files in os.walk(self.rootdir):
            for filename in files:
                filepath = Path(subdir) / filename
                if not self.__is_ofile(filename):
                    warn(f'{filepath} is not observation file')
                    continue
                site = filename[:4]
                if site in result:
                    msg = f'Duplicated file {filename}, was {result[site]}'
                    raise ValueError(msg)
                result[site] = filepath
        return result
    
