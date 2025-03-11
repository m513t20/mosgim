import numpy as np
from datetime import datetime
from scipy.signal import savgol_filter
import os
import itertools
from scipy.integrate import quad
import pyIGRF.calculate as calculate
from .geo import sub_sol
from . import (geo2mag,geo2modip)
# END OF GEOMAGNETIC AND MODIP COORDINATES SECTION



def sec_of_day(time:datetime.time)->int:
    return (time - time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
sec_of_day = np.vectorize(sec_of_day)


def sec_of_interval(time:datetime.time, time0:datetime.time)->int:
    return (time - time0).total_seconds()
sec_of_interval = np.vectorize(sec_of_interval, excluded='time0')


def getContInt(time:datetime.time, tec:float, lon:float, lat:float, el:float,  maxgap:int=30, maxjump:int=1)->tuple[int,int]:
    r = np.array(range(len(time)))
    idx = np.isfinite(tec) & np.isfinite(lon) & np.isfinite(lat) & np.isfinite(el) & (el > 10.)
    r = r[idx]
    intervals = []
    if len(r) == 0:
        return intervals
    beginning = r[0]
    last = r[0]
    last_time = time[last]
    for i in r[1:]:
        if abs(time[i] - last_time) > maxgap or abs(tec[i] - tec[last]) > maxjump:
            intervals.append((beginning, last))
            beginning = i
        last = i
        last_time = time[last]
        if i == r[-1]:
            intervals.append((beginning, last))
    return idx, intervals



#################### MAIN SCRIPT GOES HERE#######################################


header = ['datetime', 'el', 'ipp_lat', 'ipp_lon', 'tec']
dtype=zip(header,(object, float, float, float, float))
convert = lambda x: datetime.strptime(x.decode("utf-8"), "%Y-%m-%dT%H:%M:%S")
rootdir = '~/mosgim/002/'
time0 = datetime(2017, 1, 2)
derivative = False



Atec = Atime = Along = Alat =\
    Ael = Atime_ref =\
    Along_ref = Alat_ref = Ael_ref = np.array([])


for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".dat"):
            print (filepath)

            try:            
                data = np.genfromtxt(filepath, comments='#', names=header, dtype=(object, float, float, float, float), converters={"datetime": convert},  unpack=True)
                data = {k: arr for k, arr in zip(header, data)}
                tt = sec_of_day(data['datetime'])                
                idx, intervals = getContInt(tt, data['tec'], data['ipp_lon'], data['ipp_lat'], data['el'],  maxgap=35., maxjump=2.)

                print(intervals)

                for ii in intervals:
                    if (tt[ii[1]] - tt[ii[0]]) >= 1 * 60 * 60:    # disgard all the arcs shorter than 1 hour
                        tec_out = savgol_filter(data['tec'][ii[0]:ii[1]], 21, 2) # parabolic smoothing with 10 min window
                        time_out = data['datetime'][ii[0]:ii[1]]
                        ipp_lon_out = data['ipp_lon'][ii[0]:ii[1]]
                        ipp_lat_out = data['ipp_lat'][ii[0]:ii[1]]
                        el_out = data['el'][ii[0]:ii[1]]

                        ind_sparse = (tt[ii[0]:ii[1]] % 600 == 0)
                        tec_out = tec_out[ind_sparse]
                        time_out = time_out[ind_sparse]
                        ipp_lon_out = ipp_lon_out[ind_sparse]
                        ipp_lat_out = ipp_lat_out[ind_sparse]
                        el_out = el_out[ind_sparse]

                        if derivative == True:
                            dtec = tec_out[1:] - tec_out[0:-1]
                            time_out_ref = time_out[0:-1]
                            time_out = time_out[1:]
                            ipp_lon_out_ref = ipp_lon_out[0:-1]
                            ipp_lon_out = ipp_lon_out[1:]
                            ipp_lat_out_ref = ipp_lat_out[0:-1]
                            ipp_lat_out = ipp_lat_out[1:]
                            el_out_ref = el_out[0:-1]
                            el_out = el_out[1:]
                       
                        if derivative == False:

                            idx_min = np.argmin(tec_out)
                            tec0 = tec_out[idx_min]
                            t0 = time_out[idx_min]                            
                            ipp_lon0 = ipp_lon_out[idx_min]
                            ipp_lat0 = ipp_lat_out[idx_min]
                            el0 = el_out[idx_min]                            


                            tec_out = np.delete(tec_out, idx_min)
                            time_out = np.delete(time_out, idx_min)
                            ipp_lon_out = np.delete(ipp_lon_out, idx_min)
                            ipp_lat_out = np.delete(ipp_lat_out, idx_min)
                            el_out = np.delete(el_out, idx_min)


                            dtec = tec_out - tec0
                            time_out_ref = np.array([t0 for _ in range(len(time_out))])
                            ipp_lon_out_ref = ipp_lon0 * np.ones(len(ipp_lon_out))
                            ipp_lat_out_ref = ipp_lat0 * np.ones(len(ipp_lat_out))
                            el_out_ref = el0 * np.ones(len(el_out))

             

                        Atec = np.append(Atec, dtec)
                        Atime = np.append(Atime, time_out)
                        Along = np.append(Along, ipp_lon_out)
                        Alat = np.append(Alat, ipp_lat_out)
                        Ael = np.append(Ael, el_out)
                        Atime_ref = np.append(Atime_ref, time_out_ref)
                        Along_ref = np.append(Along_ref, ipp_lon_out_ref)
                        Alat_ref = np.append(Alat_ref, ipp_lat_out_ref)
                        Ael_ref = np.append(Ael_ref, el_out_ref)



                      
                    else: 
                        print('too short interval')

            except Exception:
                print('warning')


print ('number of observations', len(Atec))



print ('preparing coordinate system')


mcolat, mlt = geo2modip(np.pi/2 - np.deg2rad(Alat), np.deg2rad(Along), Atime)  # modip coordinates in rad
mcolat_ref, mlt_ref = geo2modip(np.pi/2 - np.deg2rad(Alat_ref), np.deg2rad(Along_ref), Atime_ref)  

mcolat1, mlt1 = geo2mag(np.pi/2 - np.deg2rad(Alat), np.deg2rad(Along), Atime)  # geomag coordinates in rad
mcolat1_ref, mlt1_ref = geo2mag(np.pi/2 - np.deg2rad(Alat_ref), np.deg2rad(Along_ref), Atime_ref)  


print ('saving input data')


np.savez('input_data_rel_modip300_2017_002.npz', day=time0,
         time=sec_of_interval(Atime, time0), mlt=mlt, mcolat=mcolat, el=np.deg2rad(Ael),
         time_ref=sec_of_interval(Atime_ref, time0), mlt_ref=mlt_ref, mcolat_ref=mcolat_ref, el_ref=np.deg2rad(Ael_ref), rhs=Atec)



np.savez('input_data_rel_gm_2017_002.npz', day=time0,
         time=sec_of_interval(Atime, time0), mlt=mlt1, mcolat=mcolat1, el=np.deg2rad(Ael),
         time_ref=sec_of_interval(Atime_ref, time0), mlt_ref=mlt1_ref, mcolat_ref=mcolat1_ref, el_ref=np.deg2rad(Ael_ref), rhs=Atec)





