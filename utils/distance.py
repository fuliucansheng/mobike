import sys
sys.path.append("..")
from config import *
from utils.other import *

def get_lon_lat_lon_dict():
    result_path = Config.cache_dir + '/loc_lat_lon.dict'
    if os.path.exists(result_path):
        result = pickle.load(open(result_path,"rb"))
    else:
        train = pd.read_csv(Config.train_path)
        sub = pd.read_csv(Config.test_path)
        locs = list(set(set(train["geohashed_start_loc"]) | set(train["geohashed_end_loc"]) | set(sub["geohashed_start_loc"])))
        if np.nan in locs:
            locs.remove(np.nan)
        deloc = []
        for loc in locs:
            deloc.append(Geohash.decode_exactly(loc))
        result = dict(zip(locs,deloc))
        if Config.dump: pickle.dump(result, open(result_path,"wb"))
    return result

def get_loc_lat_lon():
    result_path = Config.cache_dir + '/loc_lat_lon.hdf'
    if os.path.exists(result_path):
        lat_lon = pd.read_hdf(result_path,"w")
    else:
        loc_dict = get_lon_lat_lon_dict()
        lat_lon = pd.DataFrame([[key, values[0], values[1]] for (key, values) in loc_dict.items()],columns=['loc', 'lat', 'lon'])
        if Config.dump: lat_lon.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return lat_lon

def get_data_time_and_distance(data):
    result_path = Config.cache_dir + '/data_time_and_distance_%s.hdf'%data.shape[0]
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path,"w")
    else:
        result = data.copy()
        result["starttime"] = pd.to_datetime(result['starttime'])
        result["date"] = pd.DatetimeIndex(result.starttime).date
        result["day"] = pd.DatetimeIndex(result.starttime).day
        result["hour"] = pd.DatetimeIndex(result.starttime).hour
        result["dayofweek"] = pd.DatetimeIndex(result.starttime).dayofweek
        result = get_distance(result, "geohashed_start_loc", "geohashed_end_loc", "distance")
        if Config.dump: result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取每条记录的距离
def get_distance(result,startcolumn,endcolumn,discolumn):
    geohashed_loc = result[[startcolumn,endcolumn]].values
    loc_dict = get_lon_lat_lon_dict()
    distance = []
    for i in geohashed_loc:
        if i[0] in loc_dict and i[1] in loc_dict:
            lat1, lon1 = loc_dict[i[0]][:2]
            lat2, lon2 = loc_dict[i[1]][:2]
            distance.append(cal_distance(float(lat1),float(lon1),float(lat2),float(lon2)))
        else:
            distance.append(-999)
    result.loc[:,discolumn] = distance
    return result

def get_manhattan_distance(result, startcolumn, endcolumn, discolumn):
    geohashed_loc = result[[startcolumn,endcolumn]].values
    loc_dict = get_lon_lat_lon_dict()
    distance = []
    for i in geohashed_loc:
        if i[0] in loc_dict and i[1] in loc_dict:
            lat1, lon1 = loc_dict[i[0]][:2]
            lat2, lon2 = loc_dict[i[1]][:2]
            a = cal_distance(float(lat1),float(lon1),float(lat1),float(lon2))
            b = cal_distance(float(lat1),float(lon1),float(lat2),float(lon1))
            distance.append(a+b)
        else:
            distance.append(-999)
    result.loc[:,discolumn] = distance
    return result

def get_degree(result):
    loc_dict = get_lon_lat_lon_dict()
    geohashed_loc = result[['geohashed_start_loc','geohashed_end_loc']].values
    degrees = []
    for i in geohashed_loc:
        if i[0] in loc_dict and i[1] in loc_dict:
            lat1, lon1 = loc_dict[i[0]][:2]
            lat2, lon2 = loc_dict[i[1]][:2]
            degrees.append(cal_degree(float(lat1),float(lon1),float(lat2),float(lon2)))
        else:
            degrees.append(-999)
    result.loc[:,'degree'] = degrees
    return result