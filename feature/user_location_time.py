import sys
sys.path.append("..")
from config import *
from utils.other import *
from utils.distance import *

traincopy = None

def get_sloc_hour_fre(train, result):
    user_sloc_hour_fre = traincopy.groupby(["userid","hour","geohashed_start_loc"],as_index=False)["orderid"].agg({"user_sloc_hour_fre":"count"})
    result = pd.merge(result,user_sloc_hour_fre,on=["userid","hour","geohashed_start_loc"],how="left")
    return result

def get_eloc_hour_fre(train, result):
    user_eloc_hour_fre = traincopy.groupby(["userid","hour","geohashed_end_loc"],as_index=False)["orderid"].agg({"user_eloc_hour_fre":"count"})
    result = pd.merge(result,user_eloc_hour_fre,on=["userid","hour","geohashed_end_loc"],how="left")
    return result

def get_sloc_in_user_hourcount(train, result):
    sloc_in_user_hourcount = traincopy.groupby(["userid","geohashed_start_loc"],as_index=False)["hour"].agg({"sloc_in_user_hourcount":lambda x: np.size(np.unique(x))})
    result = pd.merge(result,sloc_in_user_hourcount,on=["userid","geohashed_start_loc"],how="left")
    return result

def get_user_location_time_feature(train, result):
    global traincopy
    traincopy = get_data_time_and_distance(train); gc.collect()
    result = get_sloc_hour_fre(train, result); gc.collect()
    result = get_eloc_hour_fre(train, result); gc.collect()
    result = get_sloc_in_user_hourcount(train, result); gc.collect()
    return result
