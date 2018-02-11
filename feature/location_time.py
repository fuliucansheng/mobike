import sys
sys.path.append("..")
from config import *
from utils.other import *
from utils.distance import *

traincopy = None
def get_sloc_hour_fre(train, result):
    sloc_hour_fre = traincopy.groupby(["hour","geohashed_start_loc"],as_index=False)["orderid"].agg({"sloc_hour_fre":"count"})
    result = pd.merge(result,sloc_hour_fre,on=["hour","geohashed_start_loc"],how="left")
    return result

def get_eloc_hour_fre(train, result):
    eloc_hour_fre = traincopy.groupby(["hour","geohashed_end_loc"],as_index=False)["orderid"].agg({"eloc_hour_fre":"count"})
    result = pd.merge(result,eloc_hour_fre,on=["hour","geohashed_end_loc"],how="left")
    return result

def get_sloc_eloc_hour_fre(train, result, user=True):
    sloc_eloc_hour_fre = traincopy.groupby(["hour","geohashed_end_loc","geohashed_start_loc"],as_index=False)["orderid"].agg({"sloc_eloc_hour_fre":"count"})
    result = pd.merge(result,sloc_eloc_hour_fre,on=["hour","geohashed_end_loc","geohashed_start_loc"],how="left")
    return result

def get_eloc_hour_dis(train, result):
    eloc_hour = traincopy.groupby("geohashed_end_loc",as_index=False)["hour"].agg({"eloc_hourmean":np.mean})
    result = pd.merge(result,eloc_hour,on="geohashed_end_loc",how="left")
    return result

def get_location_time_feature(train, result):
    global traincopy
    traincopy = get_data_time_and_distance(train)
    result = get_sloc_hour_fre(train, result); gc.collect()
    result = get_eloc_hour_fre(train, result); gc.collect()
    result = get_sloc_eloc_hour_fre(train, result); gc.collect()
    result = get_eloc_hour_dis(train, result); gc.collect()
    return result
