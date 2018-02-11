import sys
sys.path.append("..")
from config import *
from utils.other import *
from utils.distance import *

def get_other_feature(train, result):
    result = get_degree(result)
    loc_lat_lon = get_loc_lat_lon()
    loc_lat_lon.rename(columns={"loc":"geohashed_start_loc","lat":"sloc_lat","lon":"sloc_lon"},inplace=True)
    result = pd.merge(result, loc_lat_lon, on="geohashed_start_loc", how="left")
    loc_lat_lon.rename(columns={"geohashed_start_loc":"geohashed_end_loc","sloc_lat":"eloc_lat","sloc_lon":"eloc_lon"},inplace=True)
    result = pd.merge(result, loc_lat_lon, on="geohashed_end_loc", how="left")
    result["sloc_eloc_lat_sub"] = result["eloc_lat"] - result["sloc_lat"]
    result["sloc_eloc_lon_sub"] = result["eloc_lon"] - result["eloc_lon"]
    result["lon_lat_rate"] = result["sloc_eloc_lon_sub"]/result["sloc_eloc_lat_sub"]
    result["lon_sub_dis_rate"] = result["sloc_eloc_lon_sub"]/result["distance"]
    result["lat_sub_dis_rate"] = result["sloc_eloc_lat_sub"]/result["distance"]

    result["hoursub"] = result["hour"] - result["eloc_hourmean"]
    result["hoursub_abs"] = np.abs(result["hour"] - result["eloc_hourmean"])

    result["eloc_hour_count_rate"] = result["eloc_hour_fre"]/result["eloc_count"]
    result["hourrate"] = result["hour_fre"]/train.shape[0]
    result["elocrate"] = result["eloc_count"]/train.shape[0]
    result["eloc_hour_dis"] = result["eloc_hour_count_rate"]*result["elocrate"]/result["hourrate"]

    result["user_eloc_hour_count_rate"] = result["user_eloc_hour_fre"]/result["user_eloc_count"]
    result["user_hourrate"] = result["user_hour_fre"]/train.shape[0]
    result["user_elocrate"] = result["user_eloc_count"]/train.shape[0]
    result["user_eloc_hour_dis"] = result["user_eloc_hour_count_rate"]*result["user_elocrate"]/result["user_hourrate"]
    result["rule_dis_elocfeat"] = result["eloc_usercount"]/(0.01*result["distance"])
    return result
