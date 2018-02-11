import sys
sys.path.append("..")
from config import *
from utils.other import *
from utils.distance import *

traincopy = None
def get_user_most_frequent_eloc(train, result):
    user_eloc_count = train.groupby(["userid","geohashed_end_loc"],as_index=False)["userid"].agg({"user_eloc_count":"count"})
    user_eloc_count.sort_values(["userid","user_eloc_count"],ascending=True,inplace=True)
    user_most_freq_eloc = user_eloc_count.groupby("userid",as_index=False).last()[["userid","geohashed_end_loc"]]
    user_most_freq_eloc.rename(columns={"geohashed_end_loc":"user_most_freq_eloc"},inplace=True)
    result = pd.merge(result, user_most_freq_eloc, on="userid", how="left")
    result = get_distance(result, "geohashed_start_loc", "user_most_freq_eloc", "user_most_freq_eloc_dis")
    result = get_manhattan_distance(result, "geohashed_start_loc", "user_most_freq_eloc", "user_most_freq_eloc_manhattan_dis")
    return result

def get_sloc_most_frequent_eloc(train, result):
    sloc_eloc_count = train.groupby(["geohashed_start_loc","geohashed_end_loc"],as_index=False)["userid"].agg({"sloc_eloc_count":"count"})
    sloc_eloc_count.sort_values(["geohashed_start_loc","sloc_eloc_count"],ascending=True,inplace=True)
    sloc_most_freq_eloc = sloc_eloc_count.groupby("geohashed_start_loc",as_index=False).last()[["geohashed_start_loc","geohashed_end_loc"]]
    sloc_most_freq_eloc.rename(columns={"geohashed_end_loc":"sloc_most_freq_eloc"},inplace=True)
    result = pd.merge(result, sloc_most_freq_eloc, on="geohashed_start_loc", how="left")
    result = get_distance(result, "geohashed_start_loc", "sloc_most_freq_eloc", "sloc_most_freq_eloc_dis")
    result = get_manhattan_distance(result, "geohashed_start_loc", "sloc_most_freq_eloc", "sloc_most_freq_eloc_manhattan_dis")
    return result

def get_user_regular_bike(train, result):
    user_eloc_sloc_count = train.groupby(['userid','geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'user_eloc_sloc_count':'count'})
    user_eloc_sloc_count.rename(columns = {'geohashed_start_loc':'geohashed_end_loc','geohashed_end_loc':'geohashed_start_loc'},inplace=True)
    traincopy = train.copy()
    traincopy = pd.merge(traincopy,user_eloc_sloc_count,on=['userid','geohashed_start_loc','geohashed_end_loc'],how="left")
    user_regular_bike = traincopy.groupby("userid",as_index=False)["user_eloc_sloc_count"].agg({"user_regular_bike":lambda x: np.sum(x>0)/np.size(x)})
    result = pd.merge(result,user_regular_bike,on='userid',how='left')
    return result

def get_user_distance_quantile(train, result):
    user_dis_q2 = traincopy.groupby("userid")['distance'].quantile(0.2).reset_index()
    user_dis_q2.columns = ["userid", "user_dis_q2"]
    user_dis_q8 = traincopy.groupby("userid")['distance'].quantile(0.8).reset_index()
    user_dis_q8.columns = ["userid", "user_dis_q8"]
    user_dis_q = pd.merge(user_dis_q2, user_dis_q8, on="userid", how="left")
    result = pd.merge(result, user_dis_q, on="userid", how="left")
    result["user_dis_q2"] = result["user_dis_q2"] - result["distance"]
    result["user_dis_q8"] = result["user_dis_q8"] - result["distance"]
    return result

def get_last_order_diff(result, test):
    testcopy = test.copy()
    testcopy.sort_values(["userid","starttime"],inplace=True,ascending=True)
    testcopy["user_last_order_time_diff"] = (pd.DatetimeIndex(testcopy.starttime)-\
                                             pd.DatetimeIndex(testcopy.starttime.shift(1))).total_seconds()
    result = pd.merge(result, testcopy[["orderid","user_last_order_time_diff"]], on="orderid", how="left")
    return result

def get_eloc_in_user_hourcount(train, result):
    sloc_in_user_hourcount = traincopy.groupby(["userid","geohashed_end_loc"],as_index=False)["hour"].agg({"eloc_in_user_hourcount":lambda x: np.size(np.unique(x))})
    result = pd.merge(result,sloc_in_user_hourcount,on=["userid","geohashed_end_loc"],how="left")
    return result

def get_sec_sub_feature(train, result):
    train_copy = train.copy()
    train_copy["curr_seconds"] = pd.to_datetime(train_copy.starttime).astype("int")/(1e9)%(3600*24)
    result["curr_seconds"] = pd.to_datetime(result.starttime).astype("int")/(1e9)%(3600*24)
    eloc_sec = train_copy.groupby("geohashed_end_loc",as_index=False)["curr_seconds"].agg({"eloc_secmean":np.mean})
    result = pd.merge(result, eloc_sec, on="geohashed_end_loc", how="left")
    result["seconds_sub"] = result["curr_seconds"] - result["eloc_secmean"]
    result["seconds_sub_abs"] = np.abs(result["curr_seconds"] - result["eloc_secmean"])
    return result

def get_user_eloc_lasttime(train, result):
    train.sort_values("starttime", inplace=True)
    user_eloc_last = train.groupby(["userid","geohashed_end_loc"], as_index=False).last()[["userid","geohashed_end_loc","starttime"]]
    user_eloc_last.rename(columns={"starttime":"eloc_lasttime"},inplace=True)
    result = pd.merge(result, user_eloc_last, on=["userid","geohashed_end_loc"], how="left")
    result["user_eloc_lasttime"] = (pd.DatetimeIndex(result.starttime)-pd.DatetimeIndex(result.eloc_lasttime)).total_seconds()
    del result["eloc_lasttime"]
    return result

def get_last_feature(train, result, test):
    global traincopy
    traincopy = get_data_time_and_distance(train)

    result["user_curr_dis_med_dis_sub_abs"] = np.abs(result["distance"] - result["user_dis_med"])
    result["user_curr_dis_med_dis_sub"] = result["distance"] - result["user_dis_med"]
    result["sloc_curr_dis_med_dis_sub_abs"] = np.abs(result["distance"] - result["sloc_dis_med"])
    result["sloc_curr_dis_med_dis_sub"] = result["distance"] - result["sloc_dis_med"]
    result["eloc_curr_dis_med_dis_sub_abs"] = np.abs(result["distance"] - result["eloc_dis_med"])
    result["eloc_curr_dis_med_dis_sub"] = result["distance"] - result["eloc_dis_med"]
    result = get_user_most_frequent_eloc(train, result)
    result = get_sloc_most_frequent_eloc(train, result)
    result = get_user_regular_bike(train, result)
    result = get_user_distance_quantile(train, result)
    result = get_last_order_diff(result, test)
    result["user_eloc_as_sloc_rate"] = result["user_eloc_as_sloc_count"]/result["user_count"]
    result["user_eloc_count_rate"] = result["user_eloc_count"]/result["user_count"]
    #result = get_eloc_in_user_hourcount(train, result)
    #result["rule_user_eloc_with_in_user_hourcount"] = result["user_eloc_count"]/result["eloc_in_user_hourcount"]

    result = get_sec_sub_feature(train, result)
    result = get_user_eloc_lasttime(train, result)
    return result
