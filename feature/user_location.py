import sys
sys.path.append("..")
from config import *
from utils.other import *
from utils.distance import *

traincopy = None
# 获取当前用户起始地点的极大极小平均的出行距离
def get_user_sloc_distance(train,result):
    user_sloc_dis = traincopy.groupby(["geohashed_start_loc","userid"],as_index=False)["distance"].agg({"user_sloc_dis_max":np.max,"user_sloc_dis_min":np.min,"user_sloc_dis_med":np.mean})
    result = pd.merge(result,user_sloc_dis,on=["geohashed_start_loc","userid"],how="left")
    result.loc[:,"user_sloc_dis_sub_abs"] = np.abs(result["distance"] - result["user_sloc_dis_med"])
    result.loc[:,"user_sloc_dis_sub"] = result["distance"] - result["user_sloc_dis_med"]
    return result

# 获取当前用户终止地点的极大极小平均的出行距离
def get_user_eloc_distance(train,result):
    user_eloc_dis = traincopy.groupby(["geohashed_end_loc","userid"],as_index=False)["distance"].agg({"user_eloc_dis_max":np.max,"user_eloc_dis_min":np.min,"user_eloc_dis_med":np.mean})
    result = pd.merge(result,user_eloc_dis,on=["geohashed_end_loc","userid"],how="left")
    result.loc[:,"user_eloc_dis_sub_abs"] = np.abs(result["distance"] - result["user_eloc_dis_med"])
    result.loc[:,"user_eloc_dis_sub"] = result["distance"] - result["user_eloc_dis_med"]
    return result

# 统计用户去过多少次当前终止地点
def get_user_eloc_count(train,result):
    user_eloc_count = train.groupby(['geohashed_end_loc','userid'], as_index=False)['userid'].agg({'user_eloc_count': 'count'})
    result = pd.merge(result, user_eloc_count, on=['geohashed_end_loc','userid'], how='left')
    return result

# 统计用户去过多少次当前起始地点
def get_user_sloc_count(train,result):
    user_sloc_count = train.groupby(['geohashed_start_loc','userid'], as_index=False)['userid'].agg({'user_sloc_count': 'count'})
    result = pd.merge(result, user_sloc_count, on=['geohashed_start_loc','userid'], how='left')
    return result

# 统计用户把当前终点作为多少次起始地点
def get_user_eloc_as_sloc_count(train,result):
    user_eloc_as_sloc_count = train.groupby(['geohashed_start_loc','userid'], as_index=False)['userid'].agg({'user_eloc_as_sloc_count': 'count'})
    user_eloc_as_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
    result = pd.merge(result, user_eloc_as_sloc_count, on=['geohashed_end_loc','userid'], how='left')
    return result

def get_user_sloc_eloc_ordercount(train, result):
    user_sloc_eloc_ordercount = train.groupby(['userid','geohashed_start_loc', 'geohashed_end_loc'],as_index=False)['orderid'].agg({"user_sloc_eloc_ordercount":"count"})
    result = pd.merge(result, user_sloc_eloc_ordercount, on=['userid','geohashed_start_loc', 'geohashed_end_loc'], how="left")
    return result

# 针对当前终点对所有起始点按距离排序
def get_user_sloc_rate_rank(result):
    result.loc[:,"user_sloc_rate"] = result["user_sloc_eloc_ordercount"]/result["user_eloc_count"]
    result = rank(result,'geohashed_end_loc','user_sloc_rate',ascending=False)
    result.rename(columns={"rank":"user_sloc_rank"},inplace=True)
    return result

# 针对当前起始地点对所有可能的终点按距离排序
def get_user_eloc_rate_rank(result):
    result.loc[:,"user_eloc_rate"] = result["user_sloc_eloc_ordercount"]/result["user_sloc_count"]
    result = rank(result,'geohashed_start_loc','user_eloc_rate',ascending=False)
    result.rename(columns={"rank":"user_eloc_rank"},inplace=True)
    return result

def get_user_location_feature(train, result):
    global traincopy
    traincopy = get_data_time_and_distance(train)
    result = get_user_sloc_distance(train,result); gc.collect()
    result = get_user_eloc_distance(train,result); gc.collect()
    result = get_user_eloc_count(train,result); gc.collect()
    result = get_user_sloc_count(train,result); gc.collect()
    result = get_user_eloc_as_sloc_count(train,result); gc.collect()
    result = get_user_sloc_eloc_ordercount(train, result); gc.collect()
    result = get_user_sloc_rate_rank(result); gc.collect()
    result = get_user_eloc_rate_rank(result); gc.collect()
    return result
