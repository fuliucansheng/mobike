import sys
sys.path.append("..")
from config import *
from utils.other import *
from utils.distance import *

traincopy = None
# 获取当前起始地点的极大极小平均的出行距离
def get_sloc_distance(train,result):
    sloc_dis = traincopy.groupby("geohashed_start_loc",as_index=False)["distance"].agg({"sloc_dis_max":np.max,"sloc_dis_min":np.min,"sloc_dis_med":np.mean})
    result = pd.merge(result,sloc_dis,on="geohashed_start_loc",how="left")
    result.loc[:,"sloc_dis_sub_abs"] = np.abs(result["distance"] - result["sloc_dis_med"])
    result.loc[:,"sloc_dis_sub"] = result["distance"] - result["sloc_dis_med"]
    return result

# 获取当前终止地点的极大极小平均的出行距离
def get_eloc_distance(train,result):
    eloc_dis = traincopy.groupby("geohashed_end_loc",as_index=False)["distance"].agg({"eloc_dis_max":np.max,"eloc_dis_min":np.min,"eloc_dis_med":np.mean})
    result = pd.merge(result,eloc_dis,on="geohashed_end_loc",how="left")
    result.loc[:,"eloc_dis_sub_abs"] = np.abs(result["distance"] - result["eloc_dis_med"])
    result.loc[:,"eloc_dis_sub"] = result["distance"] - result["eloc_dis_med"]
    return result

# 获取当前路线的用户热度
def get_sloc_eloc_usercount(train,result):
    user_count = train.groupby(['geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'sloc_eloc_usercount': lambda x: np.unique(x).size})
    result = pd.merge(result,user_count,on=['geohashed_start_loc','geohashed_end_loc'],how="left")
    return result

# 获取当前折返路线的用户热度
def get_eloc_sloc_usercount(train,result):
    eloc_sloc_usercount = train.groupby(['geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'eloc_sloc_usercount': lambda x: np.unique(x).size})
    eloc_sloc_usercount.rename(columns = {'geohashed_start_loc':'geohashed_end_loc','geohashed_end_loc':'geohashed_start_loc'},inplace=True)
    result = pd.merge(result,eloc_sloc_usercount,on=['geohashed_start_loc','geohashed_end_loc'],how='left')
    return result

# 统计当前终止地点的热度
def get_eloc_count(train,result):
    eloc_count = train.groupby('geohashed_end_loc', as_index=False)['userid'].agg({'eloc_count': 'count'})
    result = pd.merge(result, eloc_count, on='geohashed_end_loc', how='left')
    return result

# 统计当前起始地点的热度
def get_sloc_count(train,result):
    sloc_count = train.groupby('geohashed_start_loc', as_index=False)['userid'].agg({'sloc_count': 'count'})
    result = pd.merge(result, sloc_count, on='geohashed_start_loc', how='left')
    return result


# 统计当前终止地点的用户热度
def get_eloc_usercount(train,result):
    eloc_usercount = train.groupby('geohashed_end_loc',as_index=False)['userid'].agg({'eloc_usercount': lambda x: np.unique(x).size})
    result = pd.merge(result, eloc_usercount,on='geohashed_end_loc',how="left")
    return result

# 统计当前终点作为起始地点的热度
def get_eloc_as_sloc_count(train,result):
    eloc_as_sloc_count = train.groupby('geohashed_start_loc', as_index=False)['userid'].agg({'eloc_as_sloc_count': 'count'})
    eloc_as_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
    result = pd.merge(result, eloc_as_sloc_count, on='geohashed_end_loc', how='left')
    return result

# 统计当前起始地点有多少终止地点
def get_sloc_eloccount(train,result):
    sloc_eloccount = train.groupby('geohashed_start_loc',as_index=False)['geohashed_end_loc'].agg({'sloc_eloccount': lambda x: np.unique(x).size})
    result = pd.merge(result, sloc_eloccount, on='geohashed_start_loc', how='left')
    return result

# 统计当前终止地点有多少起始地点
def get_eloc_sloccount(train,result):
    eloc_sloccount = train.groupby('geohashed_end_loc',as_index=False)['geohashed_start_loc'].agg({'eloc_sloccount': lambda x: np.unique(x).size})
    result = pd.merge(result, eloc_sloccount, on='geohashed_end_loc', how='left')
    return result

def get_sloc_eloc_ordercount(train, result):
    sloc_eloc_ordercount = train.groupby(['geohashed_start_loc', 'geohashed_end_loc'],as_index=False)['orderid'].agg({"sloc_eloc_ordercount":"count"})
    result = pd.merge(result, sloc_eloc_ordercount, on=['geohashed_start_loc', 'geohashed_end_loc'], how="left")
    return result

# 针对当前终点对所有起始点按距离排序
def get_sloc_rate_rank(result):
    result.loc[:,"sloc_rate"] = result["sloc_eloc_ordercount"]/result["eloc_count"]
    result = rank(result,'geohashed_end_loc','sloc_rate',ascending=False)
    result.rename(columns={"rank":"sloc_rank"},inplace=True)
    result = rank(result,'geohashed_end_loc','distance',ascending=False)
    result.rename(columns={"rank":"sloc_dis_rank"},inplace=True)
    return result

# 针对当前起始地点对所有可能的终点按距离排序
def get_eloc_rate_rank(result):
    result.loc[:,"eloc_rate"] = result["sloc_eloc_ordercount"]/result["sloc_count"]
    result = rank(result,'geohashed_start_loc','eloc_rate',ascending=False)
    result.rename(columns={"rank":"eloc_rank"},inplace=True)
    result = rank(result,'geohashed_start_loc','distance',ascending=False)
    result.rename(columns={"rank":"eloc_dis_rank"},inplace=True)
    return result

def get_location_feature(train, result):
    global traincopy
    traincopy = get_data_time_and_distance(train)
    result = get_sloc_distance(train,result); gc.collect()
    result = get_eloc_distance(train,result); gc.collect()
    result = get_sloc_eloc_usercount(train,result); gc.collect()
    result = get_eloc_sloc_usercount(train,result); gc.collect()
    result = get_eloc_count(train,result); gc.collect()
    result = get_sloc_count(train,result); gc.collect()
    result = get_eloc_usercount(train,result); gc.collect()
    result = get_eloc_as_sloc_count(train,result); gc.collect()
    result = get_sloc_eloccount(train,result); gc.collect()
    result = get_eloc_sloccount(train,result); gc.collect()
    result = get_sloc_eloc_ordercount(train, result); gc.collect()
    result = get_sloc_rate_rank(result); gc.collect()
    result = get_eloc_rate_rank(result); gc.collect()
    return result
