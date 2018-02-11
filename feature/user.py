import sys
sys.path.append("..")
from config import *
from utils.other import *
from utils.distance import *

traincopy = None
# 获取用户出行的距离的极大极小均值
def get_user_distance(train,result):
    user_dis = traincopy.groupby("userid",as_index=False)["distance"].agg({"user_dis_max":np.max,"user_dis_min":np.min,"user_dis_med":np.mean})
    result = pd.merge(result, user_dis, on="userid", how="left")
    return result

# 统计用户的出行频率
def get_user_count(train,result):
    user_count = train.groupby('userid',as_index=False)['geohashed_start_loc'].agg({'user_count':'count'})
    result = pd.merge(result,user_count,on=['userid'],how='left')
    return result


def get_user_feature(train, result):
    global traincopy
    traincopy = get_data_time_and_distance(train); gc.collect()
    result = get_user_distance(train,result); gc.collect()
    result = get_user_count(train,result); gc.collect()
    return result
