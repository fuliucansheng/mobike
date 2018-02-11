import sys
sys.path.append("..")
from config import *
from utils.other import *
from utils.distance import *

traincopy = None
def get_user_hour_fre(train, result):
    user_hour_fre = traincopy.groupby(["userid","hour"],as_index=False)["orderid"].agg({"user_hour_fre":"count"})
    result = pd.merge(result,user_hour_fre,on=["userid","hour"],how="left")
    return result

def get_user_time_feature(train, result):
    global traincopy
    traincopy = get_data_time_and_distance(train); gc.collect()
    result = get_user_hour_fre(train, result); gc.collect()
    return result
