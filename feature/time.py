import sys
sys.path.append("..")
from config import *
from utils.other import *
from utils.distance import *

traincopy = None

# 统计每个小时的出行频率
def get_hour_fre(train, result):
    hour_fre = traincopy.groupby("hour",as_index=False)["orderid"].agg({"hour_fre":"count"})
    result = pd.merge(result,hour_fre,on="hour",how="left")
    return result

def get_time_feature(train, result):
    global traincopy
    traincopy = get_data_time_and_distance(train); gc.collect()
    result = get_hour_fre(train, result); gc.collect()
    return result
