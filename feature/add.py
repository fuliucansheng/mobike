import sys
sys.path.append("..")
from config import *
from utils.other import *
from utils.distance import *

traincopy = None

def get_degree_dis(train, result):
    sloc_degree = traincopy.groupby('geohashed_start_loc',as_index=False)["degree"].agg({"sloc_degree_mean":np.mean})
    result = pd.merge(result, sloc_degree, on="geohashed_start_loc", how="left")
    result["degree_sub_abs"] = np.abs(result["degree"] - result["sloc_degree_mean"])
    result["degree_sub"] = result["degree"] - result["sloc_degree_mean"]
    return result

def get_user_next_order_dis(result, test):
    testcopy = test.copy()
    testcopy.sort_values(["userid","starttime"],inplace=True,ascending=True)
    testcopy["user_next_sloc"] = testcopy.geohashed_start_loc.shift(-1)
    testcopy["next_userid"] = testcopy.userid.shift(-1)
    result = pd.merge(result, testcopy[["orderid","user_next_sloc","next_userid"]], on="orderid", how="left")
    result = get_distance(result, "geohashed_end_loc", "user_next_sloc", "user_next_order_sloc_dis")
    result.loc[result.userid!=result.next_userid, 'user_next_order_sloc_dis'] = -1000000
    del result["user_next_sloc"],result["next_userid"]
    return result

def get_bike_next_order_dis(result, test):
    testcopy = test.copy()
    testcopy.sort_values(["bikeid","starttime"],inplace=True,ascending=True)
    testcopy["bike_next_sloc"] = testcopy.geohashed_start_loc.shift(-1)
    testcopy["next_bikeid"] = testcopy.bikeid.shift(-1)
    result = pd.merge(result, testcopy[["orderid","bike_next_sloc","next_bikeid"]], on="orderid", how="left")
    result = get_distance(result, "geohashed_end_loc", "bike_next_sloc", "bike_next_order_sloc_dis")
    result.loc[result.bikeid!=result.next_bikeid, 'bike_next_order_sloc_dis'] = -1000000
    del result["bike_next_sloc"],result["next_bikeid"]
    return result

def get_diff(result):
    def cal_diff(s,t):
        abc = [str(i) for i in range(10)]+list(string.ascii_lowercase)
        dic = {i:c for c,i in enumerate(abc)}
        s,t = sorted([s,t])
        c = 0
        diff = 0
        for i,j in zip(s,t):
            if i!=j:
            	diff += len(abc)**(len(s)-c-1)*abs(dic[i]-dic[j])
            c+=1
        return diff
    loc_dict = get_lon_lat_lon_dict()
    geohashed_loc = result[['geohashed_start_loc','geohashed_end_loc']].values
    diffs = []
    for i in geohashed_loc:
        if i[0] in loc_dict and i[1] in loc_dict:
            diffs.append(cal_diff(i[0],i[1]))
        else:
            diffs.append(-999)
    result.loc[:,'diff'] = diffs
    return result

def get_add_feature(train, result, test):
    global traincopy
    #traincopy = get_data_time_and_distance(train)
    #traincopy = get_degree(train)
    #result = get_degree_dis(train, result)
    #result = get_user_next_order_dis(result, test)
    #result = get_bike_next_order_dis(result, test)
    #result["minute"] = pd.DatetimeIndex(result.starttime).minute
    #result = get_diff(result);
    return result
