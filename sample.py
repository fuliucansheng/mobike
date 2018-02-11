from config import *

# 获取当前的用户的所有结束地点
def get_user_end_loc(train, test):
    result_path = Config.cache_dir + '/user_end_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & Config.flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_eloc = train[['userid','geohashed_end_loc']].drop_duplicates()
        result = pd.merge(test[['orderid','userid']],user_eloc,on='userid',how='left')
        result = result[['orderid', 'geohashed_end_loc']]
        if Config.dump: result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取当前的用户的所有起始地点
def get_user_start_loc(train, test):
    result_path = Config.cache_dir + '/user_start_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & Config.flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        trainall = pd.read_csv(Config.train_path)
        user_sloc = trainall[['userid', 'geohashed_start_loc']].drop_duplicates()
        sub = pd.read_csv(Config.test_path)
        user_sloc_sub = sub[['userid','geohashed_start_loc']].drop_duplicates()
        user_sloc = pd.concat([user_sloc,user_sloc_sub]).drop_duplicates()
        result = pd.merge(test[['orderid', 'userid']], user_sloc, on='userid', how='left')
        result.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
        result = result[['orderid', 'geohashed_end_loc']]
        if Config.dump: result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取训练集中所有的起始--终止地点对
def get_loc_to_loc(train, test):
    result_path = Config.cache_dir + '/user_loc_to_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & Config.flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        sloc_eloc_count = train.groupby(['geohashed_start_loc', 'geohashed_end_loc'],as_index=False)['geohashed_end_loc'].agg({'sloc_eloc_count':'count'})
        sloc_eloc_count.sort_values('sloc_eloc_count',inplace=True)
        sloc_eloc = sloc_eloc_count.groupby('geohashed_start_loc').tail(20)
        result = pd.merge(test[['orderid', 'geohashed_start_loc']], sloc_eloc, on='geohashed_start_loc', how='left')
        result = result[['orderid', 'geohashed_end_loc']]
        if Config.dump: result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

def get_bike_start_loc(train, test):
    train_set = pd.read_csv(Config.train_path)
    test_set = pd.read_csv(Config.test_path)
    all_set = pd.concat([train_set, test_set])
    bike_sloc = all_set[['orderid', 'bikeid', 'geohashed_start_loc', 'starttime']]
    bike_sloc.sort_values(by=['bikeid', 'starttime'], inplace=True, ascending=True)
    bike_sloc['next_bikeid'] = bike_sloc['bikeid'].shift(-1)
    bike_sloc['geohashed_end_loc'] = bike_sloc['geohashed_start_loc'].shift(-1)
    result = bike_sloc[(bike_sloc['bikeid'] == bike_sloc['next_bikeid']) & (bike_sloc['starttime'] >= test['starttime'].min()) & (bike_sloc['starttime'] <= test['starttime'].max())]
    result = result[['orderid', 'geohashed_end_loc']].drop_duplicates()
    result["bike_start_loc"] = 1
    return result

# 构造终止地点的候选列表
def get_sample(train, test, user=True):
    result_path = Config.cache_dir + '/sample_%d.hdf' % (train.shape[0] * test.shape[0])
    if os.path.exists(result_path) & Config.flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        loc_to_loc = get_loc_to_loc(train, test)
        bike_sloc = get_bike_start_loc(train, test)
        if user:
            user_end_loc = get_user_end_loc(train, test)
            user_start_loc = get_user_start_loc(train, test)
            result = pd.concat([
                user_end_loc[['orderid','geohashed_end_loc']],
                user_start_loc[['orderid', 'geohashed_end_loc']],
                loc_to_loc[['orderid', 'geohashed_end_loc']],
                bike_sloc[['orderid', 'geohashed_end_loc']]
            ]).drop_duplicates()
        else:
            result = pd.concat([loc_to_loc[['orderid', 'geohashed_end_loc']],bike_sloc[['orderid', 'geohashed_end_loc']]]).drop_duplicates()

        del test["geohashed_end_loc"]
        result = pd.merge(result, test, on="orderid", how="left")
        result = result[result['geohashed_end_loc'] != result['geohashed_start_loc']]
        result = result[(~result['geohashed_end_loc'].isnull()) & (~result['geohashed_start_loc'].isnull())]
        if Config.dump: result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result