import sys
sys.path.append("..")
from config import *

def diff_of_minutes(time1, time2):
    d = {'5': 0, '6': 31, }
    try:
        days = (d[time1[6]] + int(time1[8:10])) - (d[time2[6]] + int(time2[8:10]))
        try:
            minutes1 = int(time1[11:13]) * 60 + int(time1[14:16])
        except:
            minutes1 = 0
        try:
            minutes2 = int(time2[11:13]) * 60 + int(time2[14:16])
        except:
            minutes2 = 0
        return (days * 1440 - minutes2 + minutes1)
    except:
        return np.nan

def cal_distance(lat1,lon1,lat2,lon2):
    dx = np.abs(lon1 - lon2)
    dy = np.abs(lat1 - lat2)
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx**2 + Ly**2) ** 0.5
    return L

def cal_degree(lat1,lon1,lat2,lon2):
    dx = (lon2-lon1)*math.cos(lat1/57.2958)
    dy = lat2-lat1
    if dx==0 and dy==0:
        degree = -1
    else:
        if dy==0:
            dy = 1.0e-24
        degree = 180-90*(math.copysign(1,dy))-math.atan(dx/dy)*57.2958
    return degree

def rank(data, feat1, feat2, ascending):
    data.sort_values([feat1,feat2],inplace=True,ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1,as_index=False)['rank'].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=feat1,how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

def rank3(data, feat1, feat2, feat3, ascending):
    data.sort_values([feat1,feat2,feat3],inplace=True,ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby([feat1,feat2],as_index=False)['rank'].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=[feat1,feat2],how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

def reshape(prediction):
    result = prediction[["orderid","pred","geohashed_end_loc"]].copy()
    result = rank(result,'orderid','pred',ascending=False)
    result = result[result['rank']<3][['orderid','geohashed_end_loc','rank']]
    result = result.set_index(['orderid','rank']).unstack()
    result.reset_index(inplace=True)
    result['orderid'] = result['orderid'].astype('int')
    result.columns = ['orderid', 0, 1, 2]
    return result

def map_score(result):
    result_path = Config.cache_dir + '/true.pkl'
    if os.path.exists(result_path):
        true = pickle.load(open(result_path, 'rb+'))
    else:
        train = pd.read_csv(Config.train_path)
        true = dict(zip(train['orderid'].values,train['geohashed_end_loc']))
        pickle.dump(true,open(result_path, 'wb+'))
    data = result.copy()
    data['true'] = data['orderid'].map(true)
    score = (sum(data['true']==data[0])
             +sum(data['true']==data[1])/2
             +sum(data['true']==data[2])/3)/data.shape[0]
    print(sum(data['true']==data[0])/data.shape[0],sum(data['true']==data[1])/data.shape[0],sum(data['true']==data[2])/data.shape[0],score)
    return score

def get_label(data):
    result_path = Config.cache_dir + '/true.pkl'
    if os.path.exists(result_path):
        true = pickle.load(open(result_path, 'rb+'))
    else:
        train = pd.read_csv(Config.train_path)
        test = pd.read_csv(Config.test_path)
        test['geohashed_end_loc'] = np.nan
        datapre = pd.concat([train,test])
        true = dict(zip(datapre['orderid'].values, datapre['geohashed_end_loc']))
        pickle.dump(true, open(result_path, 'wb+'))
    data['label'] = data['orderid'].map(true)
    data['label'] = (data['label'] == data['geohashed_end_loc']).astype('int')
    return data