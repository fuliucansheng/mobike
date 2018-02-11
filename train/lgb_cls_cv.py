import sys
sys.path.append("..")
from config import *
from data_maker import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import lightgbm as lgb

train = pd.read_csv(Config.train_path)
train1 = train[(train['starttime'] < '2017-05-23 00:00:00')]
train2 = train[(train['starttime'] >= '2017-05-23 00:00:00')]
train2['geohashed_end_loc'] = np.nan

train_feat = make_train_set(train1,train2,sub=False)

N = 5
featfilter = ['orderid','starttime','date','day','dayofweek','userid','bikeid','biketype',
'geohashed_start_loc','geohashed_end_loc','label','user_most_freq_eloc', 'sloc_most_freq_eloc']
predictors = [i for i in train_feat.columns if i not in featfilter]
feature_num = len(predictors)
logging.info(feature_num)

params = {
    "boosting":"gbdt",
    "num_leaves": 96,
    "objective": "binary",
    "learning_rate": 0.05,
    "feature_fraction": 0.886,
    "bagging_fraction": 0.886,
    "bagging_freq": 5,
    "is_unbalance": True,
    'min_data_in_leaf': 200,
    "metric": ["auc","binary_logloss"]
}
num_round = 3600

gc.collect()
kf = KFold(n_splits=N,shuffle=True)
for i,(train_index,test_index) in enumerate(kf.split(train_feat)):
    x = train_feat.loc[train_index,:]
    y = train_feat.loc[test_index,:]
    x = lgb.Dataset(x[predictors],label = x['label']); gc.collect()
    y = lgb.Dataset(y[predictors],label = y['label']); gc.collect()
    gbm = lgb.train(params, x, num_round, early_stopping_rounds=20, valid_sets=[x,y]); gc.collect()
    train_feat['pred'] = gbm.predict(train_feat[gbm.feature_name()]); gc.collect()
    result = reshape(train_feat); gc.collect()
    score = map_score(result)
    print(N,i,score)
    gbm.save_model(Config.cache_dir + "/lgb_cv_%s_%s_%s_%s.txt"%(N,i,feature_num,score)); gc.collect()

