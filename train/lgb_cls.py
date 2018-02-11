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

featfilter = ['orderid','starttime','date','day','dayofweek','userid','bikeid','biketype',
'geohashed_start_loc','geohashed_end_loc','label','user_most_freq_eloc', 'sloc_most_freq_eloc']
predictors = [i for i in train_feat.columns if i not in featfilter]
feature_num = len(predictors)
logging.info(feature_num)
x,y = train_test_split(train_feat,test_size=0.1,random_state=0)

params = {
    "boosting":"gbdt",
    "num_leaves": 100,
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

x = lgb.Dataset(x[predictors],label = x['label'])
y = lgb.Dataset(y[predictors],label = y['label'])

gbm = lgb.train(params, x, num_round, early_stopping_rounds=20, valid_sets=[x,y]); gc.collect()
train_feat.loc[:,'pred'] = gbm.predict(train_feat[gbm.feature_name()]); gc.collect()
result = reshape(train_feat)
score = map_score(result)
print(score)
gbm.save_model(Config.cache_dir + "/lgb_%s_%s.txt"%(feature_num,score))
