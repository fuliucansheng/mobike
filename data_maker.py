from config import *
from sample import *
from utils.distance import *
from feature.user import *
from feature.location import *
from feature.time import *
from feature.user_time import *
from feature.user_location import *
from feature.location_time import *
from feature.user_location_time import *
from feature.other import *
from feature.last import *
from feature.add import *

def make_train_set(train, test, sub=False):
    result_path = Config.cache_dir + '/train_set_%d.hdf' % (train.shape[0] * test.shape[0])
    if os.path.exists(result_path) & Config.flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        logging.info("make_train_set_begin !")
        result = get_sample(train,test)
        logging.info("get_sample done ! " + str(result.shape))
        result = get_data_time_and_distance(result)
        result = get_data_manhattan_distance(result)
        result = result[result.distance<8000]
        logging.info("filter_sample done ! " + str(result.shape))
        logging.info("get_user_feature done ! " + str(result.shape))
        result = get_user_feature(train, result)
        logging.info("get_location_feature done !" + str(result.shape))
        result = get_location_feature(train, result)
        logging.info("get_time done! " + str(result.shape))
        result = get_time_feature(train, result)
        logging.info("get_user_location_feature done ! " + str(result.shape))
        result = get_user_location_feature(train, result)
        logging.info("get_location_time_feature done ! " + str(result.shape))
        result = get_location_time_feature(train, result)
        logging.info("get_user_location_time_feature done !" + str(result.shape))
        result = get_user_location_time_feature(train, result)
        logging.info("get_user_time_feature done !" + str(result.shape))
        result = get_user_time_feature(train, result)
        logging.info("get_other_feature done ! " + str(result.shape))
        result = get_other_feature(train, result)
        logging.info("get_last_feature done! " + str(result.shape))
        result = get_last_feature(train, result, test)
        if not sub: result = get_label(result)
        if Config.dump: result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    result = get_add_feature(train, result, test)
    if Config.dump: result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result
