from numpy import *
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def feature_selection(data_train):

	del_features = ['ps_car_03_cat', 'ps_car_05_cat', 'ps_reg_03']
	
	return data_train.drop(del_features, 1)