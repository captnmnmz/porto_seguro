from numpy import *
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from feature_selection import feature_selection

def read_dataset(size_training):
    
    full_train = pd.read_csv("../../input/train.csv")
    selected_train = feature_selection(full_train)
    zero_train = selected_train.loc[selected_train['target'] == 0].values
    one_train = selected_train.loc[selected_train['target'] == 1].values
    
    zero_features_train = zero_train[:,2:]
    zero_targets_train = zero_train[:,1]
    one_features_train = one_train[:,2:]
    one_targets_train = one_train[:,1]

    random_training_zero = list(range(zero_targets_train.shape[0]))
    random.shuffle(random_training_zero)
    zero_features_train = zero_features_train[random_training_zero[0:size_training],:]
    zero_targets_train = zero_targets_train[random_training_zero[0:size_training]]

    features_train = concatenate((zero_features_train,one_features_train), axis=0).astype(float64)
    targets_train = concatenate((zero_targets_train,one_targets_train), axis=0).astype(int)

    full_test = pd.read_csv("../../input/test.csv")
    test = feature_selection(full_test).values
    features_test = test[:,1:].astype(float64)

    return features_train, targets_train, features_test


def read_dataset_acc(size_training, size_testing):
    
    full_train = pd.read_csv("../../input/train.csv")
    selected_train = feature_selection(full_train)
    zero_train = selected_train.loc[selected_train['target'] == 0].values
    one_train = selected_train.loc[selected_train['target'] == 1].values    

    zero_features_test = zero_train[:size_testing,2:].astype(float64)
    zero_targets_test = zero_train[:size_testing,1].astype(int)
    one_features_test = one_train[:size_testing,2:].astype(float64)
    one_targets_test = one_train[:size_testing,1].astype(int)

    zero_features_train = zero_train[size_testing:,2:]
    zero_targets_train = zero_train[size_testing:,1]
    one_features_train = one_train[size_testing:,2:]
    one_targets_train = one_train[size_testing:,1]

    random_training_zero = list(range(zero_targets_train.shape[0]))
    random.shuffle(random_training_zero)
    zero_features_train = zero_features_train[random_training_zero[0:size_training],:]
    zero_targets_train = zero_targets_train[random_training_zero[0:size_training]]

    features_train = concatenate((zero_features_train,one_features_train), axis=0).astype(float64)
    targets_train = concatenate((zero_targets_train,one_targets_train), axis=0).astype(int)

    return features_train, targets_train, zero_features_test, zero_targets_test, one_features_test, one_targets_test
