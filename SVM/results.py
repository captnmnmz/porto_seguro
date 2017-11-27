from numpy import *
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def results(pred):
    # Takes as input the prediction results.
    # Creates a csv file for the submission on Kaggle.
    
    submission = pd.read_csv("../../input/sample_submission.csv")
    submission['target'] = pred
    submission.to_csv('../../output/sample_submission.csv', index = False)
    return

