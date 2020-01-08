import sys, argparse, pickle, os
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
    file = open('./name-models/logistic_regression.pkl', 'rb')
    pipe = pickle.load(file)
    arr = ['adam', 'hawa']
    for str in arr:
        result = pipe.predict_proba([str])        
        print(str, "female", result[0][0]*100, "male", result[0][1]*100)