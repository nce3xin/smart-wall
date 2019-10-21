import  numpy as np
import pandas as pd

X_pos=pd.read_csv('data/all_pos.csv')
X_neg=pd.read_csv('data/all_neg.csv')

print(X_pos.shape)
print(X_neg.shape)