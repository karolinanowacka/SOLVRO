#SOLVRO KN
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns


#loading training data
X_train=np.load('/Users/karolinanowacka/Downloads/solvro-rekrutacja-zimowa-ml-2022/X_train.npy')
y_train=np.load('/Users/karolinanowacka/Downloads/solvro-rekrutacja-zimowa-ml-2022/y_train.npy')

#loading validation data
X_val=np.load('/Users/karolinanowacka/Downloads/solvro-rekrutacja-zimowa-ml-2022/X_val.npy')
y_val=np.load('/Users/karolinanowacka/Downloads/solvro-rekrutacja-zimowa-ml-2022/y_val.npy')

#loading test data
X_test=np.load('/Users/karolinanowacka/Downloads/solvro-rekrutacja-zimowa-ml-2022/X_test.npy')

#debugging
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}")

#missings check
#observation: there are no missings in the data

#in training
X_train_df=pd.DataFrame(X_train.reshape(X_train.shape[0],-1))
y_train_df=pd.DataFrame(y_train)
print(f"Missings in X_train: {X_train_df.isnull().sum()}")
print(f"Missings in y_train: {y_train_df.isnull().sum()}")
#in validation
X_val_df=pd.DataFrame(X_val.reshape(X_val.shape[0],-1))
y_val_df=pd.DataFrame(y_val)
print(f"Misings in X_val: {X_val_df.isnull().sum()}")
print(f"Missings in y_val: {y_val_df.isnull().sum()}")
#in test
X_test_df=pd.DataFrame(X_test.reshape(X_test.shape[0],-1))
print(f"Missings in X_test: {X_test_df.isnull().sum()}")


#visualization
#plt.figure(figsize=(12,6))
#sns.histplot(X_train.flatten(), bins=50, kde=True)
#plt.title('feature distribution in X_train')
#plt.xlabel('feature value')
#plt.ylabel('frequency')
#plt.show()

#conversion of labels
y_train_indices=np.argmax(y_train, axis=1)
y_val_indices=np.argmax(y_val, axis=1)
