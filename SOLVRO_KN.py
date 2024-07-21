#SOLVRO KN
import numpy as np
import pandas as pd
import torch as trch
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

#conversion of labels
y_train_indices=np.argmax(y_train, axis=1)
y_val_indices=np.argmax(y_val, axis=1)

#missings check
#training
X_train_df=pd.DataFrame(X_train.reshape(X_train.shape[0],-1))
y_train_df=pd.DataFrame(y_train_indices)
print(f"Missings in X_train: {X_train_df.isnull().sum()}")
print(f"Missings in y_train: {y_train_df.isnull().sum()}")
#validation
X_val_df=pd.DataFrame(X_val.reshape(X_val.shape[0],-1))
y_val_df=pd.DataFrame(y_val_indices)
print(f"Misings in X_val: {X_val_df.isnull().sum()}")
print(f"Missings in y_val: {y_val_df.isnull().sum()}")
#test
X_test_df=pd.DataFrame(X_test.reshape(X_test.shape[0],-1))
print(f"Missings in X_test: {X_test_df.isnull().sum()}")
#observation: there are no missings in the data

#EDA
#starting with summary statistics of 1D data 
#training
print(f"y_train statistics: {y_train_df.describe()}")
#validation
print(f"y_val statistics: {y_val_df.describe()}")

#visualization of 3D data (location of particle over time)
def plot_trajectories(X, num_samples):
    for i in range(num_samples):
        plt.figure(figsize=(10,10))
        x_coords = X[i, :, 0]
        y_coords = X[i, :, 1]

        
        plt.plot(x_coords, y_coords, color='blue', label=f'Sample {i}')
        plt.scatter(x_coords[0],y_coords[0], color='green', s=50, label='Start', zorder=5)
        plt.scatter(x_coords[-1],y_coords[-1],color='red',s=50,label='End', zorder=5)
       
        plt.title('particle trajectory')
        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        plt.legend()
        plt.show()

plot_trajectories(X_train, 2)







