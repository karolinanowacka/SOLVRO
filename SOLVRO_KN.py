#SOLVRO KN
import numpy as np
import pandas as pd
import torch as torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


#loading training data
X_train = np.load('/Users/karolinanowacka/Downloads/solvro-rekrutacja-zimowa-ml-2022/X_train.npy')
y_train = np.load('/Users/karolinanowacka/Downloads/solvro-rekrutacja-zimowa-ml-2022/y_train.npy')

#loading validation data
X_val = np.load('/Users/karolinanowacka/Downloads/solvro-rekrutacja-zimowa-ml-2022/X_val.npy')
y_val = np.load('/Users/karolinanowacka/Downloads/solvro-rekrutacja-zimowa-ml-2022/y_val.npy')

#loading test data
X_test = np.load('/Users/karolinanowacka/Downloads/solvro-rekrutacja-zimowa-ml-2022/X_test.npy')

#debugging
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}")

#check for missing values
#training
print(f"Missing values in X_train: {np.isnan(X_train).sum()}")
print(f"Missing values in y_train: {np.isnan(y_train).sum()}")
#validation
print(f"Missing values in X_val: {np.isnan(X_val).sum()}")
print(f"Missing values in y_val: {np.isnan(y_val).sum()}")
#test
print(f"Missing values in X_test: {np.isnan(X_test).sum()}")

#check for duplicates
def check_duplicates_3d(array):
    flattened_array = array.reshape(array.shape[0],-1)
    unique_rows, indices = np.unique(flattened_array, axis = 0, return_index = True )
    num_duplicates = flattened_array.shape[0] - unique_rows.shape[0]
    return num_duplicates

def check_duplicates_2d(array):
    unique_rows, indices = np.unique(array, axis = 0, return_index = True )
    num_duplicates = array.shape[0] - unique_rows.shape[0]
    return num_duplicates

#training
num_duplicates_X_train = check_duplicates_3d(X_train)
#validation
num_duplicates_X_val = check_duplicates_3d(X_val)
#test
num_duplicates_X_test = check_duplicates_3d(X_test)

#debugging
print(f"duplicates in training data: {num_duplicates_X_train}")
print(f"duplicates in validation data: {num_duplicates_X_val}")
print(f"duplicates in test data: {num_duplicates_X_test}")

#checking scales and units
#separating coordinates
X_coords = X_train[:, :, 0]
y_coords = X_train[:, :, 1]

#statistics for x
X_min=np.min(X_coords)
X_max=np.max(X_coords)
X_mean=np.mean(X_coords)
X_std=np.std(X_coords)

#statistics for y
y_min=np.min(y_coords)
y_max=np.max(y_coords)
y_mean=np.mean(y_coords)
y_std=np.std(y_coords)

#debugging
print(f"X coords statistics: min: {X_min}, max: {X_max}, mean: {X_mean}, std: {X_std}")
print(f"y coords statistics: min: {y_min}, max: {y_max}, mean: {y_mean}, std: {y_std}")
#observation: data has the same scales and units


#outlier detection using Z-score and original features (x,y coordinates)
X_coords_reshaped = X_coords.reshape(X_coords.shape[0],-1)
y_coords_reshaped = y_coords.reshape(y_coords.shape[0],-1)
data = np.column_stack((X_coords_reshaped, y_coords_reshaped))
data_df = pd.DataFrame(data)

z_scores = np.abs( (data_df-data_df.mean()) / data_df.std() )
outliers = (z_scores > 3).any(axis = 1)

#debugging
print(f"amount of outliers: {outliers.sum()}")
#detected 84 outliers

#removing outliers
X_train_cleaned = X_train[~outliers]
y_train_cleaned = y_train[~outliers]

#debugging
print(f"X_train shape: {X_train_cleaned.shape}")
print(f"y_train shape: {y_train_cleaned.shape}")

#feature scaling
scaler = StandardScaler()
X_train_cleaned_reshaped = X_train_cleaned.reshape(X_train_cleaned.shape[0], -1)
X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
scaler.fit(X_train_cleaned_reshaped)
X_train_cleaned_normalized = scaler.transform(X_train_cleaned_reshaped).reshape(X_train_cleaned.shape)
X_val_normalized = scaler.transform(X_val_reshaped).reshape(X_val.shape)

#debugging
print(f"X_train_cleaned_normalized shape: {X_train_cleaned_normalized.shape}")
print(f"X_val_normalized shape: {X_val_normalized.shape}")

#check for overlapping
total_samples = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
train_idx = set(range(X_train.shape[0]))
val_idx = set(range(X_train.shape[0], X_train.shape[0]+X_val.shape[0]))
test_idx = set(range(X_train.shape[0] + X_val.shape[0], total_samples))

#debugging
assert len(train_idx.intersection(val_idx)) == 0, "found overalapping elements"
assert len(train_idx.intersection(test_idx)) == 0, "found overalapping elements"
assert len(val_idx.intersection(test_idx)) == 0, "found overalapping elements"



#visualization of 3D data (location of particle over time)
def plot_trajectories(X, num_samples):
    for i in range(num_samples):
        plt.figure(figsize = (10,10))
        x_coords = X[i, :, 0]
        y_coords = X[i, :, 1]

        
        plt.plot(x_coords, y_coords, color = 'blue', label = f'Sample {i}')
        plt.scatter(x_coords[0],y_coords[0], color = 'green', s = 50, label = 'Start', zorder = 5)
        plt.scatter(x_coords[-1],y_coords[-1],color = 'red',s = 50,label = 'End', zorder = 5)
       
        plt.title('particle trajectory')
        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        plt.legend()
        plt.show()

plot_trajectories(X_train, 1)

#conversion of labels
y_train_indices = np.argmax(y_train, axis=1)
y_val_indices = np.argmax(y_val, axis=1)


#Data Pipeline
class TrajectoryDataset(Dataset):
    def __init__(self, data, labels, scaler = None):
        self.data = data
        self.labels = labels
        self.scaler = scaler

        if self.scaler is not None:
            data_reshaped = self.data.reshape(self.data.shape[0], -1)
            self.data = self.scaler.transform(data_reshaped).reshape(self.data.shape)

    def __len__(self):
        return len(self.data)
    
    def __getitem(self, idx):
        return torch.tensor(self.data[idx], dtype = torch.float32), torch.tensor(self.labels[idx], dtype = torch.long)
    
X_train_cleaned_reshaped = X_train_cleaned.reshape(X_train_cleaned.shape[0], -1)
X_val_cleaned_reshaped = X_val.reshape(X_val.shape[0], -1)

scaler = StandardScaler()
scaler.fit(X_train_cleaned_reshaped)

train_dataset = TrajectoryDataset(X_train_cleaned, y_train, scaler = scaler)
val_dataset = TrajectoryDataset(X_val, y_val, scaler = scaler)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 4)
val_loader = DataLoader(val_dataset, batch_size = 64, shuffle = False, num_workers = 4)

#model
class TrajectoryModel(pl.LightningModule):
    def __init__(self):
        super(TrajectoryModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels = 2, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = torch.nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.fc1 = torch.nn.Linear(64*300, 128)
        self.fc2 = torch.nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = F.cross_entropy(outputs, labels)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = F.cross_entropy(outputs, labels)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)
        return optimizer

#initialization
model = TrajectoryModel()
trainer = pl.Trainer(max_epochs = 10)
trainer.fit(model, train_loader, val_loader)









############expired code
#missings check 
#training
#X_train_df=pd.DataFrame(X_train.reshape(X_train.shape[0],-1))
#y_train_df=pd.DataFrame(y_train_indices)
#print(f"Missings in X_train: {X_train_df.isnull().sum()}")
#print(f"Missings in y_train: {y_train_df.isnull().sum()}")
#validation
#X_val_df=pd.DataFrame(X_val.reshape(X_val.shape[0],-1))
#y_val_df=pd.DataFrame(y_val_indices)
#print(f"Misings in X_val: {X_val_df.isnull().sum()}")
#print(f"Missings in y_val: {y_val_df.isnull().sum()}")
#test
#X_test_df=pd.DataFrame(X_test.reshape(X_test.shape[0],-1))
#print(f"Missings in X_test: {X_test_df.isnull().sum()}")
#observation: there are no missings in the data



#EDA
#starting with summary statistics of 1D data 
#training
#print(f"y_train statistics: {X_train.describe()}")
#validation
#print(f"y_val statistics: {X_val.describe()}")
