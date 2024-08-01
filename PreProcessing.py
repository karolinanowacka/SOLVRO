import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_data():
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
    print(f"unique labels in y_train: {np.unique(y_train)}")

    return X_train, y_train.astype(int), X_val, y_val.astype(int), X_test

def check_missing_values(data):
    if np.isnan(data).any():
        raise ValueError("there are missing values in dataset")
    print(f"missing values {data}: {np.isnan(data).sum()}")

def remove_outliers(X_train, y_train, threshold=3):
    mean = np.mean(X_train, axis = 0)
    std_dev = np.std(X_train, axis = 0)

    std_dev = np.where(std_dev == 0, 1, std_dev)
    
    z_scores = (X_train - mean) / std_dev
    mask = np.abs(z_scores) < threshold
    mask = mask.all(axis = 2).all(axis = 1)
    return X_train[mask], y_train[mask]

def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_train = scaler.fit_transform(X_train).reshape(X_train.shape[0], 300, 2)
    X_val = scaler.transform(X_val).reshape(X_val.shape[0], 300, 2)
    X_test = scaler.transform(X_test).reshape(X_test.shape[0], 300, 2)
    return X_train, X_val, X_test, scaler


def check_duplicates(data):
    data_reshaped = data.reshape(data.shape[0], -1)
    data_tuples = [tuple(row) for row in data_reshaped]
    seen = set()
    for item in data_tuples:
        if item in seen:
            return True
        seen.add(item)
    return False

def check_overlapping(data):
    start_points = data[:, 0, :]
    end_points = data[:, -1, :]
    seen_start_points = set()
    seen_end_points = set()

    for i in range(len(start_points)):
        start_point_tuple = tuple(start_points[i])
        end_point_tuple = tuple(end_points[i])
        if start_point_tuple in seen_start_points and end_point_tuple in seen_end_points:
            return True
        seen_start_points.add(start_point_tuple)
        seen_end_points.add(end_point_tuple)
    return False

def plot_trajectories(X, num_samples):
    for i in range(num_samples):
        plt.figure(figsize = (10,10))
        x_coords = X[i, :, 0]
        y_coords = X[i, :, 1]

        plt.plot(x_coords, y_coords, color = 'blue', label = f'Sample {i}')
        plt.scatter(x_coords[0], y_coords[0], color = 'green', s = 50, label = 'Start', zorder = 5)
        plt.scatter(x_coords[-1], y_coords[-1], color = 'red', s = 50, label = 'End', zorder = 5)
       
        plt.title('Particle Trajectory')
        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        plt.legend()
        plt.show()