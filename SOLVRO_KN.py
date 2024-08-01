#SOLVRO KN
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from TrajectoryDataset import TrajectoryDataset
from TrajectoryModel import TrajectoryModel
from PreProcessing import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import time
import cProfile

def main():
    print("loading data...")
    X_train, y_train, X_val, y_val, X_test = load_data()

    print("check for missing values...")
    check_missing_values(X_train)
    check_missing_values(y_train)

    print("check for duplicates...")
    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.time()
    if check_duplicates(X_train):
        raise ValueError("duplicates found in the training data")
    end_time = time.time()
    profiler.disable()
    profiler.print_stats(sort = 'cumtime')
    print(f"time taken to check duplicates: {end_time - start_time:.2f} seconds")
    
    print("check for overlapping...")
    if check_overlapping(X_train):
        raise ValueError("overlapping trajectories found in the training data")

    print("removing outliers...")
    X_train, y_train = remove_outliers(X_train, y_train)
    print("scaling data...")
    X_train, X_val, X_test, scaler = scale_data(X_train, X_val, X_test)

    print("creating datasets...")
    train_dataset = TrajectoryDataset(X_train, y_train, scaler)
    val_dataset = TrajectoryDataset(X_val, y_val, scaler)
    # test_dataset = TrajectoryDataset(X_test)

    print("creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers = 7, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size = 32, num_workers = 7, persistent_workers = True)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("initializing model...")
    model = TrajectoryModel()
    
    print("setting up tensorboard logger...")
    logger = TensorBoardLogger("tb_logs", name = "my_model")

    print("setting up callbacks...")
    checkpoint_callback = ModelCheckpoint(
        monitor = 'val_f1',
        filename = 'best-checkpoint-{epoch:02d}-{val_f1:.2f}',
        save_top_k = 1,
        mode = 'max',
        dirpath = 'checkpoints'
    )

    early_stopping_callback = EarlyStopping(
        monitor = 'validation_loss',
        patience = 3,
        verbose = True,
        mode = 'min'
    )

    print("initializing trainer...")
    trainer = pl.Trainer(
        max_epochs = 50,
        callbacks = [checkpoint_callback, early_stopping_callback],
        logger = logger,
        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        devices = 1 
    )
    print(f"unique labels in y_train: {np.unique(y_train)}")
    print(f"y_train example: {y_train[:5]}")
    
    print("starting training...")
    trainer.fit(model, train_loader, val_loader)
    print("training finished...")
    
if __name__ == '__main__':
    main()
