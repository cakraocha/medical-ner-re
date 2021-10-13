# This file acts as a constant for hyperparameter tuning

train_param = {
    'batch_size': 32,
    'shuffle': True,
    'num_workers': 2
}

dev_param = {
    'batch_size': 16,
    'shuffle': True,
    'num_workers': 2
}

MAX_LEN = 125
EPOCHS = 5
LEARNING_RATE = 2e-05
TRAIN_DEV_SPLIT = 0.8