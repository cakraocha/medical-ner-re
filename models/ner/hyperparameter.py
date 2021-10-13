# This file acts as a constant for hyperparameter tuning

# If using GPU, you can set num_workers > 0
# Otherwise (using CPU), set the num_workers = 0
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

MAX_LEN = 125  # The length of additional padding for BERT
EPOCHS = 5
LEARNING_RATE = 2e-05
TRAIN_DEV_SPLIT = 0.8