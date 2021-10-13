# This file acts as a constant for hyperparameter tuning

# If using GPU, you can set num_workers > 0
# Otherwise (using CPU), set the num_workers = 0
train_param = {
    'batch_size': 16,
    'shuffle': True,
    'num_workers': 2
}

dev_param = {
    'batch_size': 8,
    'shuffle': True,
    'num_workers': 2
}

MAX_LEN = 125  # The length of additional padding for BERT
EPOCHS = 5
LEARNING_RATE = 2e-05
TRAIN_DEV_SPLIT = 0.8
GPU = 0  # Deciding which gpu to be used. 0 is the default value

import json

def save_hp_to_json(traintime):
    """
    A function to save hyperparameter data to json file

    args:
    datetime -- datetime object to generate hyperparameter json file
                e.g. 'hyperparam_20211013_075856'

    """
    hp_data = {}
    hp_data['train_param'] = train_param
    hp_data['dev_param'] = dev_param
    hp_data['max_len'] = MAX_LEN
    hp_data['epochs'] = EPOCHS
    hp_data['learning_rate'] = LEARNING_RATE
    hp_data['train_dev_split'] = TRAIN_DEV_SPLIT
    hp_data['gpu'] = GPU
    savedate = traintime.strftime("%d%m%Y")
    savetime = traintime.strftime("%H%M%S")
    with open(f'models/ner/saved_model/hyperparam_{savedate}_{savetime}.json', 'w') as outfile:
        json.dump(hp_data, outfile)
