# Relation Extraction
Following the quick research performed in NER task, relation extraction should be related to NER. To follow consistency, we will using the same model with NER task, i.e., [pre-trained BERT base](https://arxiv.org/abs/1810.04805) and [BioBERT](http://doi.org/10.1093/bioinformatics/btz682) with modification on the model.

## Preprocessing
The data comprises of sentence and label pair. The `preprocess.py` is basically extracting the data and label from the file to be turned into a readable format for our pre-trained BERT model.

The `torchdataset.py` comes after preprocess. We are using the same tokenizer from BERT, both the base BERT and BioBERT. The torch dataset will then be ready to be fed to torch DataLoader.

## Model
As mentioned above, we are using two models:

- BERT base cased
- BioBERT

Both models will be connected to a classification layer, i.e., linear layer to match our needs in classifying sentences that have the relation between entities. The model will take the CLS layer within the hidden output of BERT since we are performing a binary classification task.

## Training
- For training, we specify the hyperparameters in `hyperparameter.py`. Most of the hyperparameters following the recommended by the author of BERT.
- The code will use GPU as the default for training.
- For every training, hyperparameter used will be saved at default folder `models/re/saved_model`. Directory can be changed in `hyperparameter.py`
- The F1 metric is used when saving the model. Higher F1 score between epochs will trigger the save model into `models/re/saved_model`.

## Results
To be constructed