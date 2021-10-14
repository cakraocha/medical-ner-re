# Named Entity Recognition
To start the task, a quick research on the [current NER progress](http://nlpprogress.com/english/named_entity_recognition.html) listed some promising model. To start the project, we will using a [pre-trained BERT base](https://arxiv.org/abs/1810.04805) as the model to detect entity from a text as the state-of-the-art model for NLP. As the training progressed, I found that there are other model that is worth to try called [BioBERT](http://doi.org/10.1093/bioinformatics/btz682), which is a pretrained BERT model that was adapted using biomedical corpora.

## Preprocessing
Since the data only consists of individual text, we need to transform the data first into sentences to be able to read the contextual representation so that our pre-trained BERT model can do the work. The `preprocess.py` is built to transform the data into a readable format for our pre-trained BERT model.

After the data preprocessed, we build a torch dataset in `torchdataset.py`. Here we are using tokenizer from BERT, both the base BERT and BioBERT. Some additional process were done in tokenizing the data to make sure that there are enough padding and special tokens to the sentence. The torch dataset will then be ready to be fed to torch DataLoader.

## Model
As mentioned above, we are using two models:

- BERT base cased
- BioBERT

Both models will be connected to another linear layer to match our needs in detecting entities specific to our data. The model will take the sequence layer of the BERT since we are going to predict token by token that belongs to specific entity.

## Training
- For training, we specify the hyperparameters in `hyperparameter.py`. Most of the hyperparameters following the recommended by the author of BERT.
- The code will use GPU as the default for training.
- For every training, hyperparameter used will be saved at default folder `models/ner/saved_model`. Directory can be changed in `hyperparameter.py`
- Since accuracy or F1 is not a good metrics when deciding the model (due to prediction of each tokens may lead to ambigious metric), loss metric is used when saving the model. Lower loss between epochs will trigger the save model into `models/ner/saved_model`.

## Results
To be constructed