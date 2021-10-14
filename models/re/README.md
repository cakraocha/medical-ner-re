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
Below is the summary of training progress of BERT base:

```
Iteration 0 of epoch 0 complete. Acc: 0.375; Loss: 0.7931786775588989; Time taken (s): 7.950665235519409
Epoch 0 complete. Dev F1: 0.8672566371681416; Dev loss: 4.041399702429771
Best f1 score improved from 0 to 0.8672566371681416
Saving model..
Iteration 0 of epoch 1 complete. Acc: 0.75; Loss: 0.4716324806213379; Time taken (s): 30.334233283996582
Epoch 1 complete. Dev F1: 0.8514851485148514; Dev loss: 4.376714199781418
Iteration 0 of epoch 2 complete. Acc: 0.875; Loss: 0.29982128739356995; Time taken (s): 21.107520818710327
Epoch 2 complete. Dev F1: 0.6582278481012658; Dev loss: 5.4453389048576355
Iteration 0 of epoch 3 complete. Acc: 0.875; Loss: 0.29643476009368896; Time taken (s): 19.45540952682495
Epoch 3 complete. Dev F1: 0.7956989247311828; Dev loss: 5.3306107968091965
Iteration 0 of epoch 4 complete. Acc: 0.875; Loss: 0.1689884066581726; Time taken (s): 19.595792531967163
Epoch 4 complete. Dev F1: 0.7912087912087913; Dev loss: 4.899741351604462
```

Strong F1 score from the start. We move to the metrics:

```
F1 Score: 0.6157094594594593

Accuracy: 0.7297297297297297

Confusion Matrix:

| preds/true | 0  | 1  |
| ---------- | -- | -- |
| 0          | 0  | 10 |
| 1          | 0  | 27 |

```

Looks like we got all wrong for the 0 label. This was due to both small training and dev dataset makes the model could not perform well.

I am curious to try using BioBERT. Below is the summary of training progress of BioBERT:

```
Iteration 0 of epoch 0 complete. Acc: 0.75; Loss: 0.68260657787323; Time taken (s): 10.71143627166748
Epoch 0 complete. Dev F1: 0.8672566371681416; Dev loss: 4.0688506960868835
Best f1 score improved from 0 to 0.8672566371681416
Saving model..
Iteration 0 of epoch 1 complete. Acc: 0.75; Loss: 0.5197035074234009; Time taken (s): 25.040106534957886
Epoch 1 complete. Dev F1: 0.8727272727272727; Dev loss: 4.059982776641846
Best f1 score improved from 0.8672566371681416 to 0.8727272727272727
Saving model..
Iteration 0 of epoch 2 complete. Acc: 0.875; Loss: 0.2677075266838074; Time taken (s): 25.442264080047607
Epoch 2 complete. Dev F1: 0.86; Dev loss: 4.16020954400301
Iteration 0 of epoch 3 complete. Acc: 0.875; Loss: 0.2229587882757187; Time taken (s): 19.271546125411987
Epoch 3 complete. Dev F1: 0.7586206896551724; Dev loss: 4.713299125432968
Iteration 0 of epoch 4 complete. Acc: 1.0; Loss: 0.05773357301950455; Time taken (s): 19.12383770942688
Epoch 4 complete. Dev F1: 0.8571428571428571; Dev loss: 5.478869237005711
```

And the metrics:

```

F1 Score: 0.6746226746226746

Accuracy: 0.7567567567567568

Confusion Matrix:

| preds/true | 0  | 1  |
| ---------- | -- | -- |
| 0          | 1  | 9  |
| 1          | 0  | 27 |

```

From a glance, it is improving from BERT base. With a small training and test dataset, I think this model is performing quite well.

## Future improvement
Since we have limited data, we can try to play with active learning to try improving the performance of the model. We can also try to scrape more data to allow the model distinguish between entities relation of sentences.