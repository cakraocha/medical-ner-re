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
Below is the summary of training progress of BERT base:

```
Iteration 0 of epoch 0 complete. Loss: 1.1754934787750244; Time taken (s): 6.683253049850464
Iteration 100 of epoch 0 complete. Loss: 0.1709175705909729; Time taken (s): 20.8367178440094
Iteration 200 of epoch 0 complete. Loss: 0.16944904625415802; Time taken (s): 20.838332414627075
Iteration 300 of epoch 0 complete. Loss: 0.0787254348397255; Time taken (s): 20.95668888092041
Iteration 400 of epoch 0 complete. Loss: 0.027441687881946564; Time taken (s): 21.025715589523315
Iteration 500 of epoch 0 complete. Loss: 0.011987805366516113; Time taken (s): 21.06816864013672
Epoch 0 complete. Dev loss: 11.93519455252681
Best dev loss improved from inf to 11.93519455252681
Saving model..
Iteration 0 of epoch 1 complete. Loss: 0.10498951375484467; Time taken (s): 34.1438729763031
Iteration 100 of epoch 1 complete. Loss: 0.013606319203972816; Time taken (s): 20.812296867370605
Iteration 200 of epoch 1 complete. Loss: 0.0026513319462537766; Time taken (s): 20.874407291412354
Iteration 300 of epoch 1 complete. Loss: 0.001877196249552071; Time taken (s): 20.91485357284546
Iteration 400 of epoch 1 complete. Loss: 0.008127033710479736; Time taken (s): 20.96239137649536
Iteration 500 of epoch 1 complete. Loss: 0.08375941216945648; Time taken (s): 21.009525775909424
Epoch 1 complete. Dev loss: 9.629669020418078
Best dev loss improved from 11.93519455252681 to 9.629669020418078
Saving model..
Iteration 0 of epoch 2 complete. Loss: 0.06203871965408325; Time taken (s): 33.09073281288147
Iteration 100 of epoch 2 complete. Loss: 0.00016346135817002505; Time taken (s): 20.949991941452026
Iteration 200 of epoch 2 complete. Loss: 0.0030584288761019707; Time taken (s): 21.062754154205322
Iteration 300 of epoch 2 complete. Loss: 0.0009280999656766653; Time taken (s): 21.073688745498657
Iteration 400 of epoch 2 complete. Loss: 0.003735884092748165; Time taken (s): 21.108911752700806
Iteration 500 of epoch 2 complete. Loss: 0.01294661220163107; Time taken (s): 21.11540198326111
Epoch 2 complete. Dev loss: 12.72815231437562
Iteration 0 of epoch 3 complete. Loss: 0.006256065797060728; Time taken (s): 32.5059871673584
Iteration 100 of epoch 3 complete. Loss: 0.0005379688227549195; Time taken (s): 21.1772518157959
Iteration 200 of epoch 3 complete. Loss: 0.0002678562595974654; Time taken (s): 21.210396766662598
Iteration 300 of epoch 3 complete. Loss: 0.0029201179277151823; Time taken (s): 21.343735933303833
Iteration 400 of epoch 3 complete. Loss: 0.0001057834379025735; Time taken (s): 21.883874654769897
Iteration 500 of epoch 3 complete. Loss: 0.0008353428565897048; Time taken (s): 22.43835139274597
Epoch 3 complete. Dev loss: 12.385817555783433
Iteration 0 of epoch 4 complete. Loss: 0.004306961316615343; Time taken (s): 28.802776098251343
Iteration 100 of epoch 4 complete. Loss: 0.001831826171837747; Time taken (s): 21.351914167404175
Iteration 200 of epoch 4 complete. Loss: 0.00021290512813720852; Time taken (s): 21.799004316329956
Iteration 300 of epoch 4 complete. Loss: 0.000987226259894669; Time taken (s): 21.8204607963562
Iteration 400 of epoch 4 complete. Loss: 0.0003302792611066252; Time taken (s): 22.1076078414917
Iteration 500 of epoch 4 complete. Loss: 0.0007649910403415561; Time taken (s): 21.85316514968872
Epoch 4 complete. Dev loss: 12.818984278157586
```

As we can see that the loss coming from the training data seems pretty promising, but the dev loss is not too good. Training more than 2 epochs leads to overfitting. To get a more picture, below is the metrics calculated from the test prediction using the model with lowest dev loss:

```
F1 Score: [0.09042386 0.17467249 0.90553551]
          [0, 1, 2] | [B, I, O]

Accuracy: 0.8183859248071192

Confusion Matrix:

| true/preds | 0/B  | 1/I | 2/O   |
| ---------- | ---- | --- | ----- |
| 0/B        | 144  | 83  | 733   |
| 1/I        | 255  | 200 | 632   |
| 2/O        | 1826 | 920 | 19704 |

```

If we just blind look at the accuracy, it is not bad. But that is because there are so many 'O' tags. If we see the performance of 'B' and 'I' tags, which are the tags that we are interested in, are not too good. This is assumed due to the specific field of entity that we are interested in and the vocab tokenizer of BERT may not read the medical entity well.

Since the BERT base is not performing well, we try to switch the model to BioBERT. Hopefully, the tokenizer of BioBERT that has been trained on biomedical data can improve the performance. Below is the summary of training progress of BioBERT:

```
Iteration 0 of epoch 0 complete. Loss: 0.9859135150909424; Time taken (s): 6.732158184051514
Iteration 100 of epoch 0 complete. Loss: 0.11999278515577316; Time taken (s): 20.83251452445984
Iteration 200 of epoch 0 complete. Loss: 0.17570951581001282; Time taken (s): 21.065897941589355
Iteration 300 of epoch 0 complete. Loss: 0.08913961052894592; Time taken (s): 21.09768843650818
Iteration 400 of epoch 0 complete. Loss: 0.1333969682455063; Time taken (s): 21.16519522666931
Iteration 500 of epoch 0 complete. Loss: 0.02529064007103443; Time taken (s): 21.199289083480835
Epoch 0 complete. Dev loss: 7.847643520915881
Best dev loss improved from inf to 7.847643520915881
Saving model..
Iteration 0 of epoch 1 complete. Loss: 0.09590290486812592; Time taken (s): 34.22254228591919
Iteration 100 of epoch 1 complete. Loss: 0.03248155489563942; Time taken (s): 20.883516311645508
Iteration 200 of epoch 1 complete. Loss: 0.007631601765751839; Time taken (s): 20.947276830673218
Iteration 300 of epoch 1 complete. Loss: 0.01668022759258747; Time taken (s): 20.990944623947144
Iteration 400 of epoch 1 complete. Loss: 0.021560974419116974; Time taken (s): 21.0311279296875
Iteration 500 of epoch 1 complete. Loss: 0.008078915067017078; Time taken (s): 21.071865558624268
Epoch 1 complete. Dev loss: 8.214697004121263
Iteration 0 of epoch 2 complete. Loss: 0.0019098587799817324; Time taken (s): 29.385834455490112
Iteration 100 of epoch 2 complete. Loss: 0.0017374013550579548; Time taken (s): 21.15403151512146
Iteration 200 of epoch 2 complete. Loss: 0.0003005070611834526; Time taken (s): 21.290010452270508
Iteration 300 of epoch 2 complete. Loss: 0.01136130653321743; Time taken (s): 21.278900623321533
Iteration 400 of epoch 2 complete. Loss: 0.08214583247900009; Time taken (s): 21.722750186920166
Iteration 500 of epoch 2 complete. Loss: 0.003017899813130498; Time taken (s): 21.92023015022278
Epoch 2 complete. Dev loss: 9.176959357588203
Iteration 0 of epoch 3 complete. Loss: 0.016553791239857674; Time taken (s): 31.7413330078125
Iteration 100 of epoch 3 complete. Loss: 0.00028419465525075793; Time taken (s): 21.226750373840332
Iteration 200 of epoch 3 complete. Loss: 0.003195198019966483; Time taken (s): 21.880147218704224
Iteration 300 of epoch 3 complete. Loss: 0.0008305172086693347; Time taken (s): 22.571011066436768
Iteration 400 of epoch 3 complete. Loss: 0.001188737340271473; Time taken (s): 22.406192302703857
Iteration 500 of epoch 3 complete. Loss: 0.04259807616472244; Time taken (s): 22.702006578445435
Epoch 3 complete. Dev loss: 9.925833207584219
Iteration 0 of epoch 4 complete. Loss: 0.0024247015826404095; Time taken (s): 29.387441396713257
Iteration 100 of epoch 4 complete. Loss: 0.001010672189295292; Time taken (s): 21.412556886672974
Iteration 200 of epoch 4 complete. Loss: 0.04292012378573418; Time taken (s): 21.878153800964355
Iteration 300 of epoch 4 complete. Loss: 0.006530369631946087; Time taken (s): 22.524221897125244
Iteration 400 of epoch 4 complete. Loss: 0.005200778134167194; Time taken (s): 22.78289484977722
Iteration 500 of epoch 4 complete. Loss: 0.02326934039592743; Time taken (s): 22.34332823753357
Epoch 4 complete. Dev loss: 11.313737532444065
```

The training progress seems to not differ much from BERT base. We move to the metrics calculated from the test prediction using the model with lowest dev loss:

```
F1 Score: [0.0921659  0.17961383 0.90508366]
          [0, 1, 2] | [B, I, O]

Accuracy: 0.8181001755316978

Confusion Matrix:

| true/preds | 0/B  | 1/I | 2/O   |
| ---------- | ---- | --- | ----- |
| 0/B        | 150  | 79  | 731   |
| 1/I        | 247  | 200 | 640   |
| 2/O        | 1898 | 861 | 19691 |

```

Turns out, using BioBERT is not improving the performance from BERT base. It is improving slightly for 'B' and I assume the BioBERT tokenizer and model have a role here. It needs more investigation, but my preliminary assumption is due to the training data itself which needs more entites data.

## Future Improvement
We can try to move to other models, such as LSTM and Bi-LSTM. We also can try to scrape more data that contains more entities so that the model can focus more on the interesting tags, i.e., 'B' and 'I'.