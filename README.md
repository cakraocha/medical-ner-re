# Information Extraction for Medical Data
This is a project to showcase information extraction, specifically named-entity recognition (NER) and relation extraction (RE) for medical data.

## Named-Entity Recognition
For this task, we would like to create a model that can read entities from a certain text. Entities are in the format of IOB tagging. The data comprises of individual text (token) with IOB tag (label). The dataset does not specify entity types, so the label just comprises of {B, I, O}.

To start the project, a quick research on the [current NLP progress](http://nlpprogress.com/english/named_entity_recognition.html) listed some promising model. To start the project, we will using a [pre-trained BERT base](https://arxiv.org/abs/1810.04805) as the model to detect entity from a text as the state-of-the-art model for NLP.

### Preprocessing
Since the data only consists of individual text, we need to transform the data first into sentences to be able to read the contextual representation so that our pre-trained BERT model can do the work. The `preprocess.py` is built to transform the data into a readable format for our pre-trained BERT model.

