# Information Extraction for Medical Data
This is a project to showcase information extraction, specifically named-entity recognition (NER) and relation extraction (RE) for medical data.

## Requirements
- GPU that supports CUDA
- PyTorch
- Transformers
- `export PYTHONPATH=../medical-ner-re`
- [BioBERT](https://github.com/dmis-lab/biobert)

## Named-Entity Recognition
For this task, we would like to create a model that can read entities from a certain text. Entities are in the format of IOB tagging. The data comprises of individual text (token) with IOB tag (label). The dataset does not specify entity types, so the label just comprises of {B, I, O}.

For details of the implementation, please visit [NER](https://github.com/cakraocha/medical-ner-re/tree/main/models/ner).

## Running NER model
- To train the model, execute `python models/ner/train.py` and make sure your `PYTHONPATH` environment variable is in the root path of this repo, i.e., `/medical-ner-re`.
- To use the BioBERT, make sure to download from [Hugging Face](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2/tree/main) and save it in the root path of this repo -> create directory `biobert_v1.1_pubmed/`. Make sure to follow the directory name as the model use the path name mentioned.
- To test the model, execute `python models/ner/test.py`

## Relation Extraction
For this task, we would like to create a model that able to predict whether there are any relations for medical entities in a sentence. We can call this a binary classification problem as label 0 denotes that there is no relation between entities in the respective sentence and label 1 otherwise.

For details of implementation, please visit [RE](https://github.com/cakraocha/medical-ner-re/tree/main/models/re).

## Running RE Model
- To train the model, execute `python models/re/train.py` and make sure your `PYTHONPATH` environment variable is in the root path of this repo, i.e., `/medical-ner-re`.
- To use the BioBERT, make sure to download from [Hugging Face](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2/tree/main) and save it in the root path of this repo -> create directory `biobert_v1.1_pubmed/`. Make sure to follow the directory name as the model use the path name mentioned.
- To test the model, execute `python models/re/test.py`