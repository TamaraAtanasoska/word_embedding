# About
This is a repository containing the final project of the Deep Learning for
Natural Language Processing course, part of the Cognitive Systems program and
the Potsdam University. 

The aim of the project is to reproduce the model and findings used in the
["Enriching Word Vectors with Subword Information"](https://arxiv.org/pdf/1607.04606.pdf) by Bojanowski et al. 

Authors of the project are Bhuvanesh Verma and Tamara Atanasoska.

# Usage
In order to run the project, it is preferred to create a conda virtual environment. Start
by installing requirements from ``requirements.txt`` as,

``pip install -r requirements.txt``

This will setup the environment to run project.

## Run
Execute following command :

### Train
``python run.py --RUN_MODE train --SUBSAMPLING --NGRAMS --DATA data/``

    --RUN_MODE : {train, val}
    --SUBSAMPLING : true if provided else false, use subsampling while training
    --NGRAMS : true if provided else false, use ngrams while training
    --DATA : path to data folder
    other arguments:
    --VERSION : load a specific (saved) model using version number and checkpoint
    --CKPT-E : checkpoint used to load a model or resume training
    --RESUME : true if provided else false, resume from given epoch

### Resume
``python run.py --RUN_MODE train --SUBSAMPLING --NGRAMS --DATA data/ --RESUME --VERSION 
VERSION_NUM --CKPT-E EPOCH_NUM``

### Evaluation
``python run.py --RUN_MODE val --SUBSAMPLING --NGRAMS --DATA data/ --VERSION 
VERSION_NUM --CKPT-E EPOCH_NUM``

###NOTE
For ngram model, we add embeddings of word and its ngrams. Resulting vector is then multiplied with
target word embedding or negative word embeddings. This multiplication can result in large 
values and hence produce `inf`, `-inf` and `nan` values on performing further operations like 
taking sigmoid or log. To tackle this situation, we replace `nan` and `inf` values with 0 and 
max/min datatype values respectively using `torch.nan_to_num()` method. In addition to this we normalize
input and output embedding vectors which also helps in tackling this situation. Though after 
adding normalization, we required more training time to get better model.

However, our model can still produce `nan` value for loss during training. So only solution for this
situation is to decrease `lr` or `batch_size`.

