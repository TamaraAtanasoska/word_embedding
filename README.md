# About
This is a repository containing the final project of the Deep Learning for
Natural Language Processing course, part of the Cognitive Systems program and
the Potsdam University. 

The aim of the project is to reproduce the model and findings used in the
["Enriching Word Vectors with Subword Information"](https://arxiv.org/pdf/1607.04606.pdf) by Bojanowski et al. 

Authors of the project are Bhuvanesh Verma and Tamara Atanasoska.

#Data

We used different version of [Wikipedia data](https://dumps.wikimedia.org/). Largest dataset we used
is first 1 Billion bytes of English Wikipedia which can be obtained from [here](http://mattmahoney.net/dc/enwik9.zip). 
Since this data dump is too big(around 1GB) it is not available in project repo. This data dump is raw web data which require some
preprocessing(details can be found [here](https://fasttext.cc/docs/en/unsupervised-tutorial.html)).

For most of the training, we used smaller version of original data dump `data/text8`, which is one-seventh in size. Next dataset we
used is Universal Dependencies dataset [v2.8](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3687)
for English(`data/UD/UD_EN_train.txt`) and German (`data/UD/UD_DE_train.txt`). 

We used another set of datasets mentioned in `Section 5.1 `of  ["Enriching Word Vectors with Subword Information"](https://arxiv.org/pdf/1607.04606.pdf)
to evaluate our model. These datasets are used for evaluating model on word similarity task.

English
1. `eval/data/WS353/wordsim_relatedness_goldstandard.txt`
2. `eval/data/EN-RG-65.txt`
3. `eval/data/EN-RW.txt`

German
1. `eval/data/GUR65.txt`
2. `eval/data/GRU350.txt`
3. `eval/data/ZG222.txt`

To evaluate model on word analogy task, we used the Google semantic/syntactic analogy datasets introduced in 
[Mikolov et al. (2013)](https://code.google.com/archive/p/word2vec/) and its [German translation](https://www.ims.uni-stuttgart.de/forschung/ressourcen/lexika/analogies/).

English : `eval/data/EN-GOOGLE.txt`

German : `eval/data/DE-GOOGLE.txt`


# Usage
In order to run the project, it is preferred to create a conda virtual environment. Start
by installing requirements from ``requirements.txt`` as,

``pip install -r requirements.txt``

This will setup the environment to run project.

## Run
Execute following command :

### Train
``python run.py --RUN_MODE train --SUBSAMPLING --NGRAMS --DATA data/text8``

    --RUN_MODE : {train, val}
    --SUBSAMPLING : true if provided else false, use subsampling while training
    --NGRAMS : true if provided else false, use ngrams while training
    --DATA : path to data file
    other arguments:
    --VERSION : load a specific (saved) model using version number and checkpoint
    --CKPT-E : checkpoint used to load a model to resume training or evaluate model
    --RESUME : true if provided else false, resume from given epoch
    --DEBUG  : to run overfitting

### Resume
``python run.py --RUN_MODE train --SUBSAMPLING --NGRAMS --DATA data/text8 --RESUME --VERSION 
VERSION_NUM --CKPT-E EPOCH_NUM``

### Evaluation
``python run.py --RUN_MODE val --VERSION VERSION_NUM --CKPT-E EPOCH_NUM``

### Overfitting
``python run.py --RUN_MODE train --DEBUG --SUBSAMPLING --NGRAMS --DATA data/text8``

###NOTE
For ngram model, we add embeddings of word and its ngrams. Resulting vector is then multiplied with
target word embedding or negative word embeddings. This multiplication can result in large 
values and hence produce `inf`, `-inf` and `nan` values on performing further operations like 
taking sigmoid or log. To tackle this situation, we replace `nan` and `inf` values with 0 and 
max/min datatype values respectively using `torch.nan_to_num()` method. In addition to this, we normalize
input and output embedding vectors which also helps in tackling this situation. Though after 
adding normalization, we required more training time to get better model.

Even after all these precautions, model can still produce `nan` value for loss during training. In this case, it is
advised to decrease `lr` or `batch_size`.



