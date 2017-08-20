#!/usr/bin/env bash

if [[ ! -e './data/en-cs/train.en' ]]; then
    mkdir -p data/en-cs
    cd data/en-cs
    mkdir -p vectors
    # training data
    wget https://nlp.stanford.edu/projects/nmt/data/wmt15.en-cs/train.en
    wget https://nlp.stanford.edu/projects/nmt/data/wmt15.en-cs/train.cs
    # testing data
    wget https://nlp.stanford.edu/projects/nmt/data/wmt15.en-cs/newstest2013.en
    wget https://nlp.stanford.edu/projects/nmt/data/wmt15.en-cs/newstest2013.cs
    wget https://nlp.stanford.edu/projects/nmt/data/wmt15.en-cs/newstest2014.en
    wget https://nlp.stanford.edu/projects/nmt/data/wmt15.en-cs/newstest2014.cs
    wget https://nlp.stanford.edu/projects/nmt/data/wmt15.en-cs/newstest2015.en
    wget https://nlp.stanford.edu/projects/nmt/data/wmt15.en-cs/newstest2015.cs
    # 50k most frequent vocab
    wget https://nlp.stanford.edu/projects/nmt/data/wmt15.en-cs/vocab.50K.en
    wget https://nlp.stanford.edu/projects/nmt/data/wmt15.en-cs/vocab.50K.cs
    # dictionary
    wget https://nlp.stanford.edu/projects/nmt/data/wmt15.en-cs/dict.en-cs
    cd ../..
fi
