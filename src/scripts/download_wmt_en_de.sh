#!/usr/bin/env bash

if [[ ! -e './data/en-de/train.en' ]]; then
    mkdir -p data/en-de
    cd data/en-de
    mkdir -p vectors
    # training data
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de
    # testing data
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de
    # 50k most frequent vocab
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.en
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.de
    # dictionary
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/dict.en-de
    cd ../..
fi
