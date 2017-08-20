#!/usr/bin/env bash

if [[ ! -e 'glove.840B.300d.w2vformat.txt' ]]; then
    wget http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip glove.840B.300d.zip
    python -m gensim.scripts.glove2word2vec --input  glove.840B.300d.txt --output glove.840B.300d.w2vformat.txt
fi
if [[ ! -e 'numberbatch-17.06.txt' ]]; then
    wget https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-17.06.txt.gz
    gunzip numberbatch-17.06.txt.gz
fi
