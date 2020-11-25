#!/bin/bash
#$ -pe openmp 1 -l h_rt=168:00:00 -l rmem=12G -m bea -M ziqi.zhang@sheffield.ac.uk



module load apps/python/anaconda3-2.5.0
#module load libs/binlibs/cudnn/5.1-cuda-8.0.44
source activate theano
#export MKL_THREADING_LAYER=GNU
#export PYTHONPATH=/home/ziqizhang/chase/python/src
export PYTHONPATH=/fastdata-sharc/li1zz/gensim-util/src

python3 -m util.embedding_util /fastdata-sharc/li1zz/wop/resources/word2vec/name_corpus 0 /fastdata-sharc/li1zz/wop/resources/word2vec/name_cbow.bin 0

