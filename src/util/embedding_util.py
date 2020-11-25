import re
from os import listdir
import datetime

import multiprocessing

import gc

import sys, gensim
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


class Corpus(object):
    """
    supervised org/per pair classifier

    """
    input_folder = None
    start_file_index = None

    def __init__(self, input_folder, start_file_index):
        self.input_folder = input_folder
        self.start_file_index=start_file_index

    def __iter__(self):
        files = [f for f in listdir(self.input_folder)]
        files=sorted(files)

        index=-1
        for f in files:
            index+=1
            if index<self.start_file_index:
                continue
            print(str(datetime.datetime.now())+" started from file:"+f)
            for line in open(self.input_folder+"/"+f,
                             encoding='utf-8', errors='ignore'):
                # assume there's one document per line, tokens separated by whitespace
                line=re.sub('[^0-9a-zA-Z]+', ' ', line).strip().lower()
                yield line.split()


'''
cbow_or_skip: 1 means skipgram, otherwise cbow
'''
def train_word2vec(input_folder, start_file_index, out_model_file, cbow_or_skip):
    cores = multiprocessing.cpu_count()
    print('num of cores is %s' % cores)
    gc.collect()

    sentences = Corpus(input_folder,start_file_index)
    print(str(datetime.datetime.now()) + ' training started...')
    model = Word2Vec(sentences=sentences,
                     size=300, window=5, min_count=3, sample=1e-4, negative=5, workers=cores, sg=cbow_or_skip)

    print(str(datetime.datetime.now()) + ' training completed, saving...')
    model.save(out_model_file)
    print(str(datetime.datetime.now()) + ' saving completed')

def preprocess_text(input_folder, start_file_index, out_file):
    sentences = Corpus(input_folder, start_file_index)

    with open(out_file, 'w', encoding="utf-8") as outfile:
        for s in sentences:
            if len(s)==0:
                continue
            new_s=' '.join(s)
            new_s=new_s.lower()
            outfile.write(new_s+"\n")

def gensim_to_vec(inFile, outFile):
    model = KeyedVectors.load(inFile, mmap='r')

def count_words(input_folder):
    files = [f for f in listdir(input_folder)]
    files = sorted(files)
    count=0
    for f in files:
        print(str(datetime.datetime.now()) + " started from file:" + f)
        for line in open(input_folder + "/" + f,
                         encoding='utf-8', errors='ignore'):
            # assume there's one document per line, tokens separated by whitespace
            line = re.sub('[^0-9a-zA-Z]+', ' ', line).strip().lower().strip()
            count+=len(line.split(" "))

    print(count)


def glove_2_wrd2vec(glove_input, output):
    glove2word2vec(glove_input_file=glove_input, word2vec_output_file=output)

def word2vec_to_nonbinary(w2v_binary, output):
    model=gensim.models.KeyedVectors. \
        load_word2vec_format(w2v_binary, binary=True)
    model.save_word2vec_format(output, binary=False)

'''
/home/zz/Work/data/embeddings/wop/desc_skip.gensim
/home/zz/Work/data/embeddings/wop/desc_skip.vec
'''
if __name__ == "__main__":
    word2vec_to_nonbinary(sys.argv[1],sys.argv[2])
    exit(0)

    glove_2_wrd2vec(sys.argv[1],sys.argv[2])
    exit(0)


    count_words("/home/zz/Downloads")
    exit(0)

    '''
    /home/zz/Work/gensim-util/word2vec/name_corpus
0
/home/zz/Work/gensim-util/word2vec/name_corpus_merged
0
    '''
    train_word2vec(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
    #preprocess_text(sys.argv[1],int(sys.argv[2]),sys.argv[3])
    gensim_to_vec(sys.argv[1],sys.argv[2])