import os
import pandas as pd
import re
import numpy as np
import fasttext.util
import datetime
from utils import load_vec, get_vec, preprocessing_text

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

embedding_path = '../embedding/fasttext/'
os.makedirs(embedding_path, exist_ok=True)

src_path = '../vectors/vectors-ko.txt'
tgt_path = '../vectors/vectors-en.txt'
nmax = 50000  # maximum number of word embeddings to load

src_embeddings, src_id2word, src_word2id = load_vec(src_path, nmax)
tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_path, nmax)
print(src_embeddings.shape, tgt_embeddings.shape)

word2id = {v: k for k, v in src_id2word.items()}

train = pd.read_csv('../data/total_train.csv', encoding='euc-kr')
test = pd.read_csv('../data/total_test.csv', encoding='euc-kr')
total = pd.concat([train, test])
total.reset_index(drop=True, inplace=True)
print(train.shape, test.shape)
print(total.head())
    
total['preprocessed'] = total['description'].map(lambda x: preprocessing_text(x))
print('After preprocessing: ', total['preprocessed'].iloc[32])

train = total[:train.shape[0]]
test = total[train.shape[0]:]
print(train.shape, test.shape)

# train.to_csv('./total_train.csv', index=False, encoding='euc-kr')
# test.to_csv('./total_test.csv', index=False, encoding='euc-kr')

ft = fasttext.load_model('../vectors/wiki.en/wiki.en.bin')
max_length = max([len(doc) for doc in total['description'] ])
#max_length = 50000
print('Max length: ', max_length)

print('[train dataset start {} ....]'.format(datetime.datetime.now()))
result = np.array([[[0.0 for j in range(300)] for i in range(max_length)] for k in range(train.shape[0])])
for k, sent in enumerate(total['description'][:train.shape[0]].tolist()):
    if k % 500 == 0:
        print('\t -- train {} row ....'.format(k))
    
    for i, word in enumerate(sent):
        if re.search('[ㄱ-ㅣ가-힣]+', word):
            result[k][i] = get_vec(word, src_embeddings, src_id2word, tgt_embeddings, tgt_id2word, ft, K=1)            
        else:
            result[k][i] = ft[word]

print('\t -- save train data .... ')
np.save(os.path.join(embedding_path, 'fasttext_x_train'), result)

print('[test dataset start {} ....]'.format(datetime.datetime.now()))
result = np.array([[[0.0 for j in range(300)] for i in range(max_length)] for k in range(test.shape[0])])
for k, sent in enumerate(total['description'][train.shape[0]:].tolist()):
    if k % 500 == 0:
        print('\t -- test {} row ....'.format(k))
    
    for i, word in enumerate(sent):
        if re.search('[ㄱ-ㅣ가-힣]+', word):
            result[k][i] = get_vec(word, src_embeddings, src_id2word, tgt_embeddings, tgt_id2word, ft, K=1)
        else:
            result[k][i] = ft[word]

print('\t -- save test data .... ')
np.save(os.path.join(embedding_path, 'fasttext_x_test'), result)

labels = np.array(train['label'].tolist())
np.save(os.path.join(embedding_path, 'fasttext_y_train'), labels)

labels = np.array(test['label'].tolist())
np.save(os.path.join(embedding_path, 'fasttext_y_test'), labels)