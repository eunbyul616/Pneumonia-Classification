import io
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
#             if len(word2id) == nmax:
#                 break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    for i, idx in enumerate(k_best):
        print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))

def get_vec(word, src_emb, src_id2word, tgt_emb, tgt_id2word, ft, K=5):
    word2id = {v: k for k, v in src_id2word.items()}
    try:
        word_emb = src_emb[word2id[word]]
    except KeyError:
        try:
            word_emb = src_emb[word2id[word[:-1]]]
        except KeyError:
            try:
                word_emb = src_emb[word2id[word[:-2]]]
            except KeyError:
                return ft[word]
        
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    
    return tgt_emb[k_best]

def preprocessing_text(raw_text, n=1):
    # 1. 따옴표 뒤 띄어쓰기
    raw_text = raw_text.replace('.', '. ').strip()
    
    # 2. 소문자 변환
    sentence = str(raw_text).lower()
    
    # therefore
    sentence = re.sub('([=|-]+)>', 'therefore', sentence)
    
    # r/o
    sentence = re.sub('r/o','therefore', sentence)
    
    #날짜
    sentence = re.sub('[\(]*[0-9]+-[0-9]+-[0-9]+[\)]*', '', sentence)
    
    # 3. tokenize
    words = []
    for sent in sent_tokenize(sentence.replace('.', '. ').strip()):
        words.append(word_tokenize(sent))
    
    # 4. stopwords 불용어 제거& 원형 복원
    #stops = set(stopwords.words('english'))
    s_list = []
    with open("../data/stopwords_en.txt", "r") as f:
         for line in f.readlines():
            s_list.append(line.strip())    

    with open("../data/stopwords_ko.txt", "r", encoding='utf-8') as f:
        for line in f.readlines():
            s_list.append(line.strip())

    try:
        s_list.remove('no')
    except Exception:
        pass
    try:
        s_list.remove('not')
    except Exception:
        pass
    try:
        s_list.remove('none')
    except Exception:
        pass
    
    stops = set(s_list)
    lm = WordNetLemmatizer()
    
    meaningful_words = []
    for ws in words:
        # 영어 뒤 한글 조사 제거
        ws = [re.sub('([a-zA-Z]+)([가-힣]+)', r'\1', w) for w in ws]
        
        # 숫자 포함 제거
        ws = [w for w in ws if not re.search('[0-9]+', w)]
        
        meaningful_words.extend([re.sub('[-=+,#?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', lm.lemmatize(w, pos="v"))\
                                 for w in ws if not w in stops if len(w) > 1])
    
    if n != 1:
        ngram_words = []
        
        if n == 2:
            ngram_words.extend([w[0]+'_'+w[1] for w in ngrams(meaningful_words, n)])
        elif n == 3:
            ngram_words.extend([w[0]+'_'+w[1] for w in ngrams(meaningful_words, n)])
            ngram_words.extend([w[0]+'_'+w[1]+'_'+w[2] for w in ngrams(meaningful_words, n)])
    
        counter = Counter(ngram_words)
        MIN_TOKEN_CNT = 3

        for k, v in counter.items():
            if v >= MIN_TOKEN_CNT:
                if len(k) > 1:
                    meaningful_words.append(k)
    
    return meaningful_words