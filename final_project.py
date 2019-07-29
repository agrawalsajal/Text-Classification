import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import re
import pickle 

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import heapq

train = True

# nltk.download('stopwords')

#### LOADING THE DATASET ####

# from sklearn.datasets import load_files
# reviews_train = load_files('data/train/')
# reviews_test = load_files('data/test/')
# X_train,y_train = reviews_train.data,reviews_train.target
# X_test,y_test = reviews_test.data,reviews_test.target


#### SAVING THE DATASET ####

# with open('X_train.pickle','wb') as f:
#     pickle.dump(X_train,f)

# with open('X_test.pickle','wb') as f:
#     pickle.dump(X_test,f)  
    
# with open('y_train.pickle','wb') as f:
#     pickle.dump(y_train,f)

# with open('y_test.pickle','wb') as f:
#     pickle.dump(y_test,f)


#### LOADING THE DATASET ####

with open('X_train.pickle','rb') as f:
    X_train = pickle.load(f)
    
with open('X_test.pickle','rb') as f:
    X_test = pickle.load(f)
    
with open('y_train.pickle','rb') as f:
    y_train = pickle.load(f)
    
with open('y_test.pickle','rb') as f:
    y_test = pickle.load(f)
    

X = X_train + X_test

y = np.concatenate([y_train,y_test])
X = X[:50000]
y = y[:50000]

length = 5000

def remove_special_symbols(sample):
    text = re.sub(r'\W', ' ', sample)
    text = re.sub(r'\d', ' ', text)
    text = text.lower()
    text = re.sub(r'br[\s$]', ' ', text)
    text = re.sub(r'\s+[a-z][\s$]', ' ',text)
    text = re.sub(r'b\s+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text
    

def process_words(words, corpus):
    n_words = []
    t_word = ''
    
    for word in words:
        antonyms = []
        if t_word in missed_words:
            if word not in stop_words:
                word = 'not_' + word
                t_word = ''
        elif t_word == 'not_':
            for syn in wordnet.synsets(word):
                for s in syn.lemmas():
                    for a in s.antonyms():
                        antonyms.append(a.name())
            if len(antonyms) >= 1:
                word = antonyms[0]
            else:
                word = t_word + word
            t_word = ''
                
        if word in missed_words:
            t_word = word  
        elif word == 'not':
            t_word = 'not_'
        
        if word != 'not' and word not in missed_words:
            n_words.append(word)
            
    text = ' '.join(n_words)
    corpus.append(text)
    return corpus


def process_data(sample, missed_words, stop_words):
    corpus = []
    for i in range(0, len(sample)):
        text = remove_special_symbols(str(sample[i]))
        words = text.split(' ')
        corpus = process_words(words, corpus)
    return corpus


stop_words = stopwords.words('english')
missed_words = ['don','won','doesn','couldn','isn','wasn','wouldn','can','ain','shouldn','not','havn','hadn','hasn','aren']

corpus = []
corpus = process_data(X[:length], missed_words, stop_words)

def count_words(corpus, stop_words):
    dict_word2count = {}
    for data in corpus:
        words = nltk.word_tokenize(data)
        for word in words:
            if word not in stop_words:
                if word not in dict_word2count.keys():
                    dict_word2count[word] = 1
                else:
                    dict_word2count[word] += 1
    return dict_word2count
     



def find_idf(corpus, dict_word2count, freq_words):
    word_idfs = {}
    for word in freq_words:
        doc_count = 0
        for data in corpus:
            if word in nltk.word_tokenize(data):
                doc_count += 1
        word_idfs[word] = np.log(len(corpus)/(1+doc_count))
    return word_idfs

def find_tf(corpus, freq_words):
    tf_matrix = {}
    for word in freq_words:
        doc_tf = []
        for data in corpus:
            frequency = 0
            for w in nltk.word_tokenize(data):
                if word == w:
                    frequency += 1
            tf_word = frequency/len(nltk.word_tokenize(data))
            doc_tf.append(tf_word)
        tf_matrix[word] = doc_tf
    return tf_matrix


    

def find_tfidf(corpus, num, stop_words):
    tfidf_matrix = []
    dict_word2count = count_words(corpus, stop_words)
    freq_words = heapq.nlargest(num,dict_word2count,key=dict_word2count.get)
    print(freq_words)
    tf_matrix = find_tf(corpus, freq_words)
    word_idfs = find_idf(corpus, dict_word2count, freq_words)
    for word in tf_matrix.keys():
        tfidf = []
        for value in tf_matrix[word]:
            score = value * word_idfs[word]
            tfidf.append(score)
        tfidf_matrix.append(tfidf)   
    X = np.asarray(tfidf_matrix)
    X = np.transpose(X)
    return X

# X = find_tfidf(corpus[:100], 20, stop_words)

# with open('X_scratch.pickle','wb') as f:
#     pickle.dump(X,f) 

# with open('X_scratch.pickle','rb') as f:
#     X = pickle.load(f)

# from sklearn.model_selection import train_test_split
# text_train, text_test, sent_train, sent_test = train_test_split(X, y[:100], test_size = 0.2, random_state = 0)


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = 500, min_df = 2, norm="l2", use_idf=True, sublinear_tf = True, max_df = 0.6, stop_words = stop_words)

print("finding tf idf vector");
X = tfidf.fit_transform(corpus[:length]).toarray()

with open('X_data.pickle','wb') as f:
    pickle.dump(X,f) 

with open('X_data.pickle','rb') as f:
    X = pickle.load(f)

with open('TFIDF.pickle','wb') as f:
    pickle.dump(tfidf,f) 

from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(X, y[:length], test_size = 0.2, random_state = 0)

from sklearn.svm import LinearSVC
classifier = LinearSVC(C = 0.1)

if train == True:

	classifier.fit(text_train,sent_train)

	with open('svcclassifier.pickle','wb') as f:
	    pickle.dump(classifier,f)
else:
	with open('svcclassifier.pickle','rb') as f:
	    classifier = pickle.load(f)

if train == True:
	sent_pred = classifier.predict(text_test)

	from sklearn.metrics import confusion_matrix
	cm = confusion_matrix(sent_test, sent_pred)
	print(cm)

	from sklearn.model_selection import cross_val_score
	accuracies = cross_val_score(estimator = classifier, X = text_train, y = sent_train, cv = 10)
	print(accuracies)
	print(accuracies.mean())
	print(accuracies.std())
else:
	print("executed training file");



























