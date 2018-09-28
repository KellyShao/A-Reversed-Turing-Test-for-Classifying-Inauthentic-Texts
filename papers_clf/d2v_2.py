#doc2vector

import csv
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_extraction import text
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve,auc,f1_score
import matplotlib.pyplot as plt
from gensim.models import doc2vec
from gensim import corpora
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_short
from gensim.parsing.preprocessing import strip_non_alphanum

stop_words = text.ENGLISH_STOP_WORDS.union([u'apr',u'archetypr',u'aug',u'configuration',u'conference',u'continuing'])#estimate
sci_file = "cs_papers/sci_after_filter.csv"
scigen_file = "cs_papers/scigen_after_filter.csv"

#import csv, build label
#label: a=1,b=0
def import_data(a_file,a_row,b_file,b_row):
    a_content = []
    a_content_1 = open(a_file, 'r')
    csv_reader_a = csv.reader(a_content_1)
    for row in csv_reader_a:
        row_new = remove_stopwords(row[a_row])
        row_new = strip_numeric(row_new)
        row_new = strip_non_alphanum(row_new)   
        row_new = strip_short(row_new,minsize = 3)
        a_content.append(row_new)
    a_length = len(a_content)
    a_label = np.ones(a_length)
    a_label = a_label.tolist()
    
    b_content = []
    b_content_1 = open(b_file, 'r')
    csv_reader_b = csv.reader(b_content_1)
    for row in csv_reader_b:
        row_new = remove_stopwords(row[a_row])
        row_new = strip_numeric(row_new)
        row_new = strip_non_alphanum(row_new)    
        row_new = strip_short(row_new,minsize = 3)
        b_content.append(row_new)
    b_length = len(b_content)
    b_label = np.zeros(b_length)
    b_label = b_label.tolist()
  
    return a_content, a_label, b_content, b_label


sci_content, sci_label,scigen_content, scigen_label = import_data(sci_file,1,scigen_file,1)
len1=len(sci_content)
len2=len(scigen_content)
data = sci_content+scigen_content
label = sci_label+scigen_label

def build_dict(data,stop_words):
    texts = [[word for word in document.lower().split() if word not in stop_words]
             for document in data]
    dictionary = corpora.Dictionary(texts)
    return texts,dictionary

content, dict = build_dict(data,stop_words)

def labelizeReviews(reviews):
        labelized = []
        for i,v in enumerate(reviews):
            label = '%s'%(i)
            labelized.append(doc2vec.LabeledSentence(v, [label]))
        return labelized

content1 = labelizeReviews(content)
model = doc2vec.Doc2Vec(content1,min_count=1,window=5,size=200)
model.save("doc_auto_human_passage.model")

def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

def f1(content, label, cross_fold):
    f1_mean = 0
    for i in range(0,cross_fold):
        print i
        content_auto = content[0:928]
        content_human = content[928:1836]
        label_auto = label[0:928]
        label_human = label[928:1836]
        random_num = np.random.randint(low=0, high=100)
        content_train_auto,content_test_auto,label_train_auto,label_test_auto = train_test_split(content_auto, label_auto, test_size=0.2,random_state=random_num)
        random_num = np.random.randint(low=0, high=100)
        content_train_human,content_test_human,label_train_human,label_test_human = train_test_split(content_human, label_human, test_size=0.2,random_state=random_num)

        content_train = content_train_auto+content_train_human
        content_test = content_test_auto+content_test_human
        label_train = label_train_auto+label_train_human
        label_test = label_test_auto+label_test_human

        #build matrics
        d2v_model = doc2vec.Doc2Vec.load("doc_auto_human_passage.model")
        #print "model load successed"
        train = getVecs(d2v_model, content_train, 200)
        #print "train builed"
        test = getVecs(d2v_model, content_test, 200)
        #print "test build"

        clf = svm.SVC(kernel='linear')
        clf_res = clf.fit(train, label_train)
        pred =  clf_res.predict(test)
        score_micro = f1_score(label_test, pred, average='micro')
        score_macro = f1_score(label_test, pred, average='macro')
        f1=(score_macro+score_micro)/2
        print 'f1: %0.20f'%f1
        f1_mean+=f1
    f1_mean = f1_mean/cross_fold
    print 'f1_mean: %0.20f'%f1_mean

f1(content1, label, 10)




