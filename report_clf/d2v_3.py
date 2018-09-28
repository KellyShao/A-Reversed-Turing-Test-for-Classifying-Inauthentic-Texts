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

stop_words = text.ENGLISH_STOP_WORDS.union([u'ap',u'dow',u's',u'p',u'spx',u'u',u'zack',u'cents',u'earnings',u'forecast',u'profit',u'estimate'])#estimate
ai_file = "file_after_filter/ai_finance_after_filter.csv"
ns_file = "file_after_filter/ns_finance_after_filter.csv"
human_file = "file_after_filter/human_finance_after_filter.csv"

#import csv, build label
#label: a=1,b=0
def import_data(file,row_content,x):
    content_1 = open(file, 'r')
    csv_reader = csv.reader(content_1)
    content = []
    for row in csv_reader:
        row_new = remove_stopwords(row[row_content])
        row_new = strip_numeric(row_new)
        row_new = strip_non_alphanum(row_new)   
        row_new = strip_short(row_new,minsize = 3)
        content.append(row_new)
    length = len(content)
    label = []
    for i in range(0,length):
        label.append(x)

    return content, label

ai_content, ai_label = import_data(ai_file,1,1)
ns_content, ns_label = import_data(ns_file,1,2)
human_content, human_label = import_data(human_file,1,0)

print len(ai_label)
print len(ns_label)
print len(human_label)

data = ai_content+ns_content+human_content
label = ai_label+ns_label+human_label


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

model = doc2vec.Doc2Vec(content1, min_count=1,window=5,size=200)
model.save("doc_ai_ns_human.model")

def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

def auc(content, label, dict, cross_fold):
    score_mean_micro = 0
    score_mean_macro = 0
    for i in range(0,cross_fold):
        content_ai = content[0:1333]
        content_ns = content[1333:2788]
        content_human = content[2788:4304]
        label_ai = label[0:1333]
        label_ns = label[1333:2788]
        label_human = label[2788:4304]
        
        random_num = np.random.randint(low=0, high=100)
        print 'random_num_ai:' +str(random_num)
        content_train_ai,content_test_ai,label_train_ai,label_test_ai = train_test_split(content_ai, label_ai, test_size=0.2,random_state=random_num)
        random_num = np.random.randint(low=0, high=100)
        print 'random_num_ns:' +str(random_num)
        content_train_ns,content_test_ns,label_train_ns,label_test_ns = train_test_split(content_ns, label_ns, test_size=0.2,random_state=random_num)
        random_num = np.random.randint(low=0, high=100)
        print 'random_num_human:' +str(random_num)
        content_train_human,content_test_human,label_train_human,label_test_human = train_test_split(content_human, label_human, test_size=0.2,random_state=random_num)

        content_train = content_train_ai+content_train_ns+content_train_human
        content_test = content_test_ai+content_test_ns+content_test_human
        label_train = label_train_ai+label_train_ns+label_train_human
        label_test = label_test_ai+label_test_ns+label_test_human
        print 'len(content_train):'+str(len(content_train))
        print 'len(content_test):'+str(len(content_test))

        #build matrics
        d2v_model =doc2vec.Doc2Vec.load("doc_ai_ns_human.model")
        print "model load successed"
        train = getVecs(d2v_model, content_train, 200)
        print "train builed"
        test = getVecs(d2v_model, content_test, 200)
        print "test build"

        clf = svm.SVC(kernel='linear')#, probability=True)
        clf_res = clf.fit(train, label_train)
        pred =  clf_res.predict(test)
        #predict_prob = clf_res.predict_proba(tfidf_test)[:,1]
        score_micro = f1_score(label_test, pred, average='micro')
        score_macro = f1_score(label_test, pred, average='macro') 
        print 'score_micro: %0.20f'%score_micro
        print 'score_macro: %0.20f'%score_macro
        score_mean_micro = score_mean_micro+score_micro
        score_mean_macro = score_mean_macro+score_macro
    score_mean_micro=score_mean_micro/10
    print 'score_mean_micro: %0.20f'%score_mean_micro
    score_mean_macro=score_mean_macro/10
    print 'score_mean_macro: %0.20f'%score_mean_macro

auc(content1, label, dict, 10)




