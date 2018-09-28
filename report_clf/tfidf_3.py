import csv
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.feature_extraction import text 
from sklearn.metrics import roc_curve,auc,f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


stop_words = text.ENGLISH_STOP_WORDS.union([u'ap',u'dow',u's',u'p',u'spx',u'u',u'zack',u'cents',u'earnings',u'forecast',u'profit',u'estimate'])#estimate
ai_file = "file_after_filter/ai_finance_after_filter.csv"
ns_file = "file_after_filter/ns_finance_after_filter.csv"
human_file = "file_after_filter/human_finance_after_filter.csv"

#import csv, build label
#label: a=1,b=0
def import_data(file,row_content,x):
    content_1 = open(file, 'r')
    csv_reader = csv.reader(content_1)
    content = [row[row_content] for row in csv_reader]
    length = len(content)
    label = []
    for i in range(0,length):
        label.append(x)

    return content, label

ai_content, ai_label = import_data(ai_file,1,1)
ns_content, ns_label = import_data(ns_file,1,2)
human_content, human_label = import_data(human_file,1,0)

content = ai_content+ns_content+human_content
label = ai_label+ns_label+human_label

#test_train split, 2:8
def auc(content, label,cross_fold):
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

        vectorizer_train=TfidfVectorizer(encoding='utf-8', decode_error='ignore', strip_accents='unicode', 
                                     token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', stop_words=stop_words, 
                                     lowercase=True, analyzer='word',max_features=100)# ngram_range=(1,2), 
        tfidf_train = vectorizer_train.fit_transform(content_train)
        word_train = vectorizer_train.get_feature_names()
        pattern = []
        for word in word_train:
            pattern.append(word)
        tfidf_metric = tfidf_train.toarray()
        pattern_array = np.array([pattern])
        print pattern_array.size
        tfidf = np.concatenate((pattern_array, tfidf_metric))
        human_content_word=pd.DataFrame(tfidf)
        human_content_word.to_csv('C:/Users/lh.Lenovo-PC/Desktop/classifier_svm/3_human_auto_tfidf.csv')

        vectorizer_test=TfidfVectorizer(encoding='utf-8', decode_error='ignore', strip_accents='unicode', 
                                     token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', stop_words=stop_words, 
                                     lowercase=True, analyzer='word',vocabulary=vectorizer_train.vocabulary_)
        tfidf_test = vectorizer_test.fit_transform(content_test)
        word_test = vectorizer_test.get_feature_names()


        clf = svm.SVC(kernel='linear', probability=True)
        clf_res = clf.fit(tfidf_train, label_train)
        pred =  clf_res.predict(tfidf_test)
        print pred
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

    #    clf_res = clf.fit(tfidf_train, label_train)
    #    pred =  clf_res.predict(tfidf_train)
    #    predict_prob = clf_res.predict_proba(tfidf_test)[:,1]
    #    score = confusion_matrix(label_test,predict_prob)
    #    print score
    #    auc_mean = auc_mean+score
    #auc_mean = auc_mean/cross_fold
    #print auc

auc(content,label,10) 