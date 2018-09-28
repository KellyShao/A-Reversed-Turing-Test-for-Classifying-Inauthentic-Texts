import csv
import math
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text 
from sklearn import metrics
from sklearn.metrics import roc_curve,auc,f1_score
import matplotlib.pyplot as plt
from gensim.models import word2vec
from gensim import corpora
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_short
from gensim.parsing.preprocessing import strip_non_alphanum


stop_words = text.ENGLISH_STOP_WORDS.union([u'ap',u'dow',u's',u'p',u'spx',u'u',u'zack',u'cents',u'earnings',u'forecast',u'profit',u'estimate'])#estimate
ai_file = "file_after_filter/ai_finance_after_filter.csv"
ns_file = "file_after_filter/ns_finance_after_filter.csv"
human_file = "file_after_filter/human_finance_after_filter.csv"

def import_data(file,row_content,x):
    content = []
    label = []
    content_1 = open(file, 'r')
    csv_reader = csv.reader(content_1)
    for row in csv_reader:
        row_new = remove_stopwords(row[row_content])
        row_new = strip_numeric(row_new)
        #row_new = strip_non_alphanum(row_new)   
        row_new = strip_short(row_new,minsize = 3)
        content.append(row_new)
    length = len(content)
    for i in range(0,length):
        label.append(x)
    
    return content,label

ai_content, ai_label = import_data(ai_file,1,1)
ns_content, ns_label = import_data(ns_file,1,2)
human_content, human_label = import_data(human_file,1,0)

content = ai_content+ns_content+human_content
label = ai_label+ns_label+human_label

def extract_sentence(content,percent):
    new_content = []
    for line in content:
        new = line.split('.')
        sum = len(new)+1
        sum = math.ceil(sum*percent)
        if sum>len(new):
            sum=sum-1
        elif sum<0:
            sum=sum+1
        cnt = 0
        new_line = ''
        for sent in new:
            cnt+=1
            if(cnt<=sum):
                new_line=new_line+sent
        new_content.append(new_line)
    return new_content

def auc(content, label,cross_fold):
    f1_mean = np.zeros(100)
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

        vectorizer_test=TfidfVectorizer(encoding='utf-8', decode_error='ignore', strip_accents='unicode', 
                                     token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', stop_words=stop_words, 
                                     lowercase=True, analyzer='word',vocabulary=vectorizer_train.vocabulary_)

        #build clf
        clf = svm.SVC(kernel='linear')#, probability=True)
        clf_res = clf.fit(tfidf_train, label_train)

        #input sentence
        for percent in range(1,101):
            new_content_test = extract_sentence(content_test,percent*0.01)
            tfidf_test = vectorizer_test.fit_transform(new_content_test)
            word_test = vectorizer_test.get_feature_names()
            
            pred =  clf_res.predict(tfidf_test)
            score_micro = f1_score(label_test, pred, average='micro')
            score_macro = f1_score(label_test, pred, average='macro')
            f1=(score_macro+score_micro)/2
            f1_mean[percent-1]+=f1
    f1_mean = f1_mean/cross_fold
    #x_axis = range(1,11)
    #x=np.array(x_axis)
    #plt.plot(x,f1_mean)
    #plt.show()
    print f1_mean
    f1_mean = list(f1_mean)
    f1_mean_csv=pd.DataFrame(f1_mean)
    f1_mean_csv.to_csv('auc/f1_array.csv',mode='a',header=False)

auc(content,label,10) 