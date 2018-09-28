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
auto_file = "file_after_filter/auto_finance_after_filter.csv"
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

auto_content, auto_label = import_data(auto_file,1,1)
human_content, human_label = import_data(human_file,1,0)
len1=len(auto_content)
len2=len(human_content)
data = auto_content+human_content
label = auto_label+human_label

def extract_sentence(content,percent):
    new_content = []
    for line in content:
        new = line.split('.')
        sum = len(new)+1
        sum = math.ceil(sum*percent)
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
        content_auto = content[0:1450]
        content_human = content[1451:2966]
        label_auto = label[0:1450]
        label_human = label[1451:2966]
        random_num = np.random.randint(low=0, high=100)
        print 'random_num_auto:' +str(random_num)
        content_train_auto,content_test_auto,label_train_auto,label_test_auto = train_test_split(content_auto, label_auto, test_size=0.2,random_state=random_num)
        random_num = np.random.randint(low=0, high=100)
        print 'random_num_human:' +str(random_num)
        content_train_human,content_test_human,label_train_human,label_test_human = train_test_split(content_human, label_human, test_size=0.2,random_state=random_num)

        content_train = content_train_auto+content_train_human
        content_test = content_test_auto+content_test_human
        label_train = label_train_auto+label_train_human
        label_test = label_test_auto+label_test_human

        vectorizer_train=TfidfVectorizer(encoding='utf-8', decode_error='ignore', strip_accents='unicode', 
                                     token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', stop_words=stop_words, 
                                     lowercase=True, analyzer='word',max_features=100)# ngram_range=(1,2), 
        tfidf_train = vectorizer_train.fit_transform(content_train)
        word_train = vectorizer_train.get_feature_names()
        tfidf_metric_train = tfidf_train.toarray()

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
        #pred =  clf_res.predict(tfidf_test)
        ##predict_prob = clf_res.predict_proba(tfidf_test)[:,1]
        #auc = metrics.roc_auc_score(label_test,pred)
        #print 'auc: %0.20f'%auc
        #auc_mean = auc_mean+auc
    #auc_mean = auc_mean/cross_fold
    #x_axis = range(1,11)
    #x=np.array(x_axis)
    #plt.plot(x,auc_mean)
    #plt.show()
    #print auc_mean
    auc_mean = list(f1_mean)
    auc_mean_csv=pd.DataFrame(auc_mean)
    auc_mean_csv.to_csv('auc/f1_array.csv',mode='a',header=False)



auc(data,label,10)
