import csv
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text 
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt


stop_words = text.ENGLISH_STOP_WORDS.union([u'ap',u'dow',u's',u'p',u'spx',u'u',u'zack',u'cents',u'earnings',u'forecast',u'profit',u'estimate'])#estimate
auto_file = "file_after_filter/auto_finance_after_filter.csv"
human_file = "file_after_filter/human_finance_after_filter.csv"

#import csv, build label
#label: a=1,b=0
def import_data(a_file,a_row,b_file,b_row):
    a_content_1 = open(a_file, 'r')
    csv_reader_a = csv.reader(a_content_1)
    a_content = [row[a_row] for row in csv_reader_a]
    a_length = len(a_content)
    a_label = np.ones(a_length)
    a_label = a_label.tolist()
    
    b_content_1 = open(b_file, 'r')
    csv_reader_b = csv.reader(b_content_1)
    b_content = [row[b_row] for row in csv_reader_b]
    b_length = len(b_content)
    b_label = np.zeros(b_length)
    b_label = b_label.tolist()
  
    return a_content, a_label, b_content, b_label


auto_content, auto_label,human_content, human_label = import_data(auto_file,1,human_file,1)
len1=len(auto_content)
len2=len(human_content)
data = auto_content+human_content
label = auto_label+human_label

def build_tfidf(content,label,len1,len2):
    vectorizer=TfidfVectorizer(encoding='utf-8', decode_error='ignore', strip_accents='unicode', 
                                     token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', stop_words=stop_words, 
                                     lowercase=True, analyzer='word')#,max_features=500)# ngram_range=(1,2), 
    tfidf = vectorizer.fit_transform(content)
    words = vectorizer.get_feature_names()
    pattern = []
    for word in words:
        pattern.append(word)
    tfidf_metric = tfidf.toarray()
    tfidf_list = tfidf_metric.tolist()
    pattern_array = np.array([pattern])
    tfidf = np.concatenate((pattern_array, tfidf_metric))
    content_word=pd.DataFrame(tfidf)
    content_word.to_csv('C:/Users/jzs1274/Desktop/classifier_svm/human_auto_tfidf.csv')

    ##calculate pattern
    #pattern = [words,(np.zeros(len(words))).tolist(),(np.zeros(len(words))).tolist()]
    #tfidf_list = tfidf_metric.tolist()
    #for i in range(0,len(tfidf_list[0])-1):#500
    #    for j in range(0,len(tfidf_list)-1):#2967
    #        if j<len1:
    #            if tfidf_list[j][i]!=0:
    #                pattern[1][i]+=1
    #        else: 
    #            if tfidf_list[j][i]!=0:
    #                pattern[2][i]+=1
    #content_word=pd.DataFrame(pattern)
    #content_word.to_csv('C:/Users/jzs1274/Desktop/classifier_svm/human_auto_tfidf_num.csv')

    #ns_pattern = []
    #ai_pattern = []
    #human_pattern = []
    #for i in range(0,len(tfidf_list[0])-1):
    #    if pattern[1][i]>400 and pattern[1][i]<480:
    #        ns_pattern.append(i)
    #    elif pattern[1][i]>950 and pattern[1][i]<980:
    #        ai_pattern.append(i)
    #    elif pattern[2][i]>1400 and pattern[2][i]<1520:
    #        human_pattern.append(i)
    #print ns_pattern
    #print ai_pattern
    #print human_pattern

#build_tfidf(data,label,len1,len2)

def auc(content, label,cross_fold):
    auc_mean = 0
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
        print 'len(content_train):'+str(len(content_train))
        print 'len(content_test):'+str(len(content_test))

        vectorizer_train=TfidfVectorizer(encoding='utf-8', decode_error='ignore', strip_accents='unicode', 
                                     token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', stop_words=stop_words, 
                                     lowercase=True, analyzer='word',max_features=100)# ngram_range=(1,2), 
        tfidf_train = vectorizer_train.fit_transform(content_train)
        word_train = vectorizer_train.get_feature_names()
        tfidf_metric_train = tfidf_train.toarray()


        vectorizer_test=TfidfVectorizer(encoding='utf-8', decode_error='ignore', strip_accents='unicode', 
                                     token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', stop_words=stop_words, 
                                     lowercase=True, analyzer='word',vocabulary=vectorizer_train.vocabulary_)
        tfidf_test = vectorizer_test.fit_transform(content_test)
        word_test = vectorizer_test.get_feature_names()


        clf = svm.SVC(kernel='linear')#, probability=True)
        clf_res = clf.fit(tfidf_train, label_train)
        pred =  clf_res.predict(tfidf_test)
        #predict_prob = clf_res.predict_proba(tfidf_test)[:,1]
        auc = metrics.roc_auc_score(label_test,pred)
        print 'auc: %0.20f'%auc
        auc_mean = auc_mean+auc
    auc_mean = auc_mean/cross_fold
    print 'auc_mean: %0.20f'%auc_mean


auc(data,label,10)








