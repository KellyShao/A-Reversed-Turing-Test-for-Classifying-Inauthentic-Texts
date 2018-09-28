import csv
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text 
from sklearn import metrics
from sklearn.metrics import roc_curve,auc, f1_score
import matplotlib.pyplot as plt


stop_words = text.ENGLISH_STOP_WORDS
file = "dialogue/Human-Machine.csv"

#import csv, build label
#label: human=1,machine=0
def import_data(file):
    human = []
    machine = []
    content = open(file, 'r')
    csv_reader = csv.reader(content)
    for row in csv_reader:
        human.append(row[2])
        machine.append(row[3])
    length = len(human)
    human_label = np.ones(length)
    human_label = human_label.tolist()
    machine_label = np.zeros(length)
    machine_label = machine_label.tolist()
    
    return human,human_label,machine,machine_label


human_content, human_label,machine_content, machine_label = import_data(file)
len1=len(human_content)
len2=len(machine_content)
data = human_content+machine_content
label = human_label+machine_label

def build_tfidf(content,label,len1,len2):
    vectorizer=TfidfVectorizer(encoding='utf-8', decode_error='ignore', strip_accents='unicode', 
                                     token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', stop_words=stop_words, 
                                     lowercase=True, analyzer='word',max_features=200)# ngram_range=(1,2), 
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
    content_word.to_csv('clf_file/sci_scigen_tfidf.csv')


def f1(content, label,cross_fold):
    f1_mean = 0
    for i in range(0,cross_fold):
        print i
        content_auto = content[0:994]
        content_human = content[994:1988]
        label_auto = label[0:994]
        label_human = label[994:1988]
        random_num = np.random.randint(low=0, high=100)
        #print 'random_num_auto:' +str(random_num)
        content_train_auto,content_test_auto,label_train_auto,label_test_auto = train_test_split(content_auto, label_auto, test_size=0.2,random_state=random_num)
        random_num = np.random.randint(low=0, high=100)
        #print 'random_num_human:' +str(random_num)
        content_train_human,content_test_human,label_train_human,label_test_human = train_test_split(content_human, label_human, test_size=0.2,random_state=random_num)

        content_train = content_train_auto+content_train_human
        content_test = content_test_auto+content_test_human
        label_train = label_train_auto+label_train_human
        label_test = label_test_auto+label_test_human
        #print 'len(content_train):'+str(len(content_train))
        #print 'len(content_test):'+str(len(content_test))

        vectorizer_train=TfidfVectorizer(encoding='utf-8', decode_error='ignore', strip_accents='unicode', 
                                     token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', stop_words=stop_words, 
                                     lowercase=True, analyzer='word',max_features=500)# ngram_range=(1,2), 
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
        score_micro = f1_score(label_test, pred, average='micro')
        score_macro = f1_score(label_test, pred, average='macro')
        f1=(score_macro+score_micro)/2
        print 'f1: %0.20f'%f1
        f1_mean+=f1
    f1_mean = f1_mean/cross_fold
    print 'f1_mean: %0.20f'%f1_mean


f1(data,label,10)









