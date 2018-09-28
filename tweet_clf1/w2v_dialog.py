import csv
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_extraction import text
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve,auc,f1_score
import matplotlib.pyplot as plt
from gensim.models import word2vec
from gensim import corpora
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_short
from gensim.parsing.preprocessing import strip_non_alphanum

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
        row1 = unicode(row[2], errors='ignore')
        row_new1 = remove_stopwords(row1)
        row_new1 = strip_numeric(row_new1)
        #row_new = strip_non_alphanum(row_new)   
        row_new1 = strip_short(row_new1,minsize = 3)
        human.append(row_new1)
        row2 = unicode(row[3], errors='ignore')
        row_new2 = remove_stopwords(row2)
        row_new2 = strip_numeric(row_new2)
        #row_new = strip_non_alphanum(row_new)   
        row_new2 = strip_short(row_new2,minsize = 3)
        machine.append(row_new2)

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


def build_dict(data,stop_words):
    texts = [[word for word in document.lower().split() if word not in stop_words]
             for document in data]
    dictionary = corpora.Dictionary(texts)
    return texts,dictionary

content, dict = build_dict(data,stop_words)
model = word2vec.Word2Vec(content, min_count=1,iter=100,alpha=0.02)
model.save("sci_scigen_passage.model")

def f1(content, label, cross_fold):
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

        #build matrics
        train = []
        test = []
        w2v_model =word2vec.Word2Vec.load("sci_scigen_passage.model")
        print "model load successed"
        for each_train in content_train:
            word_num = 0
            vector = np.zeros(100)
            for word in each_train:
                if unicode(word) in w2v_model:
                    vector += w2v_model[unicode(word)]
                    word_num += 1
            vector=vector/word_num
            train.append(vector)
        print "train builed"
        for each_test in content_test:
            word_num = 0
            vector = np.zeros(100)
            for word in each_test:
                if unicode(word) in w2v_model:
                    vector += w2v_model[unicode(word)]
                    word_num += 1
            vector=vector/word_num
            test.append(vector)
        print "test build"

        clf = svm.SVC(kernel='linear')#, probability=True)
        clf_res = clf.fit(np.nan_to_num(train), label_train)
        pred =  clf_res.predict(np.nan_to_num(test))
        #predict_prob = clf_res.predict_proba(tfidf_test)[:,1]
        score_micro = f1_score(label_test, pred, average='micro')
        score_macro = f1_score(label_test, pred, average='macro')
        f1=(score_macro+score_micro)/2
        print 'f1: %0.20f'%f1
        f1_mean+=f1
    f1_mean = f1_mean/cross_fold
    print 'f1_mean: %0.20f'%f1_mean

f1(content, label, 10)




