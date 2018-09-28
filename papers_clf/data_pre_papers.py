#data preprocessing

import csv
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

sci_file = 'cs_papers/papers_sci.csv'
scigen_file = 'cs_papers/papers_scigen.csv'

def read_file(file_name):
    content = open(file_name, 'rb')
    csv_reader = csv.reader(content)
    return csv_reader

sci_csv = read_file(sci_file)
scigen_csv = read_file(scigen_file)


def turn_to_dictionary(csv_reader):   
    dic = {}
    news_words = []
    news_content = []
    cnt = 0
    for row in csv_reader:
        content = row[1]
        content = content.replace('\xa0','')
        words = len(content.split())
        news_words.append(words)
        news_content.append(content)
        if (words not in dic):
            dic[words] = []
        dic[words].append(content)
    return dic, news_words, news_content

sci_dic, sci_news_words, sci_content = turn_to_dictionary(sci_csv)


def split_turn_to_dictionary(csv_reader):   
    dic = {}
    news_words = []
    news_content = []
    cnt = 0
    for row in csv_reader:
        content1 = row[1]
        content=content1[160:]
        content = content.replace('\x0c','')
        words = len(content.split())
        news_words.append(words)
        news_content.append(content)
        if (words not in dic):
            dic[words] = []
            dic[words].append(content)
        else:
            cnt = 0
            for i in dic[words]:
                cnt+=1
                if i[:30]==content[:30]:
                    break
                elif i[:30]!=content[:30] and cnt==len(dic[words]):
                    dic[words].append(content)
    return dic, news_words, news_content

scigen_dic, scigen_news_words, scigen_content = split_turn_to_dictionary(scigen_csv)


def draw_freq(dictionary,fig_name):
    news_axis_x = []
    news_axis_y = []
    for key in range(0,len(dictionary.keys())-1):
        key_value = dictionary[dictionary.keys()[key]]
        key_length = len(key_value)
        news_axis_x.append(int(dictionary.keys()[key])) 
        news_axis_y.append(key_length)
    plt.scatter(news_axis_x,news_axis_y)
    plt.savefig(fig_name, bbox_inches="tight")

sci_freq = 'sci_freq'
scigen_freq = 'scigen_freq'
#draw_freq(sci_dic,sci_freq)
#draw_freq(scigen_dic,scigen_freq)


def sci_filter_extrem(news_words, dictionary, n):
    array_words = np.array(news_words, np.float)
    array_mean = array_words.mean() 
    array_std = array_words.std()
    array_max = array_mean+n*array_std
    array_min = array_mean-n*array_std
    after_filter_words = []
    for words in dictionary.keys():
        if int(words)>1400 and int(words)<2700:
            after_filter_words.append(words)
    return after_filter_words, array_mean, array_std

sci_filter_words, sci_filter_mean, sci_filter_std = sci_filter_extrem(sci_news_words, sci_dic, 1)
print sci_filter_mean

def scigen_filter_extrem(news_words, dictionary, n):
    array_words = np.array(news_words, np.float)
    array_mean = array_words.mean() 
    array_std = array_words.std()
    array_max = array_mean+n*array_std
    array_min = array_mean-n*array_std
    after_filter_words = []
    for words in dictionary.keys():
        if int(words)>1500 and int(words)<2700:
            after_filter_words.append(words)
    return after_filter_words, array_mean, array_std

scigen_filter_words, scigen_filter_mean, scigen_filter_std = scigen_filter_extrem(scigen_news_words, scigen_dic, 1)
print scigen_filter_mean

def get_dictionary_length(dictionary):
    origin_list = dictionary.values()
    merged_list = list(itertools.chain.from_iterable(origin_list))
    length_dictionary = len(merged_list)
    return length_dictionary

#delete outlier's value
def filter_articles(after_filter_words,dictionary):
    dic = {}
    after_news_words = []
    after_news_content = []
    after_news_length = []
    for key in dictionary:
        if key in after_filter_words:
            dic[key] = []
            dic[key] = dic[key]+(dictionary[key])
            content= dictionary[key]
            after_news_content = after_news_content+dictionary[key]
            after_news_words.append(key)
            after_news_length.append(len(dictionary[key]))
    whole_length = get_dictionary_length(dic)
    return dic,after_news_words, after_news_content, after_news_length,whole_length

sci_after_filter_dictionary, sci_after_news_words, sci_after_news_content, sci_after_news_length, sci_whole_length= filter_articles(sci_filter_words, sci_dic)
scigen_after_filter_dictionary, scigen_after_news_words, scigen_after_news_content, scigen_after_news_length, scigen_whole_length= filter_articles(scigen_filter_words, scigen_dic)
print "sci length:",sci_whole_length
print "scigen length:",scigen_whole_length

sci_freq_after = 'sci_freq_after'
scigen_freq_after = 'scigen_freq_after'
draw_freq(sci_after_filter_dictionary, sci_freq_after)
draw_freq(scigen_after_filter_dictionary, scigen_freq_after)

sci_finance=pd.DataFrame(sci_after_news_content)
sci_finance.to_csv('cs_papers/sci_after_filter.csv',header=False)#encoding="utf-8")
scigen_finance=pd.DataFrame(scigen_after_news_content)
scigen_finance.to_csv('cs_papers/scigen_after_filter.csv',header=False)#,encoding="utf-8")
