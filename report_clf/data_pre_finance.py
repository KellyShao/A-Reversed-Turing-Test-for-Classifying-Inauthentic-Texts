import csv
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

ai_file = "raw_file/ai_finance_yahoo.csv"
ns_file = "raw_file/ns_finance_csv.csv"
human_file = "raw_file/human_finance_csv.csv"

auto_file = "raw_file/auto_finance_csv.csv"

#read file in a ditionary
#key is length of passage, value is content of passage
#this dictinary is one to many relationship
def read_file(file_name):
    content = open(file_name, 'rb')
    csv_reader = csv.reader(content)
    return csv_reader

ai_csv_reader = read_file(ai_file)
ns_csv_reader = read_file(ns_file)
human_csv_reader = read_file(human_file)

auto_csv_reader = read_file(auto_file)

    
def turn_to_dictionary(csv_reader,colunm_words,colunm_content):   
    dic = {}
    news_words = []
    news_content = []
    cnt = 0
    for row in csv_reader:
        if cnt!=0:
            news_words.append(row[colunm_words])
            content = row[colunm_content].replace('\n','')
            content = content.replace('\r','')
            content = content.replace('  ','')
            content = content.replace('\xe2\x80?',' ')
            news_content.append(content)
            if (row[colunm_words] not in dic):
                dic[row[colunm_words]] = []
            dic[row[colunm_words]].append(content)
        cnt = cnt+1
    return dic, news_words, news_content

ai_dic, ai_news_words, ai_news_content = turn_to_dictionary(ai_csv_reader,2,3)
ns_dic, ns_news_words, ns_news_content = turn_to_dictionary(ns_csv_reader,2,3)
human_dic, human_news_words, huamn_news_content = turn_to_dictionary(human_csv_reader,2,4)

def chose_and_dictionary(csv_reader,colunm_words,colunm_content):   
    dic = {}
    news_words_1 = []
    news_content_1 = []
    cnt = 0
    for row in csv_reader:
        if cnt!=0:
            news_words_1.append(row[colunm_words])
            news_content_1.append(row[colunm_content])
        cnt = cnt+1
    news_words, anews_words, news_content, anews_content = train_test_split(news_words_1, news_content_1, test_size=0.5)
    for i in range(0,len(news_words)):
        if (news_words[i] not in dic):
            dic[news_words[i]] = []
        dic[news_words[i]].append(news_content[i])
    return dic, news_words, news_content

auto_dic, auto_news_words, auto_news_content = chose_and_dictionary(auto_csv_reader,2,3)

#draw picture
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

print 'paint:'
ai_freq = 'ai_freq'
ns_freq = 'ns_freq'
human_freq = 'human_freq'
#draw_freq(ai_dic, ai_freq)
#draw_freq(ns_dic, ns_freq)
#draw_freq(human_dic,human_freq)

#auto_freq = 'auto_freq'
#draw_freq(auto_dic, auto_freq)

#using n sigma to find outliers
#remove the outliers from a list
def filter_extrem_nsigma(news_words, dictionary, n):
    array_words = np.array(news_words, np.float)
    array_mean = array_words.mean() 
    array_std = array_words.std()
    array_max = array_mean+n*array_std
    array_min = array_mean-n*array_std
    after_filter_words = []
    for words in dictionary.keys():
        if int(words)>array_min and int(words)<array_max:
            after_filter_words.append(words)
    return after_filter_words, array_mean, array_std
   
print 'filt data'
ai_filter_words, ai_filter_mean, ai_filter_std  = filter_extrem_nsigma(ai_news_words, ai_dic, 1)
ns_filter_words, ns_filter_mean, ns_filter_std  = filter_extrem_nsigma(ns_news_words, ns_dic, 1)
human_filter_words, human_filter_mean, human_filter_std  = filter_extrem_nsigma(human_news_words, human_dic, 1)
#print 'ai:', ai_filter_mean-3*ai_filter_std, ai_filter_mean, ai_filter_mean+3*ai_filter_std
#print 'ns:', ns_filter_mean-3*ns_filter_std, ns_filter_mean, ns_filter_mean+3*ns_filter_std
#print 'human:', human_filter_mean-3*human_filter_std, human_filter_mean, human_filter_mean+3*human_filter_std
print 'ai:', ai_filter_mean-1*ai_filter_std, ai_filter_mean, ai_filter_mean+1*ai_filter_std,'\nai words length:',len(ai_filter_words)
print 'ns:', ns_filter_mean-1*ns_filter_std, ns_filter_mean, ns_filter_mean+1*ns_filter_std,'\nns words length:',len(ns_filter_words)
print 'human:', human_filter_mean-1*human_filter_std, human_filter_mean, human_filter_mean+1*human_filter_std,'\nhuman words length:',len(human_filter_words)


auto_filter_words, auto_filter_mean, auto_filter_std  = filter_extrem_nsigma(auto_news_words, auto_dic, 1)
print 'auto:', auto_filter_mean-1*auto_filter_std, auto_filter_mean, auto_filter_mean+1*auto_filter_std,'\nauto words length:',len(auto_filter_words)

#get the length of dictionary
#cannot use values directly since the dictionary is one to many relationship
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

ai_after_filter_dictionary, ai_after_news_words, ai_after_news_content, ai_after_news_length, ai_whole_length= filter_articles(ai_filter_words, ai_dic)
ns_after_filter_dictionary, ns_after_news_words, ns_after_news_content, ns_after_news_length, ns_whole_length= filter_articles(ns_filter_words, ns_dic)
human_after_filter_dictionary, human_after_news_words, human_after_news_content, human_after_news_length, human_whole_length = filter_articles(human_filter_words, human_dic)
print "ai length:",ai_whole_length
print "ns length:",ns_whole_length
print "human length:",human_whole_length

auto_after_filter_dictionary, auto_after_news_words, auto_after_news_content, auto_after_news_length, auto_whole_length= filter_articles(auto_filter_words, auto_dic)
print "auto length:",auto_whole_length

print 'draw figure after filter'
ai_freq_after = 'ai_freq_after'
ns_freq_after = 'ns_freq_after'
human_freq_after = 'human_freq_after'
#draw_freq(ai_after_filter_dictionary, ai_freq_after)
#draw_freq(ns_after_filter_dictionary, ns_freq_after)
#draw_freq(human_after_filter_dictionary, human_freq_after)

auto_freq_after = 'auto_freq_after'
#draw_freq(auto_after_filter_dictionary, auto_freq_after)

ai_finance=pd.DataFrame({"ai":ai_after_news_content})
ai_finance.to_csv('C:/Users/jzs1274/Desktop/ai_finance_after_filter.csv',header=False)#encoding="utf-8")
ns_finance=pd.DataFrame({"ns":ns_after_news_content})
ns_finance.to_csv('C:/Users/jzs1274/Desktop/ns_finance_after_filter.csv',header=False)#,encoding="utf-8")
human_finance=pd.DataFrame({"human":human_after_news_content})
human_finance.to_csv('C:/Users/jzs1274/Desktop/human_finance_after_filter.csv',header=False)#,encoding="utf-8")
auto_finance=pd.DataFrame({"auto":auto_after_news_content})
auto_finance.to_csv('C:/Users/jzs1274/Desktop/auto_finance_after_filter.csv',header=False)#,encoding="utf-8")
#print(AI_finance)

























##based on the mean and length, using them to filter other dataset
#def filter_other_articles(other_dictionary,mean,std,length,nsigma):
#    global after_filter
#    length_each = round(length/2)
#    keys = other_dictionary.keys()
#    keys.sort()
#    #find nearest key
#    flag = 1000
#    for key in range(0,len(other_dictionary.keys())-1):
#        if abs(int(other_dictionary.keys()[key])-mean)<flag:
#            flag = abs(int(other_dictionary.keys()[key])-mean)
#            nearest_key = other_dictionary.keys()[key]
#    print nearest_key
#    after_filter = []
#    after_filter = after_filter + other_dictionary[nearest_key]
#    #left side
#    left_count = round(len(after_filter)/2)
#    right_count_1 = len(after_filter)-round(len(after_filter)/2)
#    nearest_key_left = nearest_key
#    while left_count<length_each and int(nearest_key_left)>0:#(mean-(nsigma)*std):
#        if str(int(nearest_key_left)-1) in keys:
#            nearest_key_left = int(nearest_key_left)-1
#            after_filter = after_filter+other_dictionary[str(nearest_key_left)]
#            left_count = len(after_filter)
#        else:
#            nearest_key_left = int(nearest_key_left)-1
#    print len(after_filter)
#    #right side
#    right_count = int(len(after_filter))-right_count_1
#    nearest_key_right = nearest_key
#    while right_count<length_each and int(nearest_key_right)<1000:#<(mean+(nsigma+2)*std):
#        if str(int(nearest_key_right)+1) in keys:
#            nearest_key_right = int(nearest_key_right)+1
#            after_filter = after_filter+other_dictionary[str(nearest_key_right)]
#            right_count = len(after_filter)
#        else:
#            nearest_key_right = int(nearest_key_right)+1
#    print len(after_filter)

#human_after_filter_passage = filter_other_articles(human_dic,ai_filter_mean,ai_filter_std, ai_after_filter_length,3)

#ai_content = [row[3] for row in csv_reader_ai]
#del ai_content[0]
#ai_news_len = [[row[2] for row in csv_reader_ai]]
#del ai_news_len[0]
#ai_length = len(ai_content)
#ai_label = np.ones(ai_length)
#ai_label = ai_label.tolist()
