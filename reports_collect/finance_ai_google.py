import requests
import time
import pandas as pd
from bs4 import BeautifulSoup


#greenwich time finance
title = []
publish_time = []
content = []
word_num = []
paragraph_num = []
next_page_list = []
cnt_page = 0

first_page_url='https://www.google.com.hk/search?q=earnings+snapshot+ap+news+%22This+story+was+generated+by+Automated+Insights%22+site:https://www.apnews.com&lr=&safe=strict&hl=zh-CN&as_qdr=all&ei=KlSNW6K1OMXm5gKh3KSoBg&start=0&sa=N&biw=1396&bih=663'

def finance_news(single_result_url):
    result_page = requests.get(single_result_url)
    result_soup = BeautifulSoup(result_page.text,"html.parser")

    result_title = result_soup.find_all("h3")
    count = 0
    for single_result_title in result_title:
        print count
        count=count+1

        title_news_content = single_result_title.get_text()
        title.append(title_news_content)
        print title_news_content

        title_url = ''
        for j in range(0,len(title_news_content)):
            if title_news_content[j] ==' 'and title_news_content[j+1] !='-':
                title_url = title_url+'-'
            elif title_news_content[j] == ' ' and title_news_content[j+1] =='-':
                break
            elif title_news_content[j] !=' 'and title_news_content[j+1] !='-':
                title_url = title_url+title_news_content[j]

        news_link_1 =single_result_title.a
        news_link_2 = news_link_1.get("href")
        news_link = news_link_2[7:]
        print news_link
        for i in range(0,len(news_link)):
            if news_link[i] == '&' and news_link[i-1] =='t':
                break
            if news_link[i] == '&' and news_link[i-1] !='t':
                news_link_1 = news_link
                news_link = news_link_1[:i]+'/'+title_url+news_link_1[i:]
                print news_link
                break
        news_page = requests.get(news_link)
        news_soup = BeautifulSoup(news_page.text,"html.parser")
        #print news_soup

        #get metadata
        #title_news = news_soup.find(class_="topTitle")
        #title_news_content = title_news.get_text()
        #title_news_content.replace('\n','')
        

        time_news = news_soup.find(class_="updatedString")
        publish_time.append(time_news.get_text())

        news_content = news_soup.find_all("p")
        string_para = ''
        cnt_para = 0
        for single_news_content in news_content:
            if cnt_para<len(news_content)-2:    #delete the tag for zack investment
                single_para = single_news_content.string
                string_para = string_para + single_para
                cnt_para = cnt_para+1
            elif cnt_para==len(news_content)-2:
                break
        content.append(string_para)
        print cnt_para
        paragraph_num.append(cnt_para)
        word_num.append(len(string_para.split()))


#def page_turner(url):
#    global cnt_page
#    cnt_page = cnt_page+1
#    if cnt_page !=1 and cnt_page < 100:
#        result_page = requests.get(url)
#        result_soup = BeautifulSoup(result_page.text,"html.parser")

#        #next page
#        next_label = result_soup.find(class_="pn")
#        next_result_link = u'https://www.google.com.hk/'+next_label.get("href")
#        next_page_list.append(next_result_link)
#        print cnt_page
#        page_turner(next_result_link)
    
#    elif cnt_page ==1:
#        next_page_list.append(url)
#        print cnt_page
#        result_page = requests.get(url)
#        result_soup = BeautifulSoup(result_page.text,"html.parser")

#        #next page
#        next_label = result_soup.find(class_="pn")
#        next_result_link = u'https://www.google.com.hk/'+next_label.get("href")
#        page_turner(url)


def get_url():
    for i in range(0,49):
        next_url =u'https://www.google.com.hk/search?q=earnings+snapshot+ap+news+%22This+story+was+generated+by+Automated+Insights%22+site:https://www.apnews.com&lr=&safe=strict&hl=zh-CN&as_qdr=all&ei=KlSNW6K1OMXm5gKh3KSoBg&start='+str(i*10)+'&sa=N&biw=1396&bih=663'
        next_page_list.append(next_url)
    return next_page_list


next_page_list = get_url()
#for single_page_url in next_page_list:
#    sports_news(single_page_url)
#AI_sports=pd.DataFrame({"title":title,"time":publish_time,"#word":word_num,"#para":paragraph_num,"content":content})
#print(AI_sports)
#AI_sports.to_csv('C:/Users/lh.Lenovo-PC/Desktop/Aai_finance_google.csv',header=False,encoding="utf-8")


cnt_all = 0
for single_page_url in next_page_list:
    cnt_all = cnt_all+1
    print "page"+str(cnt_all)
    if cnt_all<2000 and cnt_all!=30:
        title = []
        publish_time = []
        content = []
        word_num = []
        paragraph_num = []
        finance_news(single_page_url)
        AI_finance=pd.DataFrame({"title":title,"time":publish_time,"#word":word_num,"#para":paragraph_num,"content":content})
        #print(AI_finance)
        AI_finance.to_csv('C:/Users/lh.Lenovo-PC/Desktop/AI_finance_google.csv',mode = 'a',header=False,encoding="utf-8")