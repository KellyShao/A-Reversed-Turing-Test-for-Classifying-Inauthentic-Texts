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

first_page_url='http://www.oleantimesherald.com/search/?sd=desc&l=100&sort=relevance&f=html&t=article%2Cvideo%2Cyoutube%2Ccollection&app%5B0%5D=editorial&nsa=eedition&q=This+story+was+generated+by+Automated+Insights+using+data+from+Zacks+Investment+Research&o=0'

def finance_news(single_result_url):
    result_page = requests.get(single_result_url)

    result_soup = BeautifulSoup(result_page.text,"html.parser")
    result_title = result_soup.find_all(class_='tnt-asset-link')
    count = 0
    for single_result_title in result_title:
        if count<97 and count!=16 and count!=77 :
            print count
            count=count+1

            news_link = u'http://www.oleantimesherald.com'+ single_result_title.get('href')
        
            news_page = requests.get(news_link)
            news_soup = BeautifulSoup(news_page.text,"html.parser")
            #print news_soup

            #get metadata
            title_news = news_soup.find(class_="headline")
            title_news_content = title_news.get_text()
            title_news_content.replace(u'\n','')
            title_news_content.replace(u'\r','')
            title.append(title_news_content)
        

            #time_news = news_soup.find(class_="list-inline")
            #publish_time.append(time_news.get("datetime"))
            time_news = news_soup.select('meta[itemprop="datePublished"]')
            publish_time.append(time_news[0].get("content"))

            news_content = news_soup.find_all(class_="subscriber-preview")+news_soup.find_all(class_="subscriber-only")
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
            print 'para'+str(cnt_para)
            paragraph_num.append(cnt_para)
            word_num.append(len(string_para.split()))
        elif count==16 or count == 77:
            count=count+1
        else:break


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
    for i in range(0,10):
        next_url =u'http://www.oleantimesherald.com/search/?sd=desc&l=100&sort=relevance&f=html&t=article%2Cvideo%2Cyoutube%2Ccollection&app%5B0%5D=editorial&nsa=eedition&q=This+story+was+generated+by+Automated+Insights+using+data+from+Zacks+Investment+Research&o='+str(100*i)
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
    if cnt_all==10:
        title = []
        publish_time = []
        content = []
        word_num = []
        paragraph_num = []
        finance_news(single_page_url)
        AI_finance=pd.DataFrame({"title":title,"time":publish_time,"#word":word_num,"#para":paragraph_num,"content":content})
        #print(AI_finance)
        AI_finance.to_csv('C:/Users/lh.Lenovo-PC/Desktop/ai_finance_olean.csv',mode = 'a',header=False,encoding="utf-8")