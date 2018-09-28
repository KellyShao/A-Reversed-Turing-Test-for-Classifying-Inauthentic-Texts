import requests
import pandas as pd
from bs4 import BeautifulSoup


#yahoo！ finance
title = []
publish_time = []
content = []
word_num = []
paragraph_num = []
next_page_list = []
cnt_page = 0

first_page_url='https://finance.search.yahoo.com/search;_ylt=AwrC1C1T_GhbIj0AmwKTmYlQ;_ylu=X3oDMTEyNmU4bjVjBGNvbG8DYmYxBHBvcwMxBHZ0aWQDVklEQzFfMQRzZWMDc2M-?p=This+story+was+generated+by+Automated+Insights+using+data+from+Zacks+Investment+Research.&pz=10&fr=yfp-t&fr2=sb-top-finance.search&bct=0&b=1&pz=10&bct=0&xargs=0'

def finance_news(single_result_url):
    result_page = requests.get(single_result_url)
    result_soup = BeautifulSoup(result_page.text,"html.parser")

    #link on this page
    result_title = result_soup.find_all(class_="dd algo FinanceAlgoMRDS")
    count = 0
    for single_result_title in result_title:
        print count
        count=count+1
        single_result_link = single_result_title.find(class_="fz-m")
        news_link = single_result_link.get("href")
        news_page = requests.get(news_link)
        news_soup = BeautifulSoup(news_page.text,"html.parser")

        #get metadata
        news_content = news_soup.find_all(class_="canvas-atom canvas-text Mb(1.0em) Mb(0)--sm Mt(0.8em)--sm")
        title_news_content = news_soup.title.string
        title.append(title_news_content)
        print title_news_content
        time_news_content = news_soup.find(itemprop="datePublished")
        time_news = time_news_content.get("datetime")
        publish_time.append(time_news)

        #get content
        string_para = ''
        cnt_para = 0
        for single_news_content in news_content:
            if cnt_para<len(news_content)-2:    #delete the tag for zack investment
                single_para = single_news_content.get("content")
                string_para = string_para + single_para
                cnt_para = cnt_para+1
            else: break
        content.append(string_para)
        paragraph_num.append(cnt_para)
        word_num.append(len(string_para.split()))

def page_turner(url):
    global cnt_page
    cnt_page = cnt_page+1
    if cnt_page !=1 and cnt_page <38:
        result_page = requests.get(url)
        result_soup = BeautifulSoup(result_page.text,"html.parser")

        #next page
        next_label = result_soup.find(class_="next fc-14th")
        next_result_link = next_label.get("href")
        next_page_list.append(next_result_link)
        print cnt_page
        page_turner(next_result_link)
    elif cnt_page ==1:
        page_turner(url)


page_turner(first_page_url)
for single_page_url in next_page_list:
    finance_news(single_page_url)
AI_finance=pd.DataFrame({"title":title,"time":publish_time,"#word":word_num,"#para":paragraph_num,"content":content})
print(AI_finance)
AI_finance.to_csv('C:/Users/lh.Lenovo-PC/Desktop/AI_finance_csv.csv',encoding="utf-8")



