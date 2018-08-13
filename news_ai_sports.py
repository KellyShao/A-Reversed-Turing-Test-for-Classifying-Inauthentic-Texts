import requests
import pandas as pd
from bs4 import BeautifulSoup

#tucson sports
title = []
publish_time = []
content = []
word_num = []
paragraph_num = []
next_page_list = []
cnt_page = 0

first_page_url='https://tucson.com/search/?sd=desc&l=25&s=start_time&f=html&t=article%2Cvideo%2Cyoutube%2Ccollection&app%5B0%5D=editorial&nsa=eedition&q=data+from+and+in+cooperation+with+MLB+Advanced+Media+and+Minor+League+Baseball&o=0'

def sports_news(single_result_url):
    result_page = requests.get(single_result_url)
    result_soup = BeautifulSoup(result_page.text,"html.parser")

    #link on this page
    result_title = result_soup.find_all(class_="card-container")
    count = 0
    for single_result_title in result_title:
        print count
        count=count+1
        single_result_link = single_result_title.find(class_="tnt-asset-link")
        news_link = u"https://tucson.com"+single_result_link.get("href")
        news_page = requests.get(news_link)
        news_soup = BeautifulSoup(news_page.text,"html.parser")

        #get metadata
        title_news = news_soup.find(property="og:title")
        title_news_content = title_news.get("content")
        print title_news_content
        title.append(title_news_content)
        time_news_content = news_soup.find(itemprop="datePublished")
        time_news = time_news_content.get("content")
        publish_time.append(time_news)

        #get content
        string_para=''
        cnt_para = 0
        news_preview = news_soup.find_all(class_="subscriber-preview")
        for single_news_preview in news_preview:
            single_para = single_news_preview.p.string
            string_para = string_para + single_para
            cnt_para = cnt_para+1
        news_only = news_soup.find_all(class_="subscriber-only")
        for single_news_only in news_only:
            print single_news_only
            if cnt_para<(len(news_preview)+len(news_only)-2):
                single_para = single_news_only.p.string
                print single_para
                string_para = string_para + single_para
                cnt_para = cnt_para+1
            else: break
        content.append(string_para)
        paragraph_num.append(cnt_para)
        word_num.append(len(string_para.split()))

def page_turner(url):
    global cnt_page
    cnt_page = cnt_page+1
    if cnt_page !=1 and cnt_page < 100:
        result_page = requests.get(url)
        result_soup = BeautifulSoup(result_page.text,"html.parser")

        #next page
        next_label = result_soup.find(class_="next")
        next_result_link = u"https://tucson.com"+next_label.a.get("href")
        next_page_list.append(next_result_link)
        print cnt_page
        page_turner(next_result_link)
    elif cnt_page ==1:
        page_turner(url)

page_turner(first_page_url)
for single_page_url in next_page_list:
    sports_news(single_page_url)
AI_sports=pd.DataFrame({"title":title,"time":publish_time,"#word":word_num,"#para":paragraph_num,"content":content})
print(AI_sports)
AI_sports.to_csv('C:/Users/lh.Lenovo-PC/Desktop/AI_sports_csv.csv',encoding="utf-8")