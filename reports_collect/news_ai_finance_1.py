import requests
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

first_page_url='https://www.greenwichtime.com/search/?action=search&searchindex=solr&query=Earnings+Snapshot&page=1'

def finance_news(single_result_url):
    result_page = requests.get(single_result_url, headers={'Connection':'close'})
    result_soup = BeautifulSoup(result_page.text,"html.parser")

    #link on this page
    result_title = result_soup.find_all(class_="gsa-item gsa-item-regular no-photo")
    count = 0
    for single_result_title in result_title:
        print count
        count=count+1
        single_result_link = single_result_title.find(class_="headline")
        news_link = u"https://www.greenwichtime.com"+single_result_link.a.get("href")
        news_page = requests.get(news_link)
        news_soup = BeautifulSoup(news_page.text,"html.parser")

        #get metadata
        title_news = news_soup.find(class_="headline entry-title")
        title_news_content = title_news.get_text()
        title.append(title_news_content)
        print title_news_content
        time_news_content = news_soup.find(itemprop="datePublished")
        time_news = time_news_content.get("content")
        publish_time.append(time_news)

        #get content        
        news_content = news_soup.find_all("p")
        string_para = ''
        cnt_para = 0
        for single_news_content in news_content:
            if cnt_para>0:    #delete the tag for zack investment
                single_para = single_news_content.string
                string_para = string_para + single_para
                cnt_para = cnt_para+1
            else:cnt_para = cnt_para+1
        content.append(string_para)
        paragraph_num.append(cnt_para)
        word_num.append(len(string_para.split()))

#def page_turner(url):
#    global cnt_page
#    cnt_page = cnt_page+1
#    if cnt_page != 1 and cnt_page != 36 and cnt_page != 37 and cnt_page <130:#160
#        next_result_link = u"https://www.greenwichtime.com/search/?action=search&searchindex=solr&query=Earnings+Snapshot&page="+str(cnt_page)
#        next_page_list.append(next_result_link)
#        print cnt_page
#        page_turner(next_result_link)
#    elif cnt_page == 1:
#        page_turner(url)
#    elif cnt_page ==36:

def page_turner(url):
    #global cnt_page
    for i in range(30,130):
        next_result_link = u"https://www.greenwichtime.com/search/?action=search&searchindex=solr&query=Earnings+Snapshot&isRedesign=&facet_type=Article&facet_section=Business&page="+str(i)
        next_page_list.append(next_result_link)


page_turner(first_page_url)
cnt_all = 30
for single_page_url in next_page_list:
    cnt_all = cnt_all+1
    print "page"+str(cnt_all)
    if cnt_all<130:
        title = []
        publish_time = []
        content = []
        word_num = []
        paragraph_num = []
        finance_news(single_page_url)
        AI_finance=pd.DataFrame({"title":title,"time":publish_time,"#word":word_num,"#para":paragraph_num,"content":content})
        #print(AI_finance)
        AI_finance.to_csv('human_finance_csv.csv',mode='a',header=False,encoding="utf-8")
