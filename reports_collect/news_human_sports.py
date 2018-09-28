from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
import pandas as pd
import requests
import time

title = []
publish_time = []
author = []
content = []
word_num = []
paragraph_num = []
next_page_list = []
cnt_page = 0

#reuters sports
innitial_url = "https://www.reuters.com/search/news?sortBy=&dateRange=&blob=mlb+roundup"
driver = webdriver.Chrome("C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe")
driver.implicitly_wait(10)

def selenium_load_page(url):
    driver.get(url)

    #click button
    while True:
        if driver.find_element_by_class_name("search-result-more-txt").text == u"LOAD MORE RESULTS" :
            python_button = driver.find_element_by_class_name("search-result-more-txt") #FHSU
            python_button.click() #click fhsu link
            time.sleep(5)
        else:
            break

    #get source code
    time.sleep(10)
    single_page_url = driver.page_source
    result_soup = BeautifulSoup(single_page_url,"html.parser")
    result_title = result_soup.find_all(class_="search-result-media")
    return result_title
    

def decode_page(result_title):
    count = 0
    for single_result_title in result_title:
        print count
        count=count+1
        news_link = u"https://www.reuters.com"+single_result_title.a.get("href")
        news_page = requests.get(news_link)
        news_soup = BeautifulSoup(news_page.text,"html.parser")

        #get metadata
        title_news = news_soup.find(class_="ArticleHeader_headline")
        title_news_content = title_news.string
        print title_news_content
        time_news_content = news_soup.select('meta[name="analyticsAttributes.articleDate"]')
        time_news = time_news_content[0].get("content")
        author_news_content = news_soup.find(class_="BylineBar_byline")
        author_news = author_news_content.get_text()

        #get content        
        news_content = news_soup.find(class_="StandardArticleBody_body")
        news_content_1 = news_content.find_all("p")
        cnt_para = 0
        para_content = ""
        for single_news_content in news_content_1:
            single_para = single_news_content.get_text()
            if(len(single_para.split()) > 6):
                cnt_para = cnt_para+1
                para_content = para_content+single_para
            else:
                content.append(para_content)
                paragraph_num.append(cnt_para)
                word_num.append(len(para_content.split()))
                title.append(title_news_content)
                publish_time.append(time_news)
                author.append(author_news)
                cnt_para = 0
                para_content = ""
                #news_content_article = news_content_1[0].get_text()




links = selenium_load_page(innitial_url)
decode_page(links)
human_sports=pd.DataFrame({"title":title,"time":publish_time,"#word":word_num,"#para":paragraph_num,"content":content,"author":author})
print(human_sports)
human_sports.to_csv('C:/Users/lh.Lenovo-PC/Desktop/huamn_sports_csv.csv',encoding="utf-8")