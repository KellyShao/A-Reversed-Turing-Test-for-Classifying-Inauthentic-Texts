from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
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
result_title_1 = []
cnt_page = 0

#Marketwatch human finance
innitial_url = "https://www.google.com/search?q=This+information+comes+from+the+USGS+Earthquake+Notification+Service+and+this+post+was+created+by+an+algorithm+written+by+the+author.&rlz=1C1CHWL_zh-cnCN799US800&source=lnms&tbm=nws&sa=X&ved=0ahUKEwiRqvHs1PrcAhUyq1kKHXBDAGAQ_AUICigB&biw=1396&bih=663"

#chromeOptions = webdriver.ChromeOptions()

## 设置代理
#chromeOptions.add_argument("--proxy-server=http://202.20.16.82:10152")
driver = webdriver.Chrome("C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe")#,chrome_options = chromeOptions)#, chrome_options=options)
driver.implicitly_wait(10)

cnt_page = 0
def selenium_load_page(url):
    global result_title_1
    global cnt_page
    cnt_page = cnt_page+1
    
    
    #elif cnt_page<=5:
    #    cnt_page = cnt_page+1
    #    print cnt_page
    #    driver.get(url)
    #    python_button = driver.find_element_by_class_name('pn') #FHSU
    #    python_button.click()
    #    time.sleep(10)
    if cnt_page == 1:
        print cnt_page
        #driver.get(url)
        #single_page_url = driver.page_source
        #result_soup = BeautifulSoup(single_page_url,"html.parser")
        #result_title = result_soup.find_all(class_="top NQHJEb dfhHve")
        #result_station = result_soup.find_all(class_="xQ82C e8fRJf")
        #len_url = len(result_station)
        #for i in range(0,len_url-1):
        #    aaa = result_station[i]
        #    if result_station[i].string == "Los Angeles Times":
        #        result_title_1.append(result_title[i].get("href"))
        #page_finance=pd.DataFrame(result_title_1)
        #page_finance.to_csv('C:\Users\lh.Lenovo-PC\Desktop\url_quake_csv.csv',mode="a",header=False,encoding="utf-8")
        #result_title_1 = []
        #time.sleep(20)
        selenium_load_page(url)

    if cnt_page!= 1 and cnt_page < 16:
        print cnt_page
        #driver.get(url)
        single_page_url = driver.page_source
        result_soup = BeautifulSoup(single_page_url,"html.parser")
        result_title = result_soup.find_all(class_="top NQHJEb dfhHve")
        result_station = result_soup.find_all(class_="xQ82C e8fRJf")
        len_url = len(result_station)
        for i in range(0,len_url-1):
            aaa = result_station[i]
            if result_station[i].string == "Los Angeles Times":
                result_title_1.append(result_title[i].get("href"))
        page_finance=pd.DataFrame(result_title_1)
        page_finance.to_csv('C:\Users\lh.Lenovo-PC\Desktop\url_quake_csv.csv',mode="a",header=False,encoding="utf-8")
        result_title_1 = []
        time.sleep(20)
            
        next_button = result_soup.find(class_="pn")
        next_url = "https://www.google.com/"+next_button.get("href")
        selenium_load_page(next_url)

  



def decode_page(result_title):
    count = 0
    for single_result_title in result_title:
        if count == 0:
           count=count+1
        elif count>2100:
            break
        else: 
            print count
            count=count+1
            news_link = u"https://www.marketwatch.com"+single_result_title.get("href")
            news_page = requests.get(news_link)
            requests.adapters.DEFAULT_RETRIES = 5
            time.sleep(1)
            news_soup = BeautifulSoup(news_page.text,"html.parser")

            #get metadata
            title_news = news_soup.find(id="article-headline")
            title_news_content = title_news.string
            title.append(title_news_content)
            #print title_news_content
            #2018-08-11T10:48:00-04:00
            #Aug 11, 2018 10:48 a.m. ET
            #time_news_content = news_soup.find(id="published-timestamp")
            time_news_content = news_soup.select('meta[name="parsely-pub-date"]')
            time_news = time_news_content[0].get("content")
            publish_time.append(time_news)
            author_news_content = news_soup.find(class_="module-header")
            author_news = author_news_content.get_text()
            author_news_strip = author_news.replace(' ','')
            author.append(author_news_strip)

            #get content        
            news_content = news_soup.find(id="article-body")
            news_content_1 = news_content.find_all("p")
            news_content_article = news_content_1[0].get_text()

            content.append(news_content_article)
            paragraph_num.append("1")
            word_num.append(len(news_content_article.split()))



result_title = selenium_load_page(innitial_url)
decode_page(result_title)
human_finance=pd.DataFrame({"title":title,"time":publish_time,"#word":word_num,"#para":paragraph_num,"content":content,"author":author})
print(human_finance)
human_finance.to_csv('C:/Users/lh.Lenovo-PC/Desktop/huamn_finance_csv.csv',encoding="utf-8")