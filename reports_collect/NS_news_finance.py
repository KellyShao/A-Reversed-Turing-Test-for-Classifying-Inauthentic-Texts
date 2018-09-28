from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import time
import csv 

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
innitial_url = "https://www.forbes.com/search/?q=Forbes%20Earnings%20Preview#37d970ab279f7"
#driver = webdriver.Chrome("C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe")
#driver.implicitly_wait(10)

def selenium_load_page(url):
    driver.get(url)
    #click button
    python_button = driver.find_element_by_class_name('load-more-bg') #FHSU 

    cnt_ns = 0 
    while True:  
        for i in range(1,220): #5/page
            cnt_ns = cnt_ns+1
            print cnt_ns
            python_button.click() #click fhsu link
            #time.sleep(2) 
        single_page_url = driver.page_source
        result_soup = BeautifulSoup(single_page_url,"html.parser")
        result_title = result_soup.find_all(attrs={'ng-attr-target':'{{::$root.linkTarget(item.uri)}}'})
        cnt = 0
        for i in result_title:
            cnt=cnt+1
            #if(cnt%2 == 0):
            #print i.get('href') #对每项使用get函数取得tag属性值
            result_title_1.append(i.get('href'))
        if(len(result_title) > 1000): #2000
            print "succeed"
            break
        else: print "continue"
    return result_title_1




def decode_page(result_title):
    count = 0
    for single_result_title in result_title:
        if count<2000:
            count=count+1
            
        elif count>=2000:
            title = []
            publish_time = []
            author = []
            content = []
            word_num = []
            paragraph_num = []

            print count
            count=count+1
            headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0' }
            result_page = requests.get(single_result_title[1],headers=headers)
            news_soup = BeautifulSoup(result_page.text,"html.parser")
            #get metadata
            title_news = news_soup.find(class_="fs-headline speakable-headline color-body font-base")
            title_news_content = title_news.string
            title.append(title_news_content)
            time_news_content = news_soup.select('meta[property="og:updated_time"]')
            time_news = time_news_content[0].get("content")
            publish_time.append(time_news)

            #get content  
            content_1 = []
            content_whole = ''
            del_num = []      
            news_content = news_soup.find(class_="article-container color-body font-body")
            news_content_1 = news_content.find_all("p")
            for i in news_content_1:
                news_content_article = i.get_text()
                news_content_article = news_content_article.replace(u'\n',u'')
                news_content_article = news_content_article.replace(u'\r',u'')
                news_content_article = news_content_article.replace(u'\0xa',u'')
                content_1.append(news_content_article)
            len_para = len(content_1)
            for j in range(0,len_para-1):
                if content_1[j] == u"":
                    del_num.append(j)
            for k in del_num:
                del content_1[k]
            len_para = len(content_1)
            del content_1[len_para-1]
            del content_1[len_para-2]
            len_para = len(content_1)
            for n in range(0,len_para-1):
                content_whole = content_whole + content_1[n]

            content.append(content_whole)
            paragraph_num.append(len_para)
            word_num.append(len(content_whole.split()))
            ns_finance=pd.DataFrame({"title":title,"time":publish_time,"#word":word_num,"#para":paragraph_num,"content":content})
            ns_finance.to_csv('C:/Users/jzs1274/Desktop/ns_finance_csv_1.csv',header=False,mode='a',encoding="utf-8")
            



#result_title = selenium_load_page(innitial_url)
#url_ns_finance=pd.DataFrame(result_title)
#url_ns_finance.to_csv('C:/Users/lh.Lenovo-PC/Desktop/url_ns_finance_csv_1.csv',encoding="utf-8")

file_name = "C:/Users/jzs1274/Desktop/url_ns_finance_csv_1.csv"
data = pd.read_csv(file_name,index_col=False)
train_data = np.array(data)#np.ndarray()
result_title=train_data.tolist()#list


decode_page(result_title)
ns_finance=pd.DataFrame({"title":title,"time":publish_time,"#word":word_num,"#para":paragraph_num,"content":content})
print(ns_finance)
ns_finance.to_csv('C:/Users/lh.Lenovo-PC/DesktopC:/Users/jzs1274/Desktop/ns_finance_csv_1.csv',encoding="utf-8")