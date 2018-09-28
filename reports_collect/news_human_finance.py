from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
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

#Marketwatch human finance
innitial_url = "https://www.marketwatch.com/newsviewer"
driver = webdriver.Chrome("C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe")
driver.implicitly_wait(10)

def selenium_load_page(url):
    driver.get(url)
    #click button
    python_button = driver.find_element_by_name('pulse') #FHSU
    python_button.click() #click fhsu link
    time.sleep(10)

    #loading news
    #while True:
    #    cnt_mouse = 0
    #    pyautogui.moveTo(300,600)
    #    #while cnt_mouse==50:
    #    #    cnt_mouse = cnt_mouse+1
    #    #    win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL,0,0,-1)
    #    #beautifulsoup decode
    #    pyautogui.scroll(clicks=800)
    #    time.sleep(10)
    #    print "end sleep"
    #    single_page_url = driver.page_source
    #    result_soup = BeautifulSoup(single_page_url,"html.parser")
    #    result_title = result_soup.find_all(target="_blank",class_="read-more",rel="nofollow")
    #    if(len(result_title) ==2500): break

    while True:   
        print "scroll"
        time.sleep(420)
        print "stop"
        time.sleep(10)
        single_page_url = driver.page_source
        result_soup = BeautifulSoup(single_page_url,"html.parser")
        result_title = result_soup.find_all(target="_blank",class_="read-more",rel="nofollow")
        if(len(result_title) > 2000): 
            print "succeed"
            break
        else: print "continue"
    return result_title




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
            news_content_article = news_content_article.replace('\n','')
            content.append(news_content_article)
            paragraph_num.append("1")
            word_num.append(len(news_content_article.split()))



result_title = selenium_load_page(innitial_url)
decode_page(result_title)
human_finance=pd.DataFrame({"title":title,"time":publish_time,"#word":word_num,"#para":paragraph_num,"content":content,"author":author})
print(human_finance)
human_finance.to_csv('human_finance_csv.csv',header=False,encoding="utf-8")