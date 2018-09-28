from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import pandas as pd
import requests
import time

content = []
url_list = []
cnt_page = 0

#SCIgen - An Automatic CS Paper Generator
#https://pdos.csail.mit.edu/archive/scigen/

innitial_url = "https://pdos.csail.mit.edu/archive/scigen/"
driver = webdriver.Chrome("C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe")
driver.implicitly_wait(10)

driver.get(innitial_url)
time.sleep(10)

def decode_page(single_paper):
    content = []
    single_page_url = driver.page_source
    result_soup = BeautifulSoup(single_page_url,"html.parser")
    result_title = result_soup.get_text()
    result_title = result_title.encode('utf-8')
    result = result_title.replace('0xa','')
    result = result_title.replace('0xc2','')
    result = result.replace('\n','')
    result = result.replace('\r','')
    result = result.replace('  ','')
    content.append(result)
    papers=pd.DataFrame(content)
    papers.to_csv('C:/Users/lh.Lenovo-PC/Desktop/papers_scigen_csv.csv',header=False,mode='a')#,encoding="utf-8")

    


def selenium_load_page(url):
    url_list.append(current)

    driver.get(current)
    decode_page(current)
    next_paper_button = driver.find_element_by_xpath("//a[@href='/cgi-bin/scigen.cgi?author=1&']")
    next_paper_button.click()
    #time.sleep(10)
    


for i in range(0,2400):
    print i
    current=driver.current_url
    selenium_load_page(current)



#cnt = 0
#for url in url_list:
#    cnt=cnt+1
#    print cnt
#    if cnt%100==0:
#        decode_page(url)
#        papers=pd.DataFrame(content)
#        papers.to_csv('C:/Users/lh.Lenovo-PC/Desktop/papers_scigen_csv.csv',mode='a')#,encoding="utf-8")
#        content=[]
#    else:
#        decode_page(url)

##result_title = selenium_load_page(innitial_url)
#decode_page(result_title)
#human_finance=pd.DataFrame({"title":title,"time":publish_time,"#word":word_num,"#para":paragraph_num,"content":content,"author":author})
#print(human_finance)
#human_finance.to_csv('C:/Users/lh.Lenovo-PC/Desktop/huamn_finance_csv.csv',encoding="utf-8")
