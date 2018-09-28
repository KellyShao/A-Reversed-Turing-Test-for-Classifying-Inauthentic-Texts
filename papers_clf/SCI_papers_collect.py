import csv
import pandas as pd
import numpy as np

filename="SCI_papers.csv"

def import_data(filename):
    content = open(filename, 'rb')
    csv_reader = csv.reader((line.replace('\0','') for line in content))
    papers = []
    cnt = 0
    for row in csv_reader:
        cnt+=1
        if row[6]!='paper_text':
            print cnt
            text = row[6].replace('\n','')
            cnt_words = len(text.split())
            if cnt_words<3500 and cnt_words>500:
                papers.append(text)
    return papers

papers = import_data(filename)
papers=pd.DataFrame(papers)
papers.to_csv('cs_papers/papers_sci.csv',header=False,encoding="utf-8")
