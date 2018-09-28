#paint result

import csv
import numpy as np
import matplotlib.pyplot as plt

tfidf_file = "f1/f1_tfidf_sentence.csv"
w2v_file = "f1/f1_w2v_sentence.csv"

x_axis = range(1,21)
x_axis_small = [0.2,0.4,0.6,0.8,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
x1=np.array(x_axis)
x2=np.array(x_axis_small)

def read_file(file_name):
    content = open(file_name, 'rb')
    csv_reader = csv.reader(content)
    data = []
    for row in csv_reader:
        data.append(row[1])
    return data

y_tfidf = read_file(tfidf_file)
y_w2v = read_file(w2v_file)
y_small = [0.748813471,0.917895204,0.960154265,0.978624637]
y_w2v = y_small+y_w2v

w2v = np.float64(y_w2v)
p2,=plt.plot(x2,w2v)
tfidf = np.float32(y_tfidf)
p1,=plt.plot(x1,tfidf)

l1 = plt.legend([p2,p1], ["Features: w2v", "Features: tfidf"], loc='lower right')
plt.xlabel('Percentage of Article Content')
plt.ylabel('f1 score')
plt.yticks([0.4,0.5,0.6,0.7,0.8,0.9,1],['0.4','0.5','0.6','0.7','0.8','0.9','1'])
plt.xticks([0,2,4,6,8,10,12,14,16,18,20],['0','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%',])
plt.gca().add_artist(l1)
plt.grid(True, linestyle = "--")
plt.savefig('papers_f1_w2v+tfidf_small.png', dpi=300)
plt.show()
