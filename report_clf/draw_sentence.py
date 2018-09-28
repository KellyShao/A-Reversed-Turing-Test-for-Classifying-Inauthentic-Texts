import csv
import numpy as np
import matplotlib.pyplot as plt

file = "auc/f1_array.csv"

x_axis = range(1,21)
x1=np.array(x_axis)
x2=np.array(x_axis)
x3=np.array(x_axis)
x4=np.array(x_axis)

def read_file(file_name):
    content = open(file_name, 'rb')
    csv_reader = csv.reader(content)
    data = []
    for row in csv_reader:
        data.append(row[3])
    return data

y = read_file(file)

w2v_2 = np.float32(y[124:144])
p1,=plt.plot(x1,w2v_2)#,label="w2v_2")
w2v_3 = np.float64(y[144:164])
#p2,=plt.plot(x2,w2v_3)#,label="w2v_3")
tfidf_2 = np.float64(y[42:62])
p3,=plt.plot(x3,tfidf_2)#,label="tfidf_2")
tfidf_3 = np.float64(y[63:83])
#p4,=plt.plot(x4,tfidf_3)#,label="tfidf_3")

#l1 = plt.legend([p1,p2,p3,p4], ["Features: w2v_2", "Features: w2v_3","Features: tfidf_2","Features: tfidf_3"], loc='lower right')
l1 = plt.legend([p1,p3], ["Features: w2v_2", "Features: tfidf_2"], loc='lower right')

plt.xlabel('Percentage of Article Content')
plt.ylabel('Acuracy (f1)')
plt.yticks([0.4,0.5,0.6,0.7,0.8,0.9,1],['0.4','0.5','0.6','0.7','0.8','0.9','1'])
plt.xticks([0,2,4,6,8,10,12,14,16,18,20],['0','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%',])
plt.gca().add_artist(l1)
plt.grid(True, linestyle = "--")#, color = "r", linewidth = "3")
plt.savefig('figure/earnings_f1_w2v+tfidf_2.png', dpi=300)
plt.show()