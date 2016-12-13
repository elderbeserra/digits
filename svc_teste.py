from sklearn.svm import SVC
from numpy import genfromtxt, savetxt

# DIR = '/home/elder/projetos/kaggle/digit/'
DIR = '/home/projects/github/kaggle/'
dados = genfromtxt(open(DIR+'train.csv', 'r'), delimiter=',')[1:]

labels = [i[0] for i in dados]
treino = [i[1:] for i in dados]

sample = genfromtxt(open(DIR+'test.csv', 'r'), delimiter=',')[1:]

# version 1
clf = SVC()
clf.fit(treino, labels)

savetxt(DIR+'svc_1.csv', clf.predict(sample), delimiter=',', fmt='%d')
