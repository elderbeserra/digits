# -*- coding: utf-8 -*-

from sklearn import neighbors
from numpy import genfromtxt, savetxt

# DIR = '/home/elder/projetos/kaggle/digit/'
DIR = '/home/projects/github/kaggle/'
dados = genfromtxt(open(DIR+'train.csv', 'r'), delimiter=',')[1:]

labels = [i[0] for i in dados]
treino = [i[1:] for i in dados]

teste = genfromtxt(open(DIR+'test.csv', 'r'), delimiter=',')[1:]

knn = neighbors.KNeighborsClassifier(n_jobs=-1)
knn.fit(treino,labels)

# rforest = RandomForestClassifier(n_estimators=300, n_jobs=-1)
# rforest.fit(treino, labels)

savetxt(DIR+'outputknn1.csv', knn.predict(teste), delimiter=',', fmt='%d')
