import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.decomposition  import PCA
buf = [x for x in range(1,53)]
#Read Buy
buy = []
for info in os.listdir('CSV/buy'):
	domain = os.path.abspath('CSV/buy')
	info = os.path.join(domain,info)
	data = pd.read_csv(info,usecols=buf)
	data = data[0:36]
	data = np.array(data)
	h = np.zeros((6,52))
	for i in range(52):
		h[0][i]=np.sum(data[:,i][0:6])
		h[1][i]=np.sum(data[:,i][6:12])
		h[2][i]=np.sum(data[:,i][12:18])
		h[3][i]=np.sum(data[:,i][18:24])
		h[4][i]=np.sum(data[:,i][24:30])
		h[5][i]=np.sum(data[:,i][30:36])
	h = h.reshape((1,312))
	buy.append(h[0])
#Read Communicate
communicate = []
for info in os.listdir('CSV/communicate'):
	domain = os.path.abspath('CSV/Communicate')
	info = os.path.join(domain,info)
	data = pd.read_csv(info,usecols=buf)
	data = data[0:36]
	data = np.array(data)
	h = np.zeros((6,52))
	for i in range(52):
		h[0][i]=np.sum(data[:,i][0:6])
		h[1][i]=np.sum(data[:,i][6:12])
		h[2][i]=np.sum(data[:,i][12:18])
		h[3][i]=np.sum(data[:,i][18:24])
		h[4][i]=np.sum(data[:,i][24:30])
		h[5][i]=np.sum(data[:,i][30:36])
	h = h.reshape((1,312))
	communicate.append(h[0])
#Read Fun
fun = []
for info in os.listdir('CSV/fun'):
	domain = os.path.abspath('CSV/fun')
	info = os.path.join(domain,info)
	data = pd.read_csv(info,usecols=buf)
	data = data[0:36]
	data = np.array(data)
	h = np.zeros((6,52))
	for i in range(52):
		h[0][i]=np.sum(data[:,i][0:6])
		h[1][i]=np.sum(data[:,i][6:12])
		h[2][i]=np.sum(data[:,i][12:18])
		h[3][i]=np.sum(data[:,i][18:24])
		h[4][i]=np.sum(data[:,i][24:30])
		h[5][i]=np.sum(data[:,i][30:36])
	h = h.reshape((1,312))
	fun.append(h[0])
#Read Hope
hope = []
for info in os.listdir('CSV/hope'):
	domain = os.path.abspath('CSV/hope')
	info = os.path.join(domain,info)
	data = pd.read_csv(info,usecols=buf)
	data = data[0:36]
	data = np.array(data)
	h = np.zeros((6,52))
	for i in range(52):
		h[0][i]=np.sum(data[:,i][0:6])
		h[1][i]=np.sum(data[:,i][6:12])
		h[2][i]=np.sum(data[:,i][12:18])
		h[3][i]=np.sum(data[:,i][18:24])
		h[4][i]=np.sum(data[:,i][24:30])
		h[5][i]=np.sum(data[:,i][30:36])
	h = h.reshape((1,312))
	hope.append(h[0])
#Read Mother
mother = []
for info in os.listdir('CSV/mother'):
	domain = os.path.abspath('CSV/mother')
	info = os.path.join(domain,info)
	data = pd.read_csv(info,usecols=buf)
	data = data[0:36]
	data = np.array(data)
	h = np.zeros((6,52))
	for i in range(52):
		h[0][i]=np.sum(data[:,i][0:6])
		h[1][i]=np.sum(data[:,i][6:12])
		h[2][i]=np.sum(data[:,i][12:18])
		h[3][i]=np.sum(data[:,i][18:24])
		h[4][i]=np.sum(data[:,i][24:30])
		h[5][i]=np.sum(data[:,i][30:36])
	h = h.reshape((1,312))
	mother.append(h[0])
#Read Really
really = []
for info in os.listdir('CSV/really'):
	domain = os.path.abspath('CSV/really')
	info = os.path.join(domain,info)
	data = pd.read_csv(info,usecols=buf)
	data = data[0:36]
	data = np.array(data)
	h = np.zeros((6,52))
	for i in range(52):
		h[0][i]=np.sum(data[:,i][0:6])
		h[1][i]=np.sum(data[:,i][6:12])
		h[2][i]=np.sum(data[:,i][12:18])
		h[3][i]=np.sum(data[:,i][18:24])
		h[4][i]=np.sum(data[:,i][24:30])
		h[5][i]=np.sum(data[:,i][30:36])
	h = h.reshape((1,312))
	really.append(h[0])
buy = np.array(buy)
communicate = np.array(communicate)
fun = np.array(fun)
hope = np.array(hope)
mother = np.array(mother)
really = np.array(really)
buy_label = []
communicate_label = []
fun_label = []
hope_label = []
mother_label = []
really_label = []
for i in range(buy.shape[0]):
	buy_label.append(1)
for i in range(communicate.shape[0]):
	communicate_label.append(2)
for i in range(fun.shape[0]):
	fun_label.append(3)
for i in range(hope.shape[0]):
	hope_label.append(4)
for i in range(mother.shape[0]):
	mother_label.append(5)
for i in range(really.shape[0]):
	really_label.append(6)
buy_label = np.array(buy_label)
communicate_label = np.array(communicate_label)
fun_label = np.array(fun_label)
hope_label = np.array(hope_label)
mother_label = np.array(mother_label)
really_label = np.array(really_label)
dataset = np.r_[buy,communicate,fun,hope,mother,really]
label = np.r_[buy_label,communicate_label,fun_label,hope_label,mother_label,really_label]
accuracy1 = []
accuracy2 = []
accuracy3 = []
accuracy4 = []
for i in range(10):
	x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.3)
	pca=PCA(n_components=10)
	pca_x_train=pca.fit_transform(x_train)
	pca_x_test = pca.fit_transform(x_test)
	clf1 = RandomForestClassifier(criterion='entropy',n_estimators=10,n_jobs=2,random_state=1)
	clf2 = DecisionTreeClassifier()
	clf3 = MLPClassifier(learning_rate_init= 0.001,activation='relu',\
     solver='adam', alpha=0.0001,max_iter=30000)
	clf4 = KNeighborsClassifier()
	clf1.fit(x_train,y_train)
	clf2.fit(x_train,y_train)
	clf3.fit(x_train,y_train)
	clf4.fit(x_train,y_train)
	result1 = clf1.predict(x_test)
	result2 = clf2.predict(x_test)
	result3 = clf3.predict(x_test)
	result4 = clf4.predict(x_test)
	accuracy1.append((accuracy_score(result1,y_test)))
	accuracy2.append((accuracy_score(result2,y_test)))
	accuracy3.append((accuracy_score(result3,y_test)))
	accuracy4.append((accuracy_score(result4,y_test)))
print("RandomForest",accuracy1)
print("DecisionTree",accuracy2)
print("NeuralNetwork",accuracy3)
print("KNN",accuracy4)



