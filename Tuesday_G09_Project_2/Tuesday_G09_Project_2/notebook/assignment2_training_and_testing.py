import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


# Fixing frame numbers of one video to 30 frames
# :param ls: column list
# :return: mean of windows with fixed 30 frames
def get_window_mean(ls):
    WINDOW_COUNT=30
    return [np.mean(x) for x in np.array_split(ls,WINDOW_COUNT)]

# Getting gesture name and corresponding csv files
# :return: data_set consists of dictionary of gestures as keys and corresponding csv file names as values,
#          directory path containing all 6 gesture directories.
def get_csv():
    gesture_path_folder='/Users/excsimon/Documents/GitHub/CSE535_Assignment2/CSV'
    gesture_set=[x for x in os.listdir(gesture_path_folder) if x!='.DS_Store']
    data_set=[]
    for x in gesture_set:
        csv_set=[x for x in os.listdir(os.path.join(gesture_path_folder,x)) if x!='.DS_Store']
        data_set.append({x:csv_set})
    return data_set,gesture_path_folder

# Preparing data for training and testing
# :return: training data set, 
#          training labels,
#          testing data set,
#          testing labels.
def prepare_data():
    gesture_class=['communicate', 'really', 'fun', 'mother', 'hope', 'buy']
    columns=['leftShoulder_x', 'leftShoulder_y',
    'rightShoulder_x', 'rightShoulder_y', 'leftElbow_x', 'leftElbow_y', 
    'rightElbow_x', 'rightElbow_y', 'leftWrist_x', 'leftWrist_y',
    'rightWrist_x', 'rightWrist_y']
    data_set,gesture_path_folder=get_csv()
    data,label = [],[]
    for dictionary in data_set:
        gesture_name=list(dictionary.keys())[0]
        csv_list=list(dictionary.values())[0]
        for csv_name in csv_list:
            data_frame = pd.read_csv(os.path.join(gesture_path_folder,gesture_name,csv_name))
            ls=[]
            for i in range(len(columns)):
                ls+=get_window_mean(list(data_frame[columns[i]]))
            data.append(ls)
        label+=[gesture_class.index(gesture_name)]*len(csv_list)
    #x_train, x_test, y_train, y_test
    train_data,test_data,train_label,test_label=train_test_split(data,label,test_size=0.2)
    return train_data,test_data,train_label,test_label

# Training and testing 4 machine learning models
# print mean accuracy evaluations of corresponding 4 models in N iteration times
def predictor():
    accuracy1,accuracy2,accuracy3,accuracy4= [],[],[],[]
    iteration_times=20
    for i in range(iteration_times):
        train_data,test_data,train_label,test_label=prepare_data()
        #RandomForest
        clf1 = RandomForestClassifier(criterion='entropy',n_estimators=10,n_jobs=2,random_state=1)
        clf1.fit(train_data,train_label)
        res1=clf1.predict(test_data)
        #DecisionTree
        clf2 = DecisionTreeClassifier()
        clf2.fit(train_data,train_label)
        res2=clf2.predict(test_data)
        #MLP
        clf3 = MLPClassifier(learning_rate_init= 0.0001,activation='relu',solver='adam', alpha=0.0001,max_iter=30000)
        clf3.fit(train_data,train_label)
        res3=clf3.predict(test_data)
        #KNeighbors
        clf4 = KNeighborsClassifier()
        clf4.fit(train_data,train_label)
        res4=clf4.predict(test_data)

        accuracy1.append(accuracy_score(res1,test_label))
        accuracy2.append(accuracy_score(res2,test_label))
        accuracy3.append(accuracy_score(res3,test_label))
        accuracy4.append(accuracy_score(res4,test_label))
    
    print("RandomForest {0} times iteration average accuracy:{1}"\
        .format(iteration_times,np.around(np.mean(accuracy1),decimals=5)))
    print("DecisionTree {0} times iteration average accuracy:{1}"\
        .format(iteration_times,np.around(np.mean(accuracy2),decimals=5)))
    print("MLPClassifier {0} times iteration average accuracy:{1}"\
        .format(iteration_times,np.around(np.mean(accuracy3),decimals=5)))
    print("KNN {0} times iteration average accuracy:{1}"\
        .format(iteration_times,np.around(np.mean(accuracy4),decimals=5)))
    
if __name__ == "__main__":
    predictor()