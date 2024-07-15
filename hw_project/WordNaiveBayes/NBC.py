import json
import csv
import time
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt

text = []
new_dict = {}

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=[50,100,200,400,800,1600,3200,6400,12800,25600]):
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="test score")
    plt.legend(loc="best")
    return plt

def load_json(filename): 
    sentence = []
    with open(filename,"r") as t:
        text = t.readlines()
    check = len(text)
    dict_num = 0
    with open('filter_data.csv','w', encoding='UTF-8', newline='') as ft: 
        wri = csv.writer(ft)
        wri.writerow([" ", "votes", "user_id" ,"review_id","stars","date","text","type","business_id","label"])
        for i in range(check):
            listset_train_set=[]
            listset_train_set = text[i]
            ch = eval(listset_train_set)
            nu = 0
            nu = ch["votes"]["funny"] + ch["votes"]["useful"] + ch["votes"]["cool"]
            if (ch["stars"]>3.5 or nu > 0):
                ch['label'] = 1
            try:
                if (nu >= 3 and nu < 11):
                    wri.writerow([dict_num, ch["votes"],ch["user_id"],ch["review_id"],ch["stars"],ch["date"],ch["text"],ch["type"],ch["business_id"],ch["label"]])
                    new_dict[dict_num]=ch
                    dict_num += 1
            except ValueError:
                i = i+1
    with open('new_yelp_academic_dataset_review.json', 'w', encoding="utf-8") as make_file:
        json.dump(new_dict, make_file, ensure_ascii=False, indent="\t")
    return new_dict

def sortvoca():
    train_data_s = pd.read_csv("filter_data.csv")
    sentence =[]
    checknum = len(train_data_s)
    for i in range(checknum):
        sentence.append(train_data_s["text"][i])
    return sentence

def NBC_model(X, corpus, label_type, train_data):
    result_data_funny, result_data_useful, result_data_cool, result_data_positive =[],[],[],[]
    y = train_data.iloc[:, 9].values
    label_data = ["isFunny","isUseful","isCool","isPositive"]
    data_size = 50
    for j in range(10): #train_test_split(arrays, test_size, train_size, random_state, shuffle, stratify)
        i = 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 8000, train_size = data_size , random_state = 123)
        classifier = MultinomialNB(alpha=3.0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        if (label_type == "isFunny"):
            result_data_funny.append(accuracy_score(y_test, y_pred))
        elif (label_type == "isUseful"):
            result_data_useful.append(accuracy_score(y_test, y_pred))
        elif (label_type == "isCool"):
            result_data_cool.append(accuracy_score(y_test, y_pred))
        elif (label_type == "isPositive"):
            result_data_positive.append(accuracy_score(y_test, y_pred))
        
        print(label_type,"인 경우","training_data_size=",data_size, i,"회 정확도:", accuracy_score(y_test, y_pred))
        data_size = data_size * 2
        i += 1
    estimator = MultinomialNB(alpha=3.0)
    if (label_type == "isFunny"):
        plot_learning_curve(estimator, label_data[0], X, y, ylim=(0.0, 1.0), cv=None, n_jobs=3)
        return result_data_funny
    elif (label_type == "isUseful"):
        plot_learning_curve(estimator, label_data[1], X, y, ylim=(0.0, 1.0), cv=None, n_jobs=3)
        return result_data_useful
    elif (label_type == "isCool"):
        plot_learning_curve(estimator, label_data[2], X, y, ylim=(0.0, 1.0), cv=None, n_jobs=3)
        return result_data_cool
    elif (label_type == "isPositive"):
        plot_learning_curve(estimator, label_data[3], X, y, ylim=(0.0, 1.0), cv=None, n_jobs=3)
        return result_data_positive
    plt.show()

def NBC_start():
    result_set_funny, result_set_useful, result_set_cool, result_set_positive = [],[],[],[]
    train_data = pd.read_csv("filter_data.csv")
    new_data = sortvoca()
    pd.options.mode.chained_assignment = None  # default='warn'
    sw = stopwords.words("english")
    cv = CountVectorizer(max_features = 5000, stop_words=sw)
    X = cv.fit_transform(new_data).toarray()
    check_len = len(train_data)
    label_data = ["isFunny","isUseful","isCool","isPositive"]
    for label_change in label_data:
        if (label_change == "isFunny"):
            for i in range(check_len):
                if (train_data["votes"][i][10] != "0"):
                    train_data["label"][i] = "1"
                elif (train_data["votes"][i][10] == "0"):
                    train_data["label"][i] = "0"
            result_set_funny = NBC_model(X, new_data,label_change, train_data)
        elif (label_change == "isUseful"):
            for i in range(check_len):
                if (train_data["votes"][i][23] != "0"):
                    train_data["label"][i] = "1"
                elif (train_data["votes"][i][23] == "0"):
                    train_data["label"][i] = "0"
            result_set_useful = NBC_model(X, new_data,label_change, train_data)
        elif (label_change == "isCool"):
            for i in range(check_len):
                if (train_data["votes"][i][34] != "0"):
                    train_data["label"][i] = "1"
                elif (train_data["votes"][i][34] == "0"):
                    train_data["label"][i] = "0"
            result_set_cool = NBC_model(X, new_data,label_change, train_data)
        elif (label_change == "isPositive"):
            for i in range(check_len):
                if (train_data["stars"][i] > 3.5):
                    train_data["label"][i] = "1"
                elif (train_data["stars"][i] <= 3.5):
                    train_data["label"][i] = "0"
            result_set_positive = NBC_model(X, new_data,label_change, train_data)
    
    #print("result_set_funny",result_set_funny)  
    #print("result_set_useful",result_set_useful)
    #print("result_set_cool",result_set_cool)
    #print("result_set_positive",result_set_positive)
    for label_type in label_data:
        result_fig(label_type, result_set_funny,result_set_useful, result_set_cool, result_set_positive)

def result_fig(label, funny, useful, cool, positive):
    plt.figure()
    plt.xlabel("training data size")
    plt.ylabel("Score")
    if (label == "isFunny"):
        plt.title(label)
        plt.plot([50,100,200,400,800,1600,3200,6400,12800,25600],funny,marker='o') #10,20,40,80,160,320,640,1280,2560,5120]
    elif(label == "isUseful"):
        plt.title(label)
        plt.plot([50,100,200,400,800,1600,3200,6400,12800,25600],useful,marker='o') #50,100,200,400,800,1600,3200,6400,12800,25600]
    elif(label == "isCool"):
        plt.title(label)
        plt.plot([50,100,200,400,800,1600,3200,6400,12800,25600],cool,marker='o') #50,100,200,400,800,1600,3200,6400,12800,25600]
    elif(label == "isPositive"):
        plt.title(label)
        plt.plot([50,100,200,400,800,1600,3200,6400,12800,25600],positive,marker='o') #50,100,200,400,800,1600,3200,6400,12800,25600]
    plt.show()

if __name__ == "__main__":
    start_time = time.time()
    check_data = load_json('yelp_academic_dataset_review.json') #check data = 필터링 데이터 yelp_academic_dataset_review, yelp_.json
    NBC_start()
    print("End")
    print(time.time()-start_time)