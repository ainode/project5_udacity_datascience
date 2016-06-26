#!/usr/bin/python

import sys
import pickle
from sklearn.feature_selection.univariate_selection import SelectKBest,\
    f_classif
from nltk.metrics.scores import precision

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

from tester import dump_classifier_and_data 
from sklearn import cross_validation, svm, grid_search
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.svm import SVR, SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
import numpy as np
from tester import test_classifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import RandomizedPCA

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary', 'total_payments', 'bonus', 
                 'deferred_income', 'total_stock_value', 
                 'expenses', 'exercised_stock_options', 'long_term_incentive', 
                 'restricted_stock', 'to_messages', 'from_messages','director_fees',
                 'from_poi_to_this_person', 'from_this_person_to_poi', 
                 'shared_receipt_with_poi'] # You will need to use more features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "Number of data points: ",len(data_dict)

###remove outliers

data_dict.pop("TOTAL")
data_dict.pop("BHATNAGAR SANJAY")
data_dict.pop("MARTIN AMANDA K")
data_dict.pop("PAI LOU L")
data_dict.pop("WHITE JR THOMAS E")
data_dict.pop("KAMINSKI WINCENTY J")
data_dict.pop("FREVERT MARK A")
data_dict.pop("LAVORATO JOHN J")

### Task 3: Create new feature(s)
for name in data_dict:
    if data_dict[name]['from_this_person_to_poi'] != 'NaN' and data_dict[name]['from_messages'] != 'NaN':
        data_dict[name]['fraction_to_poi'] = (float(data_dict[name]['from_this_person_to_poi'])/float(data_dict[name]['from_messages']))
    else:
        data_dict[name]['fraction_to_poi'] = 'NaN'  
    if data_dict[name]['from_poi_to_this_person'] != 'NaN' and data_dict[name]['to_messages'] != 'NaN':
        data_dict[name]['fraction_from_poi'] = (float(data_dict[name]['from_poi_to_this_person'])/float(data_dict[name]['to_messages']))
    else:
        data_dict[name]['fraction_from_poi'] = 'NaN'  

features_list = features_list + ['fraction_to_poi'] + ['fraction_from_poi']                  
data = featureFormat(data_dict, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

#scale features, remove if too slow or if you are not using features that have to do 
#with the number of emails such as from_poi_to_this_person which are of a different scale comparing to say salary

scaler = MinMaxScaler()
scaler = scaler.fit(features)
features = scaler.transform(features)

#spliting data using crossvalidation's train_test_split function
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100, 1000, 10000, 100000], 'gamma':[0.1,1,10,100,1000]}
#svr = svm.SVC()

def make_pred_dt(data_dict, features):
    clf = tree.DecisionTreeClassifier(min_samples_split=10)  
    #clf = SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0, degree=3,\
    #gamma=0.8, kernel='rbf', max_iter=-1, probability=False,\
    #random_state=None, shrinking=True, tol=0.001, verbose=False)
    #clf = SVC(C=1000, kernel='rbf')
    #clf = grid_search.GridSearchCV(svr, parameters)    
    #clf = KMeans(n_clusters=2) 
    #clf = GaussianNB() 
    features = ["poi"] + features
    return test_classifier(clf, data_dict, features)

def choose_features(data_dict,features_list):
    best_precision_recall = 0.0
    best_results = tuple()  
    for i in range(2,len(features_list)):
        selector = SelectKBest(f_classif, k=i)
        selector.fit(features_train, labels_train)
        print "selected features for", i, "features: " 
        features_list_selected=[features_list[i+1] for i in selector.get_support(indices=True)] 
        print features_list_selected       
        precision, recall = make_pred_dt(data_dict, features_list_selected)
        if recall>0.4 and precision>0.4:
            if (recall+ precision) > best_precision_recall:
                best_precision_recall = recall+precision
                best_results = (precision,recall,features_list_selected,selector.scores_[selector.get_support(indices=True)])
        print "********************************************"
    return best_precision_recall, best_results

best_precision_recall, best_results = choose_features(data_dict,features_list)
#best_precisoin_recall, best_results = choose_features_pipline(data_dict, features_list)
print "Precision: ", best_results[0], " Recall: ", best_results[1]
print "Number of features: ",len(best_results[2])
print "features: ", best_results[2]
print "Feature importances(scores): ", best_results[3]




###bellow I created a few functions to try out pca, 
###GridSearchCV and pipelines
###using pca for feature selection
def make_pca(n_components, X_train):
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
    return pca.transform(X_train)
#features = make_pca(6, features)
def SVMAccuracyGrid():        
    parameters = {'kernel':('rbf','sigmoid'),\
    'C':[1,10,1e2,1e3, 1e4, 1e5, 1e6],\
    'gamma': [0,0.0001,0.0005, 0.001, 0.005, 0.01, 0.1]} 
    svr = SVC(verbose=True)
    clf = grid_search.GridSearchCV(svr, parameters)             
    #print("Best estimator found by grid search:")
    #print clf.best_estimator_                 
    return clf
##using pipeline to make sure feature scaling is done on train set only
def choose_features_pipeline(data_dict, features_list):
    scaler = MinMaxScaler()    
    best_precision_recall = 0.0
    best_results = tuple()  
    for i in range(2,len(features_list)):
        selector = SelectKBest(k=i)
        clf = Pipeline(steps=[('scaling',scaler),("SKB", selector), ("NaiveBayes", GaussianNB())])
        #clf = Pipeline(steps=[('scaling',scaler),("SKB", selector), ("SVC", SVMAccuracyGrid())])
        #features = ['poi'] + features
        precision, recall = test_classifier(clf, data_dict, features_list)
        if recall>0.4 and precision>0.4:
            if (recall+ precision) > best_precision_recall:
                best_precision_recall = recall+precision
                best_results = (precision,recall,i)
        print "********************************************"
    return best_precision_recall, best_results        
###This part is for testing different SVM and Grid search algorithms
#svc_features = ['salary', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'shared_receipt_with_poi']
#test_clf = SVMAccuracyGrid()
#test_clf = SVC(C=1000000, cache_size=200, class_weight=None, coef0=0.0, degree=3,\
#    gamma=0.1, kernel='rbf', max_iter=100, probability=False,\
#    random_state=None, shrinking=True, tol=0.001, verbose=False)###best results any for precision from 0.333 to 1.0 and for recall 0.16666
#print "this is the results for gridsearchcv: ", test_classifier(test_clf, data_dict, ["poi"] + svc_features)

###End of testing section for SVM, Grid search, pipelint and pca 





### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.

### in this part you can choose different classifiers to make the pkl files 
### depending what classifier was used to choose features. just need to uncomment
### the line
from sklearn.naive_bayes import GaussianNB
#clf = grid_search.GridSearchCV(svr, parameters)
#clf = SVC(C=100000, cache_size=200, class_weight=None, coef0=0.0, degree=3,\
#gamma=0.8, kernel='rbf', max_iter=-1, probability=False,\
#random_state=None, shrinking=True, tol=0.001, verbose=False)
#clf = GaussianNB()
clf = tree.DecisionTreeClassifier(min_samples_split=10)
#clf = KMeans(n_clusters=2)
#clf = SVMAccuracyGrid()
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
features_list = ['poi'] + best_results[2]

dump_classifier_and_data(clf, my_dataset, features_list)