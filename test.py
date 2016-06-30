#!/usr/bin/python

#this file is for testing data
#it is created by ali makki for that purpose
import sys
from time import time
import nltk
sys.path.append("../tools/")
import pickle

#from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels

#features_train, features_test, labels_train, labels_test = preprocess()

#print len(features_train)

### Here I am experimenting with enron dataset to find best 
### features for training and create new features and find outliers 
from feature_format import featureFormat, targetFeatureSplit
import numpy as np

#original features
#features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
#                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
#                 'expenses', 'exercised_stock_options', 'long_term_incentive', 
#                 'restricted_stock',
#                 'from_poi_to_this_person', 'from_this_person_to_poi', 
#                 'shared_receipt_with_poi'] # You will need to use more features

#new features
features_list = ['poi', 'salary', 'total_payments', 'bonus', 'director_fees', 
                 'deferred_income', 'total_stock_value', 'restricted_stock_deferred',
                 'expenses', 'exercised_stock_options', 'long_term_incentive', 
                 'restricted_stock', 'from_messages', 'to_messages','deferral_payments',
                 'from_poi_to_this_person', 'from_this_person_to_poi', 
                 'shared_receipt_with_poi'] # You will need to use more features

###definition of features: 
###restricted stock: certain kind of stock that is not fully transferable until a certain conditions are met.
###

with open("../final_project/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#print data_dict
data_dict.pop("TOTAL")
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
print "Type of data: ", type(data)
#print data
#print data[:,0]
all_pois = data[data[:,0] == 1]
all_non_pois = data[data[:,0] == 0]
print "Number of pois:", len(all_pois)
new_feature = np.divide(data[:,15],data[:,12])
print "new_feature is: ", new_feature
print "nan changed to zeros: ", np.nan_to_num(new_feature)
### in this section I am comparing the average of different features
### between pois and non_pois I removed director fee because the values for it were all 0 

def poi_nonpoi_with_features():
    for inx, a_feature in enumerate(features_list):
        print "non_pois with (", a_feature, "): ", len(all_non_pois[all_non_pois[:,inx]!=0.0]) 
        print "non_pois with no (", a_feature, "): ", len(all_non_pois[all_non_pois[:,inx]==0.0])   
        print "pois with (", a_feature, "): ", len(all_pois[all_pois[:,inx]!=0.0]) 
        print "pois with no (", a_feature, "): ", len(all_pois[all_pois[:,inx]==0.0])   
        
        print "..................................."    

poi_nonpoi_with_features()

###I took out three features: 1- deferral_payment 2- loan_advances 3- restricted stock deferred
###with salary there are 49 non pois without salary and 77 with salary
###with pois it is only 1 without salary (maybe salary should be removed)
###only 2 of the non_pois get "loan advances" and 1 of pois. this feature clearly
###does not give enough information and should be removed. 
###deferral_payment has been eliminated because of lack of info for many entries
###all pois have "expenses" and it is not the case with non-pois and also average 
###of "expenses" of pois and non-pois are the same.maybe this feature should be removed as well
###there is no poi with "restricted stock differed" so this feature might be taken out
###KAMINSKI WINCENTY J is a non_pois he has a from_messages of 14368 which is way higher that average and
###very different from the next person which has 6759 
    
def showAveMax():
    print "Now compute averages, standard deviations, ..... :"
    for inx, a_feature in enumerate(features_list):
        if a_feature != 'poi':
            non_zero_pois = all_pois[:,inx][np.nonzero(all_pois[:,inx])]
            if len(non_zero_pois)==0:
                print 'all stats zero or NaN.'
            else:
                print "average",a_feature ,"of pois", np.mean(non_zero_pois), " and max: ", 0 if len(non_zero_pois)==0 else np.max(non_zero_pois)\
                ,"and Median: ", 0 if len(non_zero_pois)==0 else np.median(non_zero_pois), " and 98 percentile: "\
                , np.percentile(non_zero_pois, 90), "standard deviation for pois: ", np.std(non_zero_pois)
            #non_zero_non_pois = all_non_pois[all_non_pois[:,1] != 0.0][:,inx][np.nonzero(all_non_pois[all_non_pois[:,1] != 0.0][:,inx])]
            non_zero_non_pois = all_non_pois[:,inx][np.nonzero(all_non_pois[:,inx])] 
            #non_zero_non_pois = np.nonzero(all_non_pois[:,inx])   
            print 'inx: ',inx        
            print "average",a_feature , "of non pois", np.mean(non_zero_non_pois), " and max: "\
            , np.max(non_zero_non_pois), " and median: "\
            , np.median(non_zero_non_pois)\
            , " and 90 percentile: ", np.percentile(non_zero_non_pois, 90), "standard deviation for non_pois: ", np.std(non_zero_non_pois)
            print "..................................."



#def showAveMaxDev():
#    for inx, a_feature in enumerate(features_list):
#        if a_feature != 'poi':
#            print 'average: ', 


showAveMax()

    
    
from collections import OrderedDict
pois = { k:v for k, v in data_dict.items() if v['poi'] == True}
non_pois = { k:v for k, v in data_dict.items() if v['poi'] == False}
#print data_dict

def show_persons_features(group, group_name,a_feature, number_of_persons):
    ordered_group = OrderedDict(sorted(group.items(), key=lambda x: x[1][a_feature] if x[1][a_feature]!='NaN' else 0, reverse=False))
    print a_feature," for ", group_name, ": "
    #for person in ordered_group:
    #    print person, ordered_group[person][a_feature]
    for i in xrange(1,number_of_persons+1):
        print ordered_group.items()[len(ordered_group)-i][0], ' :', ordered_group.items()[len(ordered_group)-i][1][a_feature]

print '************************************'
print 'top 4 non_pois with highest in a specific feature: '
for feature in features_list:
    show_persons_features(non_pois, 'non_pois',feature, 4)

print '************************************'
print 'top 4 pois with highest in a specific feature: '
for feature in features_list:
    show_persons_features(pois, 'pois',feature, 4)

#get threshold for outliers in each feature for pois and non-pois separately, using standard deviation
def get_thresholds(numOfStds):
    threshold_poi_list = dict()
    threshold_non_poi_list = dict()
    all_pois = data[data[:,0] == 1]
    all_non_pois = data[data[:,0] == 0]    
    for inx, a_feature in enumerate(features_list):
        if a_feature != 'poi':
            non_zero_pois = all_pois[:,inx][np.nonzero(all_pois[:,inx])]
            stdPois = np.std(non_zero_pois)
            threshold_poi_list[a_feature] = stdPois*numOfStds
            #print "standard deviation for pois feature ", a_feature,": ", stdPois
            non_zero_non_pois = all_non_pois[:,inx][np.nonzero(all_non_pois[:,inx])]
            stdNonPois = np.std(non_zero_non_pois)
            threshold_non_poi_list[a_feature] = stdNonPois*numOfStds
            #print "standard deviation for non_pois feature: ", a_feature, ": ", stdNonPois
            #print "..................................."
    return threshold_poi_list, threshold_non_poi_list
#using the lists produced by get_thresholds detect outliers for pois and non_pois separately
def detect_outliers(pois_features_threshold, non_pois_features_threshold): 
    outliers = dict()  
    for person in data_dict:
        for feature in data_dict[person]:
            if feature in features_list and feature != 'poi':
                if data_dict[person]['poi'] == True: 
                    if data_dict[person][feature] != 'NaN' and float(data_dict[person][feature]) > pois_features_threshold[feature]:
                        if outliers.has_key(person):
                            outliers[person][feature] = [pois_features_threshold[feature],data_dict[person][feature],'poi']
                        else:
                            outliers[person] = {feature:[pois_features_threshold[feature],data_dict[person][feature],'poi']}
                else:
                    if data_dict[person][feature] != 'NaN' and float(data_dict[person][feature]) > non_pois_features_threshold[feature]:
                        if outliers.has_key(person):
                            outliers[person][feature] = [non_pois_features_threshold[feature],data_dict[person][feature],'non_poi']
                        else:
                            outliers[person] = {feature:[non_pois_features_threshold[feature],data_dict[person][feature],'non_poi']}
    return outliers

pois_features_threshold, non_pois_features_threshold = get_thresholds(5.6)
outliers = detect_outliers(pois_features_threshold, non_pois_features_threshold)
print outliers













for feature in features_list:
    od_pois = OrderedDict(sorted(pois.items(), key=lambda x: x[1][feature], reverse=False))
#    [~np.isnan(A)].mean()    
    od_non_pois = OrderedDict(sorted(non_pois.items(), key=lambda x: x[1][feature], reverse=False))
    #print "feature: ",feature
    for k, v in od_pois.iteritems():
        #print k
        #print v[feature]
        if k == feature:
            #print v
            m = True    
#print od_pois
#my_dataset = od[len(od)*best_results[9]:len(od)*(1-best_results[9])]
#cleaned_pois = dict(od_pois.items()[int(len(od_pois)*best_results[9]):int(len(od_pois)*(1.0-best_results[9]))])
#cleaned_non_pois = dict(od_non_pois.items()[int(len(od_non_pois)*best_results[9]):int(len(od_non_pois)*(1.0-best_results[9]))])
#my_dataset = cleaned_pois.copy()
#my_dataset.update(cleaned_non_pois)
#print "len(my_dataset)",len(my_dataset)
