1- I have included tester.py because it is being used in cross validation process. I made a very small change so that it returns precision and recall which will be used in finding the right
number of features.

2- in poi_id.py, lines 58 to 67 are the part that new features are created. there are two main functions that help with choosing features: choose_features() and make_pred_dt().
make_pca(), SVMAccuracyGrid() and choose_features_pipeline() were created for experimenting with GridSearch were created to experiment with pca GridSearchSV and piplines and 
can be ignored. choose_features() loops over i number of features and out of all features chooses i best features and calls and passes the features to make_pred_dt() were features and 
the classifier are passed to tester.py for crossvalidation and calculation for precision and recall. the results are sent back to choose_features() where best results are evantually
returned which include the list of features that gave the best results with a specific classifier. By using decision tree I got the highest result:
Precision:  0.611764705882  Recall:  0.494
Number of features:  9
features:  ['salary', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value', 'exercised_stock_options', 'restricted_stock', 'shared_receipt_with_poi', 'fraction_to_poi']
Feature importances(scores):  [ 14.062324     7.89790888  21.83935996   9.86208592  15.75217353
  14.54427453  10.2906055    8.27015519  18.02519584]
and one of the features was the new feature that was created using email features. by changing min_sample_split parameter I could get precision and recall of both around 5.3, but when I
set it to 10 I got the above result with the highest total and acceptable value for both metrics. Guasian Naive also gave good results, but the best result was attained with out including
the new features at:
Precision:  0.550499445061  Recall:  0.496
Number of features:  10
features:  ['salary', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'shared_receipt_with_poi']
Feature importances(scores):  [ 14.062324     7.89790888  21.83935996   9.86208592  15.75217353
   7.71276707  14.54427453   7.79171585  10.2906055    8.27015519] 

3- in test.py mainly the dataset is being explored to find out how many features of pois and non pois are missing, what are the standard deviations of each feature and max, min and median
values for pois and non-pois are computed separately to give a picture of distribution of features and helps with detecting outliers. function get_threshold() takes the number of standard
deviation as argument and returns the threshold for each feature for pois and non-pois separately. Although the distribution of this data set might not be normal, standard deviation gives
some idea of which data points might be outliers. I first visually inspected the data set and then by using the first two function and my own intuition got an idea of which datapoints 
might be outliers and then used get_threshold() to get the minumum value of each feature that would be beyond that standard deviation and gave the list of those thresholds to 
detect_outliers() to find list of poi outliers and non_poi outliers. at standard deviation of 5.6 I found a number of outliers from which only one is a poi.



 
