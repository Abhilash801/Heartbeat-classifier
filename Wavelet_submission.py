#=============== wavelet challenge - heatbeat classification. machine configuration Intel xenon 4 core ,
# 14 GB RAM Azure virtual machine ============
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
matplot.style.use('ggplot') #setting plot style to ggplot
matplot.use('GTK') # working on a remote machine changing the display interface

#============================================================================================================
#============== EDA and preprocessing of data ===============================================================
#============================================================================================================

#*********** Importing the data
pwd='C:\OneDrive\MSBA\kaggle\Wavelet_ML_Challenge'
masterData=pd.read_csv(pwd+'/interview_biodb.csv')
masterData.info()
workingData = masterData[masterData.columns.delete([1,2,3])] #Excluding columns calendar_date,cap_seq and datetime
#  and making a copy of original data

#*********** converting to 32 bit # for X86 support
workingData[workingData.columns[1:23]] = workingData[workingData.columns[1:23]].astype('float32')
workingData.info()
workingData.describe()
#*********** plotting for % number of missing values for the independent variables

df=workingData[workingData.columns[1:23]].isnull().sum()
df=df.div(workingData["_user_id"].count()/100)
objects = df.sort_values(ascending=True).index
y_pos = np.arange(len(objects))
percentageOfMissingValues = df.sort_values(ascending=True)
plt.figure(figsize=(8,9))
plt.barh(y_pos, percentageOfMissingValues,align='center',color='k', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('% of missing values')
plt.title('Missing values by feature')
plt.show()


# ## Plot signifies that feature 16 and feature 8 have the highest number of missing values

#***********replace missing values with means of that particular user
import warnings
warnings.filterwarnings('ignore')
for userid in pd.unique(workingData._user_id):
    for column in workingData.columns.delete(0):
        mask=(workingData._user_id==userid) & (np.isnan(workingData[column]))
        #each missing value is replaced by corresponding mean for that user id
        workingData.loc[mask,column]= np.nanmean(workingData.loc[workingData["_user_id"]==userid,column])

#***********display number of missing values and check if replacement is successful
workingData.isnull().sum() #count missing values
workingData.describe()

#***********standardizing data to minimize effects of high absolute values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(workingData[workingData.columns[1:23]])
workingData[workingData.columns[1:23]]=X

#scaling changes the types to 64 bit, changing them back to 32 bit
workingData[workingData.columns[1:23]] = workingData[workingData.columns[1:23]].astype('float32')
workingData.describe() # verify if scaling is successful, mean should be 0 and standard deviation should be 1


#************ Understanding data with visualization

#  plotting the response variable user ids
plt.figure()
plt.hist(workingData["_user_id"],alpha=.5)
plt.title('User id  frequency distribution')
plt.xlabel('User id')
plt.ylabel('Frequency')
plt.show()
#response variable is balanced as the number of occurences of each class is comparable in the histogram

#creating a stratified sample with 30% of original data for plotting. Plotting entire data makes it cumbersome and time
#consuming
sampleData=pd.DataFrame()
for userid in pd.unique(workingData.loc[:,"_user_id"]):
    sampleCount=int(.3 * workingData.loc[workingData._user_id==userid,"_user_id"].count())
    df=workingData[workingData._user_id==userid].sample(sampleCount)
    sampleData=sampleData.append(df)
sampleData.info()

# parallel co ordinate plot
from pandas.tools.plotting import parallel_coordinates
plt.figure(figsize=(18,6))
parallel_coordinates(sampleData, '_user_id')


# andrews curves
from pandas.tools.plotting import andrews_curves
plt.figure(figsize=(18,6))
andrews_curves(sampleData,'_user_id')

# radviz spring constant plot
from pandas.tools.plotting import radviz
plt.figure(figsize=(12,10))
radviz(sampleData,"_user_id")

# As can be observed from above plots data is very closely spaced without any apparent linear or non-linear boundaries
# Initial inference - A tree based approach might work better when compared to a kernel based boundary fitting approach

# plotting predictor variables, not considering feat8 and feat16 here as majority of them would be replaced missing values
features= ["feat11","feat13","feat5","feat2","feat1","feat4","feat7","feat14","feat10","feat15","feat21","feat3",
           "feat18","feat9","feat17","feat20","feat6","feat12","feat19","feat22"]
workingData[features].hist()


# There are few features with outliers, however, I am not handling outliers as this is health data and anomalies might
#help classification accuracy


#============================================================================================================
#================ Analyze key features and dimensionality reduction =========================================
#============================================================================================================
# **************** Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import time
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 1 attribute, this would give ranking for all the features
rfe = RFE(model,1)
start_time = time.clock()
rfe = rfe.fit(workingData[features], workingData[workingData.columns[0]])
rfeExecutionTime= time.clock() - start_time
rfeExecutionTime=rfeExecutionTime/(60)
print("execution time in minutes : %d " %rfeExecutionTime)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)


# **************** Key features using extra tress classifier
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(workingData[features], workingData[workingData.columns[0]])
# display the relative importance of each attribute
print(model.feature_importances_)
print(model)

# Both the methods above yield consistent results. They disagree on only two features among the top 10 ranked features

# splitting data into testing and training sets. 30% data hold out for testing purposes.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split( workingData[workingData.columns[1:23]], workingData[workingData.columns[0]],
                                                     test_size=0.30)

#============================================================================================================
#================ Test few classification techniques with default configurations ============================
#============================================================================================================
# Each of these sections can be executed independently if the data set up is complete. Hence, imports are included at all
# places. Also, operations are expensive hence coding seperately for executionconvenience,
# instead of single iterator for all techniques

#**************** random forest base classifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

import time
start_time = time.clock()
randomforest_classifier = OneVsRestClassifier(RandomForestClassifier())
randomforest_classifier.fit(X_train, y_train)
Y_pred_randomforest = randomforest_classifier.predict(X_test)
randomforestExecutionTime= (time.clock() - start_time)/60
print("execution time in minutes : %d " %randomforestExecutionTime)
print(accuracy_score(y_test, Y_pred_randomforest))
randomForestConfusoinMatrix=confusion_matrix(y_test, Y_pred_randomforest)
print(randomForestConfusoinMatrix)

#****************** svm base classifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm

import time
svmClassifier = OneVsRestClassifier(svm.SVC(kernel="rbf")) #setting kernel to rbf as there are no apparent linear boundaries
start_time = time.clock()
svmClassifier.fit(X_train, y_train)
ypred_svm = svmClassifier.predict(X_test)
svmExecutionTime= (time.clock() - start_time)/60
print("execution time in minutes : %d " %svmExecutionTime)
print(accuracy_score(y_test, ypred_svm))
svmConfusionMatrix = confusion_matrix(y_test, ypred_svm)
print(svmConfusionMatrix)

#****************** adaboost + decision tree base classifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import time
bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=4),
    n_estimators=60,
    learning_rate=1) #setting max_depth to sqrt(# of features) , n_estimators = 60 and default learning rate, algo is SAMME.R
adaBoostClassifier=OneVsRestClassifier(bdt_real)
start_time = time.clock()
adaBoostClassifier.fit(X_train, y_train)
ypred_adaboost=adaBoostClassifier.predict(X_test)
adaBoostExecutionTime= (time.clock() - start_time)/60
print("execution time in minutes : %d " %adaBoostExecutionTime)
print(accuracy_score(y_test, ypred_adaboost))
adaboostConfusionmatrix=confusion_matrix(y_test, ypred_adaboost)
print(adaboostConfusionmatrix)

#****************** logistic regression with top 10 features
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import time
start_time = time.clock()
logisticFeatures= ["feat11","feat13","feat5","feat2","feat1","feat4","feat7","feat14","feat10"]
logisticClassifier = OneVsRestClassifier(LogisticRegression()).fit(X_train[logisticFeatures],y_train)
ypred_logisticClassifier= logisticClassifier.predict(X_test[logisticFeatures])
logisticClassifierExecutionTime= (time.clock() - start_time)/60
print("execution time in minutes : %d " %logisticClassifierExecutionTime)
print(accuracy_score(y_test, ypred_logisticClassifier))
logisticConfusionmatrix=confusion_matrix(y_test, ypred_logisticClassifier)
print(logisticConfusionmatrix)

#accuracy of .49 not considering for further study

#****************** quadratic discriminant analysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import time
start_time = time.clock()
qdaClassifier = OneVsRestClassifier(QuadraticDiscriminantAnalysis()).fit(X_train,y_train)
ypred_qdaClassifier= qdaClassifier.predict(X_test)
qdaClassifierExecutionTime= (time.clock() - start_time)/60
print("execution time in minutes : %d " %qdaClassifierExecutionTime)
print(accuracy_score(y_test, ypred_qdaClassifier))
qdaConfusionmatrix=confusion_matrix(y_test, ypred_qdaClassifier)
print(qdaConfusionmatrix)


#******************  Knn classifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

import time
knn_classifier = OneVsRestClassifier(KNeighborsClassifier())
start_time = time.clock()
knn_classifier.fit(X_train, y_train)
ypred_knn = knn_classifier.predict(X_test)
knnExecutionTime= time.clock() - start_time
knnExecutionTime=knnExecutionTime*(60)
print("execution time in minutes : %d " %knnExecutionTime)
print(accuracy_score(y_test, ypred_knn))
knnConfusionmatrix=confusion_matrix(y_test, ypred_knn)
print(knnConfusionmatrix)

# Clearly decision tree approach wins with default parameters. Proceeding with parameter tuning using cross validation

#============================================================================================================
# ================= Crossvalidation and Parameter Tuning for Random forest and AdaBoost =====================
#============================================================================================================

#*********** function for reporting best parameters from grid search cross validation
from operator import itemgetter
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

#**************** Random forest grid search

from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
randomForestCv = OneVsRestClassifier(RandomForestClassifier(n_estimators=15,n_jobs=-1)) #n_jobs=-1 all
# processors are used while execution
parameters = {

              "estimator__max_features": ["log2","sqrt"],
              "estimator__min_samples_leaf": [1,2,4],
              "estimator__min_samples_split": [2,4],
              "estimator__bootstrap": [False,True],
              "estimator__criterion": ["gini","entropy"]
}
parameters2 = {

              "estimator__max_features": ["auto","log2"], #checking auto vs log2 this time
              "estimator__min_samples_leaf": [2,4,5]
}

model_tunning = GridSearchCV(randomForestCv, param_grid=parameters)
start_time = time.clock()
model_tunning.fit(workingData[workingData.columns[1:23]], workingData[workingData.columns[0]])
randomForestGridSearchExecutionTime= time.clock() - start_time
randomForestGridSearchExecutionTime=randomForestGridSearchExecutionTime/(60)
print("execution time in minutes : %d " %randomForestGridSearchExecutionTime)
report(model_tunning.grid_scores_)

model_tunning = GridSearchCV(randomForestCv, param_grid=parameters2) #repeating for second set of parameters as min samples
#leaf did not end up as the center value
start_time = time.clock()
model_tunning.fit(workingData[workingData.columns[1:23]], workingData[workingData.columns[0]]) #use entire data for cv instead of traning alone
randomForestGridSearchExecutionTime= time.clock() - start_time
randomForestGridSearchExecutionTime=randomForestGridSearchExecutionTime/(60)
print("execution time in minutes : %d " %randomForestGridSearchExecutionTime)
report(model_tunning.grid_scores_)

# ************** AdaBoost and decision tree grid search

from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import time
adaBoostTreeCv = OneVsRestClassifier(AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=4,min_samples_leaf=4),learning_rate=1)) #using best parameters from random forest


parameters = {
              "estimator__n_estimators": [20,30] # higher this value better, choice is based on resource constraints
}

model_adaBoostTreeCv = GridSearchCV(adaBoostTreeCv, param_grid=parameters)
start_time = time.clock()
#running only on 30% of data, running on entire data is very expensive operation, execution time > ~300 minutes
model_adaBoostTreeCv.fit(sampleData[sampleData.columns[1:23]], sampleData[sampleData.columns[0]])
adaBoostTreeGridSearchExecutionTime= time.clock() - start_time
adaBoostTreeGridSearchExecutionTime=adaBoostTreeGridSearchExecutionTime/(60)
print("execution time in minutes : %d " %adaBoostTreeGridSearchExecutionTime)

print(model_adaBoostTreeCv)
report(model_adaBoostTreeCv.grid_scores_)

#============================================================================================================
#==================== Training final models with tuned parameters ==========================================
#============================================================================================================

#*************** Training final random forest model with best parameters from cross validation

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import time
randomForestTuned = OneVsRestClassifier(RandomForestClassifier(n_estimators=15,n_jobs=-1,min_samples_leaf=4,
                                                             bootstrap=False))
start_time=time.clock()
randomForestTuned.fit(X_train, y_train)
ypred_randomForestTuned = randomForestTuned.predict(X_test)
randomForestTunedTime= time.clock() - start_time
print(accuracy_score(y_test, ypred_randomForestTuned))
randomForestTunedConfusionmatrix=confusion_matrix(y_test, ypred_randomForestTuned)
print(randomForestTunedConfusionmatrix)
randomForestTunedTime=randomForestTunedTime/(60)
print("execution time in minutes : %d " %randomForestTunedTime)


#***************** Training final boosted decision tree model with best parameters from cross validation

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import time
adaBoostTuned = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=4,min_samples_leaf=4),
    n_estimators=30,
    learning_rate=1)
adaBoostTuned=OneVsRestClassifier(adaBoostTuned)
start_time=time.clock()
adaBoostTuned.fit(X_train, y_train)
adaBoostTunedTime= (time.clock() - start_time)/(60)
ypred_adaBoostTuned=adaBoostTuned.predict(X_test)
print("execution time in minutes : %d " %adaBoostTunedTime)
print(accuracy_score(y_test, ypred_adaBoostTuned))
adaboostTunedConfusionmatrix=confusion_matrix(y_test, ypred_adaBoostTuned)
print(adaboostTunedConfusionmatrix)



# Both random forest and boosted tree yield an accuracy of ~.94 with 30 % test data. Recommend using either of these on future
# data. Another idea is to build an ensemble VotingClassifier with any three of the models and have the majority prediction win

# conclusion - Use RandomForest if slight compromise on accuracy is ok, added benefits are a very fast training and
# cross validation executions. If no compromise can be acheived in terms of accuracy, use AdaBoost at the cost of
# expensive training and validation times




