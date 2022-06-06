# import necessary libraries
# import pandas for tables manipulation
import pandas as pd
# import from sklearn function for feature scaling
from sklearn.preprocessing import MinMaxScaler
# import the train test split and KFold functions to split the data
from sklearn.model_selection import train_test_split, KFold
# import the kNN classifier
from sklearn.neighbors import KNeighborsClassifier
# import the metrics we are going to use
from sklearn.metrics import accuracy_score, f1_score
# import the confusion matrix to display our results
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import resample for Bootstrap sampling
from sklearn.utils import resample
# import matplotlib to produce plots
import matplotlib.pyplot as plt
# import seaborb to produce plots
import seaborn as sns
# import numpy library
import numpy as np
# import sys in order to terminate the script
import sys

# configurations to display dataframes correctly in pycharm console
pd.set_option("display.width", 320)
pd.set_option("display.max_columns", 10)
# configuration in order to view plots in pycharm's python console
plt.interactive(True)

# create a function that returns the result of two lengths divide given the search in format (xx.xx%) as a string
def getPercentage(divider, divisor) :

    return "(" + str(round((len(divider) / len(divisor)) * 100, 2)) + "%)"

# import the dataset, with the first line as a header
# the dataset was downloaded from https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset
df = pd.read_csv("./breast-cancer.csv", header = 0)

##
## DATA EXPLORATION
##

# show the first 5 records
print(df.head(5))
# take a look at the nature of the fields
print(df.info())
# print update
print("\nWe can see the categorical variable is diagnosis with the unique values:", df["diagnosis"].unique(), "\nAll other variables are numerical and can be used as features except the id where it corresponds to a single patient (Primary KEY).\nAlso there is no column with a null value.")

##
## DATA VALIDATION
##
# first of all check for duplicate rows
print("Total number of duplicates rows:", len(df[df.duplicated()]), getPercentage(df[df.duplicated()], df))
# the id must be unique, so we don't have to see any duplicates there either
print("Total number of duplicated IDs:", len(df[df.duplicated("id")]), getPercentage(df[df.duplicated("id")], df))
# check format of the data
for col in df.columns :
    # print the number of nans in a given column - only if it is different than zero
    if len(df[df[col].isna()]) != 0 :
        print("Number of NaNs in column " + col + ":", len(df[df[col].isna()]), getPercentage(df[df[col].isna()], df))
    # the id must be unique, so we don't have to see any duplicates there either
    if col in ["id"] :
        # print number of duplicated keys - only if it is different than zero
        if len(df[df.duplicated(col)]) != 0 :
            print("Number of duplicates in column " + col + ":", len(df[df.duplicated(col)]), getPercentage(df[df.duplicated(col)], df))
    # columns that are considered numerical - all columns except the primary key and the categorical variable
    if col in [c for c in df.columns if c not in ["id", "diagnosis"]] :
        # print number of rows that are not numeric - only if it is different than zero
        if len(df[pd.to_numeric(df[col], errors = "coerce").isna()]) != 0 :
            print("Number of nonnumerical values in column " + col + ":", len(df[pd.to_numeric(df[col], errors = "coerce").isna()]), getPercentage(df[pd.to_numeric(df[col], errors = "coerce").isna()], df))

# no issues were observed, thus we continue with the EDA
# also, we drop the "id" field as we won't need it for our analysis anymore
df.drop("id", inplace = True, axis = "columns")

##
## EXPLORATORY DATA ANALYSIS (EDA)
##
# let's see how the categorical variable is distributed
# set the figure size
plt.figure(figsize = (6, 6))
# create the barplot
ax = sns.countplot(data = df, x = "diagnosis", linewidth = 0.5)
# set the percentage in top of each bar
for bar in ax.patches :
    ax.annotate("{:.2f}%".format(100. * bar.get_bbox().get_points()[1, 1] / len(df)),
                (bar.get_bbox().get_points()[:, 0].mean(), bar.get_bbox().get_points()[1, 1]),
                ha = "center",
                va = "bottom")
# set the y label
plt.ylabel("Count")
# set the x label
plt.xlabel("Diagnosis")
# set the title
plt.title("Categorical Variable Distribution")
# show the plot
plt.show()
# we can see that the categorical variable is not that balanced.
# However, for the sake of this test assignment we will proceed with no modifications
# let's proceed with the distribution of the features
# get the names of all features
features = [col for col in df.columns if col not in ["diagnosis"]]
# total number of features
print("Total number of features:", len(features))
# we will make 5 rows of 6 features diplayed in each row
rows = 5
# set the figure size
plt.figure(figsize = [15, 3 * len(features) / rows])
# for each feature
for i, feature in enumerate(features):
    # add a subplot
    plt.subplot(rows, int(len(features) / rows), i + 1)
    # create a distribution plot for each feature
    sns.distplot(df[feature])
# set the layout
plt.tight_layout()
# show the figure
plt.show()

# we can see that the features follow kinda the normal distribution
# however we can observe outliers in the dataset

##
## DATA PREPROCESSING AND CLEANING
##
# as we saw before, there are outliers in our dataset
# we will proceed by removing using the IQR method
# print the update
print("Dataset length with the outliers:", len(df))
# for each feature
for feature in features :
    # get the first quartile of the feature
    q1 = df[feature].quantile(0.15)
    # get the third quartile of the feature
    q3 = df[feature].quantile(0.85)
    # calculate the IQR
    iqr = q3 - q1
    # filter out the values outside the specified range
    df = df[df[feature] >= (q1 - (1.5 * iqr))]
    df = df[df[feature] <= (q3 + (1.5 * iqr))]
# reset the pandas' dataframe index
df.reset_index(drop = True, inplace = True)
# print the update
print("Dataset length without the outliers:", len(df))
# we can see that we removed 67 outliers from our dataset
# now let's see the correlation matrix of the features between them
# set the fig size
plt.figure(figsize = [20, 20])
# create the heatmap
sns.heatmap(df[features].corr(), vmin = -1, vmax = 1, center = 0, annot=True)
# set the tile
plt.title("Features Correlation-Plot")
# shot the plot
plt.show()
# we can see that some features have a high correlation between them
# that means that they do add the same information in our model
# we will do two tests, one with all the feature and one removing the most correlated, assuming that multicolinearity will decrease our model's accuracy
# get correlation matrix
corr = df[features].corr().corr().abs()
# find the correlated variables of over 90% and below 100% as the variables correlated with the same one
high_corr = corr[(df[features].corr().abs() > 0.8) & (df[features].corr().abs() < 1.0)]
# get the columns that needs to be filtered out
columnsToDrop = ~high_corr[(df[features].corr().corr().abs() > 0.8) & (df[features].corr().corr().abs() < 1.0)].any()
# get the final dataframe after feature selection
removedFeaturesDF = df[high_corr.columns[columnsToDrop]]
# get the labels as well
removedFeaturesDF = removedFeaturesDF.join(df["diagnosis"])
# get the names of the new features
removedFeatures = [col for col in removedFeaturesDF.columns if col not in ["diagnosis"]]
# map the categorical values into numbers
removedFeaturesDF["diagnosis"] = removedFeaturesDF["diagnosis"].map({"M": 1, "B": 0})
# map the categorical values into numbers
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
# we proceed by splitting the dataset into training (80%) and test set (20%)
rftrX, rftX, rftrY, rftY = train_test_split(removedFeaturesDF[removedFeatures].values, removedFeaturesDF["diagnosis"].values, test_size = 0.2, random_state = 20220607)
# we proceed by splitting the dataset into training (80%) and test set (20%)
tr, t = train_test_split(df.values, test_size = 0.2, random_state = 20220607)
# print the update
print("Length of training set:", len(tr), "\nLength of test set:", len(t))
# regarding the feature scaling we will normalize the data as won't have to make any assumptions on the distribution of them
# and of course we have removed the outliers
# instantiate the minmax scaler
scaler = MinMaxScaler()

##
## HYPERPARAMETER TUNING
##
def cvForHyperparameterTuning(trX, trY) :
    # now we will proceed with tuning the hyperparameters of the kNN algorithm
    # for this task, we will use the cross-validation technique using the KFolds function
    # instantiate the KFolds with 5 splits
    cv = KFold(n_splits = 5, random_state = 20220607, shuffle = True)
    # set the maximum number of k neighbors to search for (theoritacally, the ideal will be root(len(observations)) = root(502) ~ 22.04
    maxKNeighbors = 30
    # create a pandas dataframe to record the F1 scores of training and validation
    f1Scores = pd.DataFrame(columns = ["KNeighbors", "Weight", "Distance Metric", "Split", "F1 Score"])
    # for every number of neighbors
    for kNeighbors in range(1, maxKNeighbors + 1) :
        # for each weight
        for weight in ["uniform", "distance"] :
            # for each distance metric
            for distanceMetric in [1, 2] :
                # initialize a knn classifier with the current configurations
                knn = KNeighborsClassifier(n_neighbors = kNeighbors, weights = weight, p = distanceMetric)
                # set the number of split to 1
                split = 1
                # Loop over the cross-validation splits:
                for trainIndex, valIndex in cv.split(trX):
                    # split the data given the folds
                    trainX, valX, trainY, valY = trX[trainIndex], trX[valIndex], trY[trainIndex], trY[valIndex]
                    # normalize the data
                    trainX = scaler.fit_transform(trainX)
                    valX = scaler.fit_transform(valX)
                    # fit the model on the current split of data
                    model = knn.fit(trainX, trainY)
                    # make predictions on the validation set
                    valPredictions = model.predict(valX)
                    # append the configuration and the score in the dataframe
                    f1Scores = f1Scores.append({"KNeighbors" : kNeighbors,
                                                "Weight" : weight,
                                                "Distance Metric" : distanceMetric,
                                                "Split" : split,
                                                "F1 Score" : f1_score(valY, valPredictions)},
                                               ignore_index = True)
                    # change the counter
                    split += 1

    # return the results
    return f1Scores.\
        groupby(list(f1Scores.columns[:-2])).\
        agg(MeanF1Scores = ("F1 Score", "mean")).\
        sort_values("MeanF1Scores", ascending = False).\
        reset_index()

# calculate the mean F1 score for each configuration with the removed features
removedFeaturesAverageF1Scores = cvForHyperparameterTuning(rftrX, rftrY)
# calculate the mean F1 score for each configuration without the removed features
averageF1Scores = cvForHyperparameterTuning(tr[:, 1:], tr[:, 0])
# print the top 5 scores with the removed features
print("With Feature Selection\n", removedFeaturesAverageF1Scores.head(5))
# print the top 5 scores without the removed features
print("Without Feature Selection\n", averageF1Scores.head(5))

##
## MAKE PREDICTIONS
##
# we can see that the feature selection did not improve our model
# thus we will continue by using all features
# now we will use the top configuration to test our model on the unseen data
# initialize a knn classifier with the best configuration of the validation data
knn = KNeighborsClassifier(n_neighbors = averageF1Scores.loc[0, "KNeighbors"],
                           weights = averageF1Scores.loc[0, "Weight"],
                           p = averageF1Scores.loc[0, "Distance Metric"])
# keep a copy of unnormalized trainining data
tr2 = tr
# normalize the train data
tr = scaler.fit_transform(tr)
# normalize the test data
t = scaler.fit_transform(t)
# fit the model on the training set
model = knn.fit(tr[:, 1:], tr[:, 0])
# make predictions on the test set
tPredictions = model.predict(t[:, 1:])
# get the confusion matrix of the predictions
cm = confusion_matrix(t[:, 0], tPredictions)
# transform the cm to display it
cmDisplay = ConfusionMatrixDisplay(confusion_matrix = cm)
# create the plot
cmDisplay = cmDisplay.plot(cmap = plt.cm.Greens, values_format = "g")
# set the title
plt.title("kNN\nKNeighbors : " + str(averageF1Scores.loc[0, "KNeighbors"]) + " - Weight : " + str(averageF1Scores.loc[0, "Weight"]) + " - Distance Metric : " + str(averageF1Scores.loc[0, "Distance Metric"]), pad = 10, fontsize = 10)
# show the plot
plt.show()
# keep the accuracy
kfoldsAccuracy = round(accuracy_score(t[:, 0], tPredictions) * 100, 4)
# keep the f1 score
kfoldsF1 = round(f1_score(t[:, 0], tPredictions) * 100, 4)
# print the update
print("Accuracy: " + str(kfoldsAccuracy) + "%\nF1 Score: " + str(kfoldsF1) + "%")

##
## BOOTSTRAP METHOD
##
# we will use all features for the bootstrap method
# in order to compare it to the previous method, we will use the same training and test sets
# now we will proceed with tuning the hyperparameters of the kNN algorithm
# for this task, we will use the cross-validation technique using the KFolds function
# instantiate the KFolds with 5 splits
cv = KFold(n_splits=5, random_state=20220607, shuffle=True)
# set the maximum number of k neighbors to search for (theoritacally, the ideal will be root(len(observations)) = root(502) ~ 22.04
maxKNeighbors = 30
# create a pandas dataframe to record the F1 scores of training and validation
f1Scores = pd.DataFrame(columns=["KNeighbors", "Weight", "Distance Metric", "Iteration", "F1 Score"])
# for every number of neighbors
for kNeighbors in range(1, maxKNeighbors + 1):
    # for each weight
    for weight in ["uniform", "distance"]:
        # for each distance metric
        for distanceMetric in [1, 2]:
            # initialize a knn classifier with the current configurations
            knn = KNeighborsClassifier(n_neighbors=kNeighbors, weights=weight, p=distanceMetric)
            # set the number of split to 1
            iteration = 1
            # for each iteration
            for i in range(5) :
                # bootstrap from the train data with replacement
                train = resample(tr2, n_samples = int(len(tr2) * 0.8), replace = True)
                # test data will be all except the ones already picked
                val = np.array([x for x in tr2 if x.tolist() not in train.tolist()])  # picking rest of the data not considered in training sample
                # normalize the data
                train = scaler.fit_transform(train)
                test = scaler.fit_transform(val)
                # train the model
                knn.fit(train[:, 1:], train[:, 0])
                # make the predictions
                valPredictions = model.predict(val[:, 1:])
                # append the configuration and the score in the dataframe
                f1Scores = f1Scores.append({"KNeighbors": kNeighbors,
                                            "Weight": weight,
                                            "Distance Metric": distanceMetric,
                                            "Iteration": iteration,
                                            "F1 Score": f1_score(val[:, 0], valPredictions)},
                                           ignore_index=True)
                # change the counter
                iteration += 1

# return the results
bootstrapAverageF1Scores = f1Scores. \
    groupby(list(f1Scores.columns[:-2])). \
    agg(MeanF1Scores=("F1 Score", "mean")). \
    sort_values("MeanF1Scores", ascending=False). \
    reset_index()

# print the top 5 scores with the removed features
print("With Bootstap\n", bootstrapAverageF1Scores.head(5))

# initialize a knn classifier with the best configuration of the validation data
knn = KNeighborsClassifier(n_neighbors = bootstrapAverageF1Scores.loc[0, "KNeighbors"],
                           weights = bootstrapAverageF1Scores.loc[0, "Weight"],
                           p = bootstrapAverageF1Scores.loc[0, "Distance Metric"])
# normalize the train data
tr = scaler.fit_transform(tr2)
# fit the model on the training set
model = knn.fit(tr[:, 1:], tr[:, 0])
# make predictions on the test set
tPredictions = model.predict(t[:, 1:])
# get the confusion matrix of the predictions
cm = confusion_matrix(t[:, 0], tPredictions)
# transform the cm to display it
cmDisplay = ConfusionMatrixDisplay(confusion_matrix = cm)
# create the plot
cmDisplay = cmDisplay.plot(cmap = plt.cm.Greens, values_format = "g")
# set the title
plt.title("kNN\nKNeighbors : " + str(averageF1Scores.loc[0, "KNeighbors"]) + " - Weight : " + str(averageF1Scores.loc[0, "Weight"]) + " - Distance Metric : " + str(averageF1Scores.loc[0, "Distance Metric"]), pad = 10, fontsize = 10)
# show the plot
plt.show()
# keep the accuracy
bootstrapAccuracy = round(accuracy_score(t[:, 0], tPredictions) * 100, 4)
# keep the f1 score
bootstrapF1 = round(f1_score(t[:, 0], tPredictions) * 100, 4)
# print the update
print("Accuracy: " + str(round(accuracy_score(t[:, 0], tPredictions) * 100, 4)) + "%\nF1 Score: " + str(round(f1_score(t[:, 0], tPredictions) * 100, 4)) + "%")
# print the results of kfolds vs bootstrap
print("\nRESULTS\nKFolds vs Bootstrap\n",
      pd.DataFrame({"Method" : ["KFolds", "Bootstrap", "ABS(Difference)"],
                    "Accuracy" : [str(kfoldsAccuracy) + "%", str(bootstrapAccuracy) + "%", str(round(abs(kfoldsAccuracy - bootstrapAccuracy), 4)) + "%"],
                    "F1 Score" : [str(kfoldsF1) + "%", str(bootstrapF1) + "%", str(round(abs(kfoldsF1 - bootstrapF1), 4)) + "%"]}))

# terminate script
sys.exit()

