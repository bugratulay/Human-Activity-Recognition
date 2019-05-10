import pandas as pd
import time
from matplotlib import pyplot
import numpy as np
import itertools
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, StandardScaler , MinMaxScaler
from sklearn.pipeline import Pipeline
import sklearn.neural_network as nn
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold,cross_val_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier,AdaBoostClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.manifold import TSNE
#Neural Network
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD,Adam


train = pd.read_csv('/home/bugratulay/Downloads/train.csv')
test = pd.read_csv('/home/bugratulay/Downloads/test.csv')



peek=train.head()
print(peek)
print(train.Activity.value_counts()) # counting activities
print(train.shape)
print(test.shape)
#shuffling data

test =shuffle(test) # ruin the sequence data
train=shuffle(train)
print(train.head())

#Separate Input and Output Labels
# dropping Activity and subject
trainData = train.drop(['Activity','subject'], axis=1).values
trainLabel = train.Activity.values

testData = test.drop(['Activity','subject'], axis=1).values
testLabel = test.Activity.values

#encoding Labels

encoder = LabelEncoder()

#encoding test labels

encoder.fit(testLabel)
testLabelE = encoder.transform(testLabel)

#encoding train labels

encoder.fit(trainLabel)
trainLabelE = encoder.transform(trainLabel)


### DATA VISUALIZATION
# #undo shuffling
# train.reset_index(drop=True, inplace=True)
# test.reset_index(drop=True, inplace=True)
#
# trainDataDF = train.drop(['Activity','subject'], axis=1)
# testDataDF = test.drop(['Activity','subject'], axis=1)
# trainLabelDF=train[['Activity']]
# testLabelDF=test[['Activity']]
#
# data = [trainDataDF, testDataDF]
# all_data=pd.concat(data)
# all_data.reset_index(drop=True, inplace=True)
# labels=[trainLabelDF,testLabelDF]
# all_labels= pd.concat(labels)
# all_labels.reset_index(drop=True, inplace=True)
# pca = PCA(n_components=2)
# #pca1 = PCA(.8) # capture the %80 of the variance
# #pca1.fit_transform(all_data) # see pca.n_components_(11) to know how many principal comp.s are needed for %80variance
#
# principalComponents = pca.fit_transform(all_data)
# print(pca.explained_variance_ratio_)
#
# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['principal component 1', 'principal component 2'])
#
# print(principalDf.shape)
# print(all_labels.shape)
#
# finalDf = pd.concat([principalDf,all_labels], axis = 1)
#
# fig = pyplot.figure(figsize = (8,8))
# ax = fig.add_subplot(111) #2d
# #ax = fig.add_subplot(111,projection='3d') #3d
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2D Projection of the data with PCA', fontsize = 15)
#
# targets = ['WALKING', 'STANDING', 'LAYING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING']
# colors = ['r', 'g', 'b','k','y','m']
#
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['Activity'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                 , c = color
#                , s = 25)
# ax.legend(targets)
# ax.grid()
#
#
#
# n_sne = 7000
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(all_data)
#
# tsneDf = pd.DataFrame(data = tsne_results
#              , columns = ['tsne 1', 'tsne 2'])
#
# final_tsneDf = pd.concat([tsneDf,all_labels], axis = 1)
#
# fig = pyplot.figure(figsize = (8,8))
# ax = fig.add_subplot(111)
# ax.set_xlabel('t-SNE dimension 1', fontsize = 15)
# ax.set_ylabel('t-SNE dimension 2', fontsize = 15)
#
# ax.set_title('2D Projection of the data with t-SNE', fontsize = 15)
#
# targets = ['WALKING', 'STANDING', 'LAYING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING']
# colors = ['r', 'g', 'b','k','y','m']
#
# for target, color in zip(targets,colors):
#     indicesToKeep = final_tsneDf['Activity'] == target
#     ax.scatter(final_tsneDf.loc[indicesToKeep, 'tsne 1']
#                , final_tsneDf.loc[indicesToKeep, 'tsne 2']
#                , c = color
#                , s = 25)
# ax.legend(targets)
# ax.grid()
# pyplot.show()



# ## Multiple Classifier Evaluation
# start=time.time()
# # pca= PCA(0.9) # capture the %85 of the variance
# # pca.fit(trainData)
# # Reduced_trainData=pca.transform(trainData)
# # Reduced_testData=pca.transform(testData) # use the same transform for test
# # Standardize the dataset
# pipelines = []
# pipelines.append(('LR', Pipeline([('Scaler', StandardScaler()),('LR',
# LogisticRegression(C=100))])))
# pipelines.append(('QDA', Pipeline([('Scaler', StandardScaler()),('LDA',
# QuadraticDiscriminantAnalysis())])))
# pipelines.append(('KNN', Pipeline([('Scaler', StandardScaler()),('KNN',
# KNeighborsClassifier(7))])))
# pipelines.append(('D.Tree', Pipeline([('Scaler', StandardScaler()),('CART',
# DecisionTreeClassifier())])))
# pipelines.append(('NB', Pipeline([('Scaler', StandardScaler()),('NB',
# GaussianNB())])))
# pipelines.append(('SVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC(C=100,kernel='rbf',decision_function_shape='ovo' ))])))
# results = []
# names = []
# total_time=[]
# for name, model in pipelines:
#     model.fit(trainData,trainLabelE)
#     y_pred = model.predict(testData)
#     results.append(accuracy_score(y_pred,testLabelE))
#     names.append(name)
#     end=time.time()
#     total_time.append(start-end)
#     msg = "%s: %f:" % (name, accuracy_score(y_pred,testLabelE))
#     print(msg)
#
# score_df = pd.DataFrame({'Model':names,'Scores':results, 'Time':total_time}).set_index('Model')
# print(score_df)
# ax=score_df.plot.bar()
# ax.set_xticklabels(score_df.index,rotation=45,fontsize=10)
# pyplot.grid(True)


#Classification Models

#Decision Tree
#SVM
# Neural Network
# Random Forest
# Gradient Boosting Method
# DNN
# Autoencoder

# Logistic Regression
start = time.time()
model = LogisticRegression(C=0.1)
print(model)
model.fit(trainData,trainLabelE)
LogisticReg_Score=model.score(testData,testLabelE)
print(LogisticReg_Score)  # 0.95
end = time.time()
total_time =end-start
print('Time of Logistic Regression %.4f' % total_time)

# GridSearch for Logistic Regressor C parameter
'''
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
param_grid = dict(C=c_values)
model = LogisticRegression()
kfold = KFold(n_splits=10, random_state=7)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
grid_result = grid.fit(trainData, trainLabel)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
'''

# try PCA and LDA for Logistic Regression


start = time.time()
pca= PCA(0.9) # capture the %85 of the variance
pca.fit(trainData)
Reduced_trainData=pca.transform(trainData)
Reduced_testData=pca.transform(testData) # use the same transform for test

model.fit(Reduced_trainData,trainLabelE)
PCA_Score=model.score(Reduced_testData,testLabelE)
print("PCA Score:%f" %(PCA_Score)) #worse performance but train faster
end = time.time()
total_time =end-start
print('Time of PCA %.4f' % total_time)

start = time.time()
LDA=LinearDiscriminantAnalysis()
LDA.fit(trainData,trainLabelE)
LDA_traindata=LDA.transform(trainData)
LDA_testdata=LDA.transform(testData)
model.fit(LDA_traindata,trainLabelE)
lda_score=model.score(LDA_testdata,testLabelE)
print("LDA Score:%f" %(lda_score)) #worse performance but train faster
end = time.time()
total_time =end-start
print('Time of LDA %.4f' % total_time)

# n_sne = 7000
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_traindata=tsne.fit_transform(trainData)
# tsne_testdata=tsne.fit_transform(testData)
# model.fit(tsne_traindata,trainLabelE)
# label=model.predict(tsne_testdata)
# tsne_score = model.score(tsne_testdata,testLabelE)
# print("TSNE Score:%f" %(tsne_score)) #worse performance but train faster

# Scale Features

scaler = MinMaxScaler()
scaler.fit(LDA_traindata)
Scaled_trainData = scaler.transform(LDA_traindata)
Scaled_testData = scaler.transform(LDA_testdata)

model.fit(Scaled_trainData,trainLabelE)
Scaled_Score=model.score(Scaled_testData,testLabelE)
print("Scaled Data Score:%f" %(Scaled_Score)) #0.947


# kfold=10
# cv_results = cross_val_score(model, Scaled_trainData, trainLabelE, cv=kfold, scoring='accuracy')
# msg = " %f (%f)" % (cv_results.mean(), cv_results.std())
# print(msg)


# appylying supervised neural network using multi-layer perceptron

mlpSGD  =  nn.MLPClassifier(hidden_layer_sizes=(90,) \
                        , max_iter=1000, alpha=1e-4  \
                        , solver='sgd' ,verbose=10   \
                        , tol=1e-19    , random_state =1 \
                        , learning_rate_init=.001)

mlpADAM  =  nn.MLPClassifier(hidden_layer_sizes=(90,) \
                        , max_iter=1000, alpha=1e-4  \
                        , solver='adam' ,verbose=10   \
                        , tol=1e-19    , random_state =1 \
                        , learning_rate_init=.001)

nnModelADAM = mlpADAM.fit(Scaled_trainData, trainLabelE)

predicted = nnModelADAM.predict(Scaled_testData)
matrix = confusion_matrix(testLabelE, predicted)
print(matrix)
print(nnModelADAM.score(Scaled_testData,testLabelE))

## Keras Neural Network
n_input = Scaled_trainData.shape[1] # number of features
n_output = 6 # number of possible labels
n_samples =  Scaled_trainData.shape[0] # number of training samples
n_hidden_units = 40
Y_test=to_categorical(testLabelE) # one-hot encoded labels
Y_train=to_categorical(trainLabelE)

def create_model():
    model = Sequential()
    model.add(Dense(n_hidden_units,input_dim=n_input,activation="relu"))
    model.add(Dense(n_hidden_units,input_dim=n_input,activation="relu"))
    model.add(Dense(n_output,activation="softmax"))
    # Compile Model
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=create_model, epochs=20, batch_size=10, verbose=False)
estimator.fit(Scaled_trainData,Y_train)
print("Keras Classifier Score:{}".format(estimator.score(Scaled_testData,Y_test))) # 0.95


# Ensemble Methods
Y_test=to_categorical(testLabelE) # one-hot encoded labels
Y_train=to_categorical(trainLabelE)

model=ExtraTreesClassifier(n_estimators=500)
model.fit(Scaled_trainData,Y_train)
print("ExtraTree Classifier results %.3f" %model.score(Scaled_testData,Y_test)) # 0.915

model=RandomForestClassifier(n_estimators=500)
model.fit(Scaled_trainData,Y_train)
print("RandomForest Classifier results %.3f" %model.score(Scaled_testData,Y_test)) # 0.90

##Knearest Neighbors

clf= KNeighborsClassifier(n_neighbors=12)
knnModel = clf.fit(Scaled_trainData,trainLabelE)
y_te_pred = clf.predict(Scaled_testData)

acc = accuracy_score(testLabelE,y_te_pred)
print("K-Nearest Neighbors Accuracy: %.5f" %(acc)) #.907

##

#Keras model with different optimizers and dropout layers
Y_test=to_categorical(testLabelE) # one-hot encoded labels
Y_train=to_categorical(trainLabelE)

model = Sequential()
model.add(Dense(64, input_dim=Scaled_trainData.shape[1], activation="relu"))
model.add(Dropout(0.5)) # To avoid overfitting
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(6, activation="softmax"))

sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])
model.fit(Scaled_trainData,Y_train,nb_epoch=30,batch_size=128)
score = model.evaluate(Scaled_testData,Y_test,batch_size=128)
print(score)  # 0.956

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(Scaled_trainData,Y_train,nb_epoch=30,batch_size=128)
score = model.evaluate(Scaled_testData,Y_test,batch_size=128)
print(score)  # 0.95

# ## Feature Selection and SVM
#
# model = SVC(kernel="linear")
# model = LogisticRegression()
# rfe = RFE(model, 3)
# fit = rfe.fit(Scaled_trainData, trainLabelE)
# print("Num Features: %d" % fit.n_features_)
# print("Selected Features: %s" % fit.support_)
# print("Feature Ranking: %s" % fit.ranking_)
#
# best_features =[]
# for ix,val in enumerate(rfe.support_):
#     if val==True:
#         best_features.append(testData[:,ix])
#


##Ploting Confusion Matrix
def plot_confusion_matrix (cm, classes, normalize=False, title='Confusion Matrix', cmap=pyplot.cm.Blues,
                           decsnTreeClf=None):
    #This function prints and plots the confusion matrix.
    pyplot.imshow(cm,interpolation='nearest',cmap=cmap)
    pyplot.title(title)
    pyplot.colorbar()
    tick_marks=np.arange(len(classes))
    pyplot.xticks(tick_marks,classes,rotation=45)
    pyplot.yticks(tick_marks,classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pyplot.text(j,i,cm[i,j],horizontalalignment="center", color="white" if cm[i,j]> thresh else "black" )

    pyplot.tight_layout()
    pyplot.ylabel('True Label')
    pyplot.xlabel('Predicted Label')
    pyplot.show()
decsnTreeClf = DecisionTreeClassifier(criterion='entropy')
tree= decsnTreeClf.fit(trainData,trainLabelE)
testPred = tree.predict(testData)

acc= accuracy_score(testLabelE,testPred)
cfs=confusion_matrix(testLabelE,testPred)



print("Accuracy: %f" %acc)

pyplot.figure()
class_names = encoder.classes_
plot_confusion_matrix(cfs,classes=class_names,title="DecisionTree Confusuion Matrix")


## Multiple Classifier Evaluation

# Standardize the dataset
pipelines = []
pipelines.append(('LR', Pipeline([('Scaler', StandardScaler()),('LR',
LogisticRegression(C=100))])))
pipelines.append(('QDA', Pipeline([('Scaler', StandardScaler()),('LDA',
QuadraticDiscriminantAnalysis())])))
pipelines.append(('KNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsClassifier(7))])))
pipelines.append(('D.Tree', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeClassifier())])))
pipelines.append(('NB', Pipeline([('Scaler', StandardScaler()),('NB',
GaussianNB())])))
pipelines.append(('SVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC(C=100,kernel='rbf',decision_function_shape='ovo' ))])))
results = []
names = []
for name, model in pipelines:
    model.fit(trainData,trainLabelE)
    y_pred = model.predict(LDA_testdata)
    results.append(accuracy_score(y_pred,testLabelE))
    names.append(name)
    msg = "%s: %f" % (name, accuracy_score(y_pred,testLabelE))
    print(msg)

score_df = pd.DataFrame({'Model':names,'Scores':results}).set_index('Model')
print(score_df)
ax=score_df.plot.bar()
ax.set_xticklabels(score_df.index,rotation=45,fontsize=10)
pyplot.grid(True)


# Tune scaled SVM
#Best: 0.987622 using {'C': 100, 'kernel': 'rbf'}

scaler = StandardScaler().fit(trainData)
rescaledX = scaler.transform(trainData)
c_values = [0.1, 1.0, 100, 1000]
kernel_values = ['linear', 'rbf']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC()
kfold = KFold(n_splits=5, random_state=7)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
grid_result = grid.fit(rescaledX, trainLabelE)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
