#import matplotlib.pyplot as plt

#from scipy.Date_Series2tpack import Date_Series2t
#import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# from scipy.Date_Series2tpack import Date_Series2t
import csv
import statistics
#from idlelib.idle_test.test_browser import f1
import pickle
import numpy as np
import pandas as pd
import recall as recall
import scipy.fftpack
# import matplotlib.pyplot as plt
from sklearn import decomposition
from PyAstronomy import pyaC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn import decomposition
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from numpy import asarray
from numpy import savetxt


label_meal=list()
label_no_meal=list()
total_data=pd.DataFrame()
for i in range(5):
    df = pd.read_csv('mealData{}.csv'.format(i+1), sep='\t',header=None)
    total_data=total_data.append(df,ignore_index=True)
label_total=total_data.shape[0]
label_meal=[1 for i in range(label_total)]


for i in range(5):
    ef = pd.read_csv('Nomeal{}.csv'.format(i + 1), sep='\t', header=None)
    total_data = total_data.append(ef, ignore_index=True)

label_total_1 = total_data.shape[0]-label_total
label_no_meal = [0 for i in range(label_total_1)]
label_final=label_meal+label_no_meal
label_final=np.array(label_final)
total_data = total_data[0].str.split(',', expand=True)
total_data.fillna(0,inplace=True)
total_data=total_data.replace("NaN",0)
total_data.dropna(how='all')
total_data=total_data.astype(float)

#CGM Velocity and Zero Crossing

data_tran=total_data.T
cgm_velocity=data_tran.pct_change()
cgm_velocity=cgm_velocity.T

cgm_velocity=np.array(cgm_velocity)
x=[i*5 for i in range(total_data.shape[1])]
cgm_velocity_final=list()
for i in range(total_data.shape[0]):
    zero_crossing=np.where(np.diff(np.sign(cgm_velocity[i])))[0]
    cgm_velocity_final.append(zero_crossing[0:2])
cgm_velocity_final=pd.DataFrame(cgm_velocity_final)
cgm_velocity_final=cgm_velocity_final.to_numpy()

#Fast_Fourier_Transform
total_data_arr=np.array(total_data)
Fast_Fourier_Transform=abs(np.fft.fft(total_data_arr))
fft_freq=np.fft.fftfreq(total_data_arr.shape[-1])
Fast_Fourier_Transform=np.array(Fast_Fourier_Transform)
fft_freq=np.array(fft_freq)
Fourier_peak=list()
Fourier_frequency=list()
for i in range(len(Fast_Fourier_Transform)):
    index=np.argsort(Fast_Fourier_Transform)[i][-16:]
    peak=Fast_Fourier_Transform[i][index]
    Fourier_peak.append(peak)
    freq=abs(fft_freq[index])
    freq.sort()
    fr=freq[[0,1,3,5,7,9,11,13]]
    Fourier_frequency.append(fr)
Fourier_peak=np.unique(Fourier_peak,axis=1)
Fourier_peak=np.array(Fourier_peak)

#Polynomial Fit
Poly_fit=list()
x=[i*5 for i in range(total_data.shape[1])]
for i in range(len(total_data)):
    poly_fit=np.polyfit(x,total_data.iloc[i],3)
    Poly_fit.append(poly_fit)
Poly_fit=np.array(Poly_fit)


# Rolling Mean and Rolling Deviation
tr = total_data.T
rolling_mean = tr.rolling(window=3,min_periods=3).mean()
rolling_dev = tr.rolling(window=3,min_periods=3).std()
rolling_mean=rolling_mean.T
rolling_dev=rolling_dev.T
rolling_mean_arr = np.array(rolling_mean)
rolling_dev_arr=np.array(rolling_dev)


#Calculating Feature Matrix
Feature_matrix=np.append(rolling_mean_arr,rolling_dev_arr,axis=1)
Feature_matrix_1=np.append(Feature_matrix,Fourier_peak,axis=1)
Feature_matrix_2=np.append(Feature_matrix_1,cgm_velocity_final,axis=1)
Feature_final=np.append(Feature_matrix_2,Poly_fit,axis=1)
Feature_final=np.nan_to_num(Feature_final)
#Feature_final.fillna(0,inplace=True)


from numpy import *
where_are_NaNs = isnan(Feature_final)
Feature_final[where_are_NaNs] = 0

#PCA Calculation
sc = StandardScaler()
Feature_std = sc.fit_transform(Feature_final)
pca = decomposition.PCA(n_components=5)
Feature_std_pca = pca.fit_transform(Feature_std)
Feature_std_pca=Feature_std_pca.transpose()
PCA_mul=np.dot(Feature_std_pca,Feature_final)

pcaTransformed = pca.transform(Feature_final)
k=5

# k-fold cross validation and MLP Classifier
np.random.seed(1)
X=pcaTransformed
y= label_final
accuracy = []
f1=[]
p1=[]
recall=[]
k_fold = KFold(n_splits = 10, shuffle = True)
classifier = MLPClassifier(alpha = 0.3, max_iter = 1000)
for train_index, test_index in k_fold.split(X):
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    classifier.fit(Xtrain, ytrain)
    predicted = classifier.predict(Xtest)
    accuracy.append(accuracy_score(ytest, predicted))
    f1.append(f1_score(ytest, predicted))
    p1.append(average_precision_score(ytest, predicted))
    recall.append(recall_score(ytest, predicted))

filename = 'final_model.pickle'
pickle.dump(classifier, open(filename, 'wb'))
print("Accuracy: ", (np.mean(accuracy)*100))
print("F1 Score", (np.mean(f1)*100))
print("Precision",(np.mean(p1)*100))
print("Recall",(np.mean(recall)*100))

