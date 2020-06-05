#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# from scipy.Date_Series2tpack import Date_Series2t
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# from scipy.Date_Series2tpack import Date_Series2t
# from idlelib.idle_test.test_browser import f1
# import recall as recall
# import matplotlib.pyplot as plt
# from PyAstronomy import pyaC
import numpy as np
import pandas as pd
import pickle
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import sklearn.cluster

label_meal=list()
label_no_meal=list()
total_data=pd.DataFrame()
total_data_carb=pd.DataFrame()
df = pd.read_csv('proj3_test.csv', sep='\t',header=None)
total_data=total_data.append(df,ignore_index=True)
label_total=total_data.shape[0]
""""
carb = pd.read_csv('carb.csv', sep='\t',header=None)
total_data_carb=total_data_carb.append(carb,ignore_index=True)
#label_total=total_data.shape[0]

"""

label_total_1 = total_data.shape[0]-label_total
label_no_meal = [0 for i in range(label_total_1)]
label_final=label_meal+label_no_meal
label_final=np.array(label_final)
total_data = total_data[0].str.split(',', expand=True)

#total_data.replace(r'^\s*$', np.nan, regex=True)
#total_data.fillna(0,inplace=True)
total_data.replace("",np.nan,inplace=True)
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
#pcaTransformed = pd.DataFrame(pcaTransformed)
#pcaTransformed = pd.DataFrame(pcaTransformed)
#pcaTransformed.columns = ['P1', 'P2']
#pcaTransformed.columns = ['P1', '1', 'P3', 'P4']
#print(pcaTransformed)
#pcaTransformed = normalize(pcaTransformed)
#pcaTransformed=pcaTransformed.head(250)
#dbscan = DBSCAN()
#dbscan.fit(pcaTransformed)
Test_data = normalize(pcaTransformed)


#test_X = np.array(test_X)
with open('training_data.pkl', 'rb') as f:
    X_normalized_train = pickle.load(f)

with open('kmeans.pkl', 'rb') as f:
    label_list = pickle.load(f)
with open('dbscan.pkl', 'rb') as f:
    label_list_dbscan = pickle.load(f)
with open('kmeansout.pkl', 'rb') as f:
    pred_y = pickle.load(f)
with open('dbscanout.pkl', 'rb') as f:
    dbscan = pickle.load(f)
dist={}
from scipy.spatial import distance
for j in range(0,len(Test_data)):
    dist[j]=list()
    for i in range(0,len(X_normalized_train)):
        dist[j].append(((distance.euclidean(X_normalized_train[i],Test_data[j]))))
#print(dist[20])
min_list=[]
for ele in dist:
    x=[dist[ele].index(i) for i in dist[ele]]
    min_list.append(sorted(zip(x,dist[ele]),key=lambda t: t[1])[0:5])
#print(min_list)
list_five={}
list_max=[]
list_dbscan=[]

for j in range(len(min_list)):
    maj=list()
    maj_db=list()
    for i in min_list[j]:
        maj.append(label_list[pred_y.labels_[i[0]]]+1)
        maj_db.append(label_list_dbscan[dbscan.labels_[i[0]]]+1)
    list_max.append((max(set(maj),key=maj.count)))
    list_dbscan.append((max(set(maj_db),key=maj_db.count)))
list_final=[]
for i in range(0,len(list_max)):
    list_final.append([list_dbscan[i],list_max[i]])


df = pd.DataFrame(list_final)


df.to_csv('Project3_Output.csv', index=False,header=None)



