import pandas as pd
import matplotlib.pyplot as plt
import scipy
#from scipy.Date_Series2tpack import Date_Series2t
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore')
#Loading data
Date_Series1 = pd.read_csv('Datafolder//CGMDatenum/CGMDatenumLunchPat1.csv')
Date_Series2= pd.read_csv('Datafolder//CGMDatenum/CGMDatenumLunchPat2.csv')
Date_Series3=pd.read_csv('Datafolder//CGMDatenum/CGMDatenumLunchPat3.csv')
Date_Series4=pd.read_csv('Datafolder//CGMDatenum/CGMDatenumLunchPat4.csv')
Date_Series5=pd.read_csv('Datafolder//CGMDatenum/CGMDatenumLunchPat5.csv')
CGM_Series1 = pd.read_csv('Datafolder//CGMDatenum/CGMSeriesLunchPat1.csv')
CGM_Series2= pd.read_csv('Datafolder//CGMDatenum/CGMSeriesLunchPat2.csv')
CGM_Series3= pd.read_csv('Datafolder//CGMDatenum/CGMSeriesLunchPat3.csv')
CGM_Series4= pd.read_csv('Datafolder//CGMDatenum/CGMSeriesLunchPat4.csv')
CGM_Series5 = pd.read_csv('Datafolder//CGMDatenum/CGMSeriesLunchPat5.csv')

#Concatenating 5 Patient's Data
glucose_level=pd.concat([CGM_Series1,CGM_Series2,CGM_Series3,CGM_Series4,CGM_Series5])
time_stamp=pd.concat([Date_Series1,Date_Series2,Date_Series3,Date_Series4,Date_Series5])

#Replacing empty fields with average of two adjacent NaN values
glucose_level.fillna(0,inplace=True)
time_stamp.fillna(0,inplace=True)

#Fast_Fourier_Transform
Fast_Fourier_Transform=abs(scipy.fft.fft(glucose_level))
fft_freq=scipy.fft.fftfreq(42, d=1.0)

plt.stem(fft_freq,Fast_Fourier_Transform[7])
Fast_Fourier_Transform=np.array(Fast_Fourier_Transform)
fft_freq=np.array(fft_freq)
Fourier_peak=list()
Fourier_frequency=list()
for i in range(len(Fast_Fourier_Transform)):
    index=np.argsort(Fast_Fourier_Transform)[i][-9:]

    peak=Fast_Fourier_Transform[i][index]
    Fourier_peak.append(peak)
    freq=abs(fft_freq[index])
    freq.sort()
    fr=freq[[0,1,3,5,7]]
    Fourier_frequency.append(fr)
Fourier_peak=np.unique(Fourier_peak,axis=1)
Fourier_peak=np.array(Fourier_peak)
print("Fourier Top 5 Peaks")
print(Fourier_peak.shape)
print(Fourier_peak)
Fourier_frequency=np.array(Fourier_frequency)
print("Fourier top 5 frequencies")
print(Fourier_frequency.shape)
print(Fourier_frequency)


#Feauture 2 Rolling Mean and Deviation
rolling_mean = glucose_level.rolling(window=3,min_periods=3).mean()
rolling_dev = glucose_level.rolling(window=3,min_periods=3).std()
time_stamp_array = np.array(time_stamp)
rolling_mean_arr = np.array(rolling_mean)
rolling_dev_arr=np.array(rolling_dev)
time_series = [i for i in range(len(time_stamp_array[32]))]
plt.plot(time_series, rolling_mean_arr[32])
plt.title("Rolling Mean")
plt.ylabel("Glucose Level")
plt.xlabel("Time")

plt.show()
plt.plot(time_series, rolling_dev_arr[32])
plt.title("Rolling Deviation")
plt.ylabel("Glucose Level")
plt.xlabel("Time")
plt.show()
print("Rolling_Mean")
print(rolling_mean)
print("Rolling_Deviation")
print(rolling_dev)

#Feauture 3 Polynomial Fit
Poly_fit=list()
x=[i*5 for i in range(len(time_stamp.iloc[100]))]
for i in range(len(glucose_level)):
    poly_fit=np.polyfit(x,glucose_level.iloc[i],5)
    Poly_fit.append(poly_fit)
Poly_fit=np.array(Poly_fit)
x=[i*5 for i in range(len(time_stamp.iloc[10]))]
plt.plot(x,np.polyval(Poly_fit[7],x),label='Poly_fitFit')
plt.plot(x,glucose_level.iloc[7],label='Glucose_level')
plt.legend()
plt.title("Poly_fit")
plt.ylabel("Glucose Level")
plt.xlabel("Time")
plt.show()


#Feature 4 Inter Quartile Range
glucose_level = np.array(glucose_level)
Inter_Qua_Range = []
for i in range(len(glucose_level)):
    Quartile1 = np.percentile(glucose_level[i], 25, interpolation = 'midpoint')
    Quartile3 = np.percentile(glucose_level[i], 75, interpolation = 'midpoint')
    Inter_Qua_Range.append(Quartile3 - Quartile1)
Inter_Qua_Range = np.array(Inter_Qua_Range)
Inter_Quartile=list()
Inter_Quartile.append(Inter_Qua_Range)
Inter_Quartile=np.array(Inter_Quartile)
Inter_Quartile_arr = Inter_Quartile.T
print("Inter_Quartile_Range")
print(Inter_Quartile.shape)
print(Inter_Quartile)

#Calculating Feature Matrix
Feature_matrix=np.append(rolling_mean_arr,rolling_dev_arr,axis=1)
Feature_matrix_1=np.append(Feature_matrix,Fourier_peak,axis=1)
Feature_matrix_2=np.append(Feature_matrix_1,Fourier_frequency,axis=1)
Feature_matrix_3=np.append(Feature_matrix_2,Inter_Quartile_arr,axis=1)
Feature_final=np.append(Feature_matrix_3,Poly_fit,axis=1)
#Feature_final.fillna(0,inplace=True)


from numpy import *
where_are_NaNs = isnan(Feature_final)
Feature_final[where_are_NaNs] = 0
print("Feature Matrix")
print(Feature_final.shape)
print(Feature_final)

#PCA Calculation
sc = StandardScaler()
Feature_std = sc.fit_transform(Feature_final)
pca = decomposition.PCA(n_components=5)
Feature_std_pca = pca.fit_transform(Feature_std)
Feature_std_pca=Feature_std_pca.transpose()
PCA_mul=np.dot(Feature_std_pca,Feature_final)

#Eigen Value Calculation
cov_mat = np.cov(Feature_std.T)
print("Covariance")
print(cov_mat)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print("Eigen Values")
print(eig_vals)
print("Eigen Vectors")
print(eig_vecs)
pcaExpVariance = pca.explained_variance_
print("Explained Variance = ", pcaExpVariance)
pcaratio= pca.explained_variance_ratio_
print("PCA ratio:",sum(pcaratio))

#PCA Feature Plotting
pcaTransformed = pca.transform(Feature_final)
print("After PCA")
print(pcaTransformed.shape)
print(pcaTransformed)
plt.bar(list(range(0, 5)), pcaExpVariance)
plt.title("Variance Explained")
plt.show()
plt.scatter(list(range(0, 216)), pcaTransformed[0:216, 0])
plt.title("PCA-Feature1")
plt.ylabel("Feature Vectors")
plt.xlabel("Time")
plt.show()
plt.scatter(list(range(0, 216)), pcaTransformed[0:216, 1])
plt.title("PCA-Feature2")
plt.ylabel("Feature Vectors")
plt.xlabel("Time")
plt.show()
plt.scatter(list(range(0, 216)), pcaTransformed[0:216, 2])
plt.title("PCA-Feature3")
plt.ylabel("Feature Vectors")
plt.xlabel("Time")
plt.show()
plt.scatter(list(range(0, 216)), pcaTransformed[0:216, 3])
plt.title("PCA-Feature4")
plt.ylabel("Feature Vectors")
plt.xlabel("Time")
plt.show()
plt.scatter(list(range(0, 216)), pcaTransformed[0:216, 4])

plt.title("PCA-Feature5")
plt.ylabel("Feature Vectors")
plt.xlabel("Time")
plt.show()