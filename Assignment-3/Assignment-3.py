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
for i in range(5):
    df = pd.read_csv('mealData{}.csv'.format(i+1), sep='\t',header=None).head(50)
    total_data=total_data.append(df,ignore_index=True)
label_total=total_data.shape[0]
label_meal=[1 for i in range(label_total)]

mealamount1=pd.read_csv("mealAmountData1.csv",header=None)
mealamount2=pd.read_csv("mealAmountData2.csv",header=None)
mealamount3=pd.read_csv("mealAmountData3.csv",header=None)
mealamount4=pd.read_csv("mealAmountData4.csv",header=None)
mealamount5=pd.read_csv("mealAmountData5.csv",header=None)

#mealamount1=mealamount1.drop(mealamount1.index[[30,31,32]])
mealamount1=mealamount1[:50]
#mealamount2=mealamount2.drop(mealamount2.index[6])
mealamount2=mealamount2[:50]
#mealamount3=mealamount3.drop(mealamount3.index[[19,42]])
mealamount3=mealamount3[:50]
mealamount4=mealamount4[:50]
mealamount5=mealamount5[:50]




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
#pcaTransformed = pd.DataFrame(pcaTransformed)
#pcaTransformed = pd.DataFrame(pcaTransformed)
#pcaTransformed.columns = ['P1', 'P2']
#pcaTransformed.columns = ['P1', '1', 'P3', 'P4']
#print(pcaTransformed)
#pcaTransformed = normalize(pcaTransformed)
#pcaTransformed=pcaTransformed.head(250)
#dbscan = DBSCAN()
#dbscan.fit(pcaTransformed)

X = pcaTransformed[:250]
train_X = X[0:150]

#train_X = np.array(train_X)
test_X = X[150:250]
#test_X = np.array(test_X)



from sklearn.datasets.samples_generator import make_blobs
#from sklearn.cluster import KMeans


test_1=list()

#print(mealamount1.shape())
carbs=[mealamount1[0],mealamount2[0],mealamount3[0],mealamount4[0],mealamount5[0]]
#carbs=[mealamount1[0]]
carbsdata=pd.DataFrame()
carbsdata=pd.concat(carbs)
carbslist = carbsdata.tolist()
#x=total_data
carbsdatalabel=pd.DataFrame()
carbsdatalabel=pd.concat(carbs,ignore_index=True)



carb_X = carbsdatalabel.head(150)
carbsdatalabel_X = np.array(train_X)
carb_Y = carbsdatalabel.tail(100)
carb_Y = np.array(carb_Y)

#carbsdatalabel['index']=[i for i in range(0,len(carbsdatalabel))]
#print(carbsdatalabel)
#carbsdatalabel['label']=[(x-x%10)/10 for x in carbsdatalabel['value']]
#carbsdatalabel['label']=[(carbsdatalabel['label'] if carbsdatalabel['label'] >=x in carbsdatalabel['value']]

def create_bins(lower_bound, width, quantity):
    """ create_bins returns an equal-width (distance) partitioning.
        It returns an ascending list of tuples, representing the intervals.
        A tuple bins[i], i.e. (bins[i][0], bins[i][1])  with i > 0
        and i < quantity, satisfies the following conditions:
            (1) bins[i][0] + width == bins[i][1]
            (2) bins[i-1][0] + width == bins[i][0] and
                bins[i-1][1] + width == bins[i][1]
    """

    bins = []
    for low in range(lower_bound,
                     lower_bound + quantity * width + 1, width):
        bins.append((low, low + width))
    return bins

bins = create_bins(lower_bound=1,
                   width=20,
                   quantity=6)


def find_bin(value, bins):
    """ bins is a list of tuples, like [(0,20), (20, 40), (40, 60)],
        binning returns the smallest index i of bins so that
        bin[i][0] <= value < bin[i][1]
    """

    for i in range(0, len(bins)):
        if value==0:
            return 0
        elif bins[i][0] < value <= bins[i][1]:
            return i+1
    return -3

binned_weights = []
bin_result=[]
#bin_dataframe=pd.DataFrame()
for value in carbsdatalabel:
    bin_index = find_bin(value, bins)
    bin_result.append(bin_index)

    #print(value)
#print(bin_result)
#print("Printing")
#bin_dataframe['bin index']=bin_result
#bin_dataframe['index']=[i for i in range(0, len(bin_result))]
sum=0
total=len(carbsdatalabel)
for i in range(0,len(carbsdatalabel)):
    if carbsdatalabel[i]==bin_result[i]:
        sum=sum+1
#print(sum/total)
#print("carb")
#print(carbsdatalabel)

from collections import Counter

frequencies = Counter(bin_result)
#print(frequencies)
from sklearn.metrics import accuracy_score
#result=accuracy_score(bin_result,dbscan.labels_)
#print(result)


X_normalized_train = normalize(train_X)
Y_normalized_test=normalize(test_X)
with open('training_data.pkl', 'wb') as f:
    pickle.dump(X_normalized_train, f)
# Converting the numpy array into a pandas DataFrame
X_principal_train = pd.DataFrame(X_normalized_train)
from sklearn.cluster import DBSCAN
#dbscan=DBSCAN(eps=0.074, min_samples=2).fit(train_X)
#dbscan=DBSCAN(eps=0.08, min_samples=3).fit(X_principal_train)
dbscan=DBSCAN(eps=0.033, min_samples=2).fit(X_principal_train)
with open('dbscanout.pkl', 'wb') as f:
    pickle.dump(dbscan, f)

for i in range(0, len(train_X)):
    if dbscan.labels_[i] == 0:
        c1 = plt.scatter(pcaTransformed[i,0],pcaTransformed[i,1],c='r',marker='+')
    elif dbscan.labels_[i] == 1:
        c2 = plt.scatter(pcaTransformed[i,0],pcaTransformed[i,1],c='g',marker='o')
    elif dbscan.labels_[i] == -1:
        c3 = plt.scatter(pcaTransformed[i,0], pcaTransformed[i,1], c='b', marker='*')
        #Print("dbscan labels")


label_list_dbscan=[]
label_list_minus=[]
bin_result_dbscan_zero=[[],[],[],[],[],[],[],[],[],[],[]]
for j in range(0,10):
    for i in range(0,len(train_X)):
        if dbscan.labels_[i]==j:
            bin_result_dbscan_zero[j].append(bin_result[i])
        elif dbscan.labels_[i]==-1:
            bin_result_dbscan_zero[10].append(bin_result[i])
    label_list_dbscan.append(max(set(bin_result_dbscan_zero[j]), key = bin_result_dbscan_zero[j].count))

#print(label_list)
label_list_dbscan_f=label_list_dbscan[0:6]

label_list_minus.append(max(set(bin_result_dbscan_zero[10]), key = bin_result_dbscan_zero[10].count))
for j in range(0,10):
    for i in range(0,len(carb_X)):
        if dbscan.labels_[i]==j:
            dbscan.labels_[i]=label_list_dbscan[j]
        elif dbscan.labels_[i]==-1:
            dbscan.labels_[i]=label_list_minus[0]

with open('dbscan.pkl', 'wb') as f:
    pickle.dump(label_list_dbscan, f)





#print(pred_y.labels_)
    #print(pred_y.labels_[j])
#print("Predicted labels")
#print(bin_result_kmeans_zero)
#print(dbscan.labels_[0:150])






import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

#X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
#plt.scatter(X[:,0], X[:,1])



kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit(X_normalized_train)
with open('kmeansout.pkl', 'wb') as f:
    pickle.dump(pred_y, f)

#print("Kmeans labels")
#print(bin_result)
#print("actual")
#print(pred_y.labels_)
plt.scatter(X_normalized_train[:,0], X_normalized_train[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
from sklearn.metrics import accuracy_score
result_kmeans=accuracy_score(bin_result[0:150],pred_y.labels_[0:150])
#print(result_kmeans)


#print("after max")
#Testing Part

label_list=[]
bin_result_kmeans_zero=[[],[],[],[],[],[],[]]
for j in range(0,6):
    for i in range(0,len(carb_X)):
        if pred_y.labels_[i]==j:
            bin_result_kmeans_zero[j].append(bin_result[i])
    label_list.append(max(set(bin_result_kmeans_zero[j]), key = bin_result_kmeans_zero[j].count))

for j in range(0,6):
    for i in range(0,len(carb_X)):
        if pred_y.labels_[i]==j:
            pred_y.labels_[i]=label_list[j]
#print(pred_y.labels_)
    #print(pred_y.labels_[j])
#print("Predicted labels")
#print(pred_y.labels_[0:150])
with open('kmeans.pkl', 'wb') as f:
    pickle.dump(label_list, f)
#result_acc=accuracy_score(pred_y.labels_,bin_result)

correct = 0
for i in range(len(carb_X)):

    if pred_y.labels_[i] == bin_result[i]:
        correct += 1
#print("Print-Acc")
#print(correct/len(carbsdatalabel))


            #bin_result_kmeans_zero.append(bin_result[j])
    #frequencies_zero = Counter(bin_result_kmeans_zero)
        #print(frequencies_zero)
#for bin_v, count in frequencies_zero.most_common(1):
    #print(kmeans.labels_[0])
    #print(frequencies_zero.most_common(1)[0][kmeans.labels_[j]])








#plt.show()
#plt.scatter(pcaTransformed[:, 0], pcaTransformed[:, 1], c=y_kmeans, s=50, cmap='viridis')

#centers = kmeans.cluster_centers_
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

"""""
plt.figure(figsize=(9, 9))
plt.scatter(pcaTransformed['1'], pcaTransformed['2'], c=cvec)

# Building the legend
plt.legend((r, g, b, k), ('Label 0', 'Label 1', 'Label 2', 'Label -1'))

plt.show()

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
# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
distance=0.0
"""""

dist={}
from scipy.spatial import distance
for j in range(0,len(test_X)):
    dist[j]=list()
    for i in range(0,len(train_X)):
        dist[j].append(((distance.euclidean(X_normalized_train[i],Y_normalized_test[j]))))
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
        maj.append(label_list[pred_y.labels_[i[0]]])
        maj_db.append(label_list_dbscan_f[dbscan.labels_[i[0]]])
    list_max.append((max(set(maj),key=maj.count)))
    list_dbscan.append((max(set(maj_db),key=maj_db.count)))
test_result=bin_result[150:250]
test_result_dummy=bin_result[0:100]



