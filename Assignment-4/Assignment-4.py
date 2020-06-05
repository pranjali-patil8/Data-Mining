import csv
import statistics
#from idlelib.idle_test.test_browser import f1
import pickle
import numpy as np
from mlxtend.frequent_patterns import apriori
from numpy import asarray
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

freq_final=[]
max_conf_final=[]
min_conf_final=[]
for i in range(1,6):
    total_data = pd.DataFrame()
    total_data_cgm = pd.DataFrame()
    total_data_lunch = pd.DataFrame()
    df = pd.read_csv('DataFolder//InsulinBolusLunchPat'+str(i)+'.csv')
    total_data = total_data.append(df, ignore_index=True)
    #total_data = total_data[0].str.split(',', expand=True)
    total_data.fillna(0.0,inplace=True)
    total_data=total_data.replace("NaN",0)
    total_data.dropna(how='all')
    ef = pd.read_csv('DataFolder//CGMSeriesLunchPat'+str(i)+'.csv')
    total_data_cgm = total_data_cgm.append(ef, ignore_index=True)
    #total_data_cgm = total_data_cgm[0].str.split(',', expand=True)
    total_data_cgm.fillna(0,inplace=True)
    total_data_cgm=total_data_cgm.replace("NaN",0)
    total_data_cgm.dropna(how='all')
    gf = pd.read_csv('DataFolder//CGMSeriesLunchPat'+str(i)+'.csv')
    total_data_lunch = total_data_lunch.append(gf, ignore_index=True)
    #total_data_lunch = total_data_lunch[0].str.split(',', expand=True)
    total_data_lunch.fillna(0,inplace=True)
    total_data_lunch=total_data_lunch.replace("NaN",0)
    total_data_lunch.dropna(how='all')

    list_final=list()
    total_data_num=total_data.to_numpy()

    for i in total_data_num:
        list_final.append(max(i))


    list_data_cgm=list()
    total_data_cgm=total_data_cgm.to_numpy()
    for j in total_data_cgm:
        list_data_cgm.append(max(j))
    list_data_lunch=list()

    total_data_lunch=total_data_lunch.to_numpy()
    for k in range(0,len(total_data_lunch)):
        list_data_lunch.append((total_data_lunch[k][5]))


    def create_bins(lower_bound, width, quantity):

        bins = []
        for low in range(lower_bound,
                         lower_bound + quantity * width + 1, width):
            bins.append((low, low + width))
        return bins

    bins = create_bins(lower_bound=40,
                       width=10,
                       quantity=36)


    def find_bin(value, bins):


        for i in range(0, len(bins)):
            if value==0:
                return 0
            elif bins[i][0] < value <= bins[i][1]:
                return i+1
        return -3

    binned_weights = []
    bin_result=[]

    for value in list_data_cgm:
        bin_index = find_bin(value, bins)
        bin_result.append(bin_index+100)
    lunch_bin=[]
    for value in list_data_lunch:
        bin_index = find_bin(value, bins)
        lunch_bin.append(bin_index+1000)

    x=list()
    for i in range(len(bin_result)):
        x.append([bin_result[i],lunch_bin[i],list_final[i]])
    x=np.array(x)







    from apyori import apriori

    association_rules = apriori(x, min_support=0.01, min_confidence=0, min_lift=3, min_length=3)
    association_results = list(association_rules)
    assoc_rule=pd.DataFrame(association_results)

    #Calculating Frequent itemsets
    #Calculating confidence
    list_1=[]
    list_2=[]
    list_4=[]
    print("")
    for i in association_results:
        if len(i[2])==6:
            list_1.append([int(list(list(i[2][5][0]))[0]),int(list(list(i[2][5][0]))[1]),float(list(i[2][5][1])[0])])
            if (i[2][5][2])==1:
                list_2.append([int(list(list(i[2][5][0]))[0]), int(list(list(i[2][5][0]))[1]), float(list(i[2][5][1])[0])])


    min_1=2
    for i in association_results:
        if len(i[2]) == 6:
            if min_1>i[2][5][2]:
                min_1=i[2][5][2]
    if min_1==1:
        min_1=2
    else:
        min_cal=min_1
    list_3=[]
    for i in association_results:
        if len(i[2])==6:
            if (i[2][5][2])==min_cal:
                list_3.append([int(list(list(i[2][5][0]))[0]), int(list(list(i[2][5][0]))[1]), float(list(i[2][5][1])[0])])
    min_conf=[]

    a=0
    b=0

    for i in range(0,len(list_1)):
        if (list_1[i][0]>=1000 ):
            a=list_1[i][0]
            list_1[i][0]=list_1[i][1]
            list_1[i][1]=a

    for i in range(0,len(list_1)):
        list_1[i][1]=int(list_1[i][1]-1000)
    for i in range(0,len(list_1)):
        list_1[i][0]=int(list_1[i][0]-100)
    for i in range(0,len(list_2)):
        if (list_2[i][0]>=1000):
            b=list_2[i][0]
            list_2[i][0]=list_2[i][1]
            list_2[i][1]=b
    for i in range(0,len(list_2)):
        list_2[i][1]=int(list_2[i][1]-1000)
    for i in range(0,len(list_2)):
        list_2[i][0]=int(list_2[i][0]-100)
    c=0
    for i in range(0,len(list_3)):
        if (list_3[i][0]>=1000):
            c=list_3[i][0]
            list_3[i][0]=list_3[i][1]
            list_3[i][1]=c
    for i in range(0,len(list_3)):
        list_3[i][1]=int(list_3[i][1]-1000)
    for i in range(0,len(list_3)):
        list_3[i][0]=int(list_3[i][0]-100)
    freq_item=[]

    for i in list_1:
        freq_item.append(['{'+str(i[0])+','+str(i[1])+','+str(i[2])+'}'][0])

    max_conf=[]
    for i in list_2:
        max_conf.append(['{'+str(i[0])+','+str(i[1])+'->'+str(i[2])+'}'][0])
    for i in list_3:
        min_conf.append(['{'+str(i[0])+','+str(i[1])+'->'+str(i[2])+'}'][0])

    for i in freq_item:
        freq_final.append(i)
    for i in max_conf:
        max_conf_final.append(i)
    for i in min_conf:
        min_conf_final.append(i)




freq_item_d=pd.DataFrame(freq_final)
freq_item_d.to_csv("Frequent_item_final.csv",header=None,index=None)
max_conf_d=pd.DataFrame(max_conf_final)
max_conf_d.to_csv("Maximum_confidence_final.csv",header=None,index=None)
min_conf_d=pd.DataFrame(min_conf_final)
min_conf_d.to_csv("Minimum_confidence_final.csv",header=None,index=None)

