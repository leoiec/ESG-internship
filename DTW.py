
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from numpy.fft import fft
from numpy.fft import ifft

import os


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

path=r'/home/leo/Desktop/traces--2MHz-sampling/powertrace-seed-3048297489193367074'



#Dynamic Time Warping function
def DTWDistance(s1, s2,w):
    DTW={}
    
    w = max(w, abs(len(s1)-len(s2)))
    
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
  
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
		
    return sqrt(DTW[len(s1)-1, len(s2)-1])



#LBKeogh Distance function
def LB_Keogh(s1,s2,r):
    LB_sum=0
    m=0
    for ind,i in enumerate(s1):
        
        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        
        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
            m+=1
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
            m+=1
    
    return (sqrt(LB_sum),sqrt(m))




## ENTRENAR CLASIFICADOR(ES)
#Leer cada directorio para entrenar cada clase
n_class=0
class_mean=[]
class_names=[]
for root, dirs, files in os.walk(path):
    #print dirs
    if root==path:
        class_names.append(dirs)
        continue
    print root #Label
    #assign LABEL to Class
    
    n=10000
    rows=len(files)
    class_samples=np.zeros((rows,n))
    i=0
    for name in files:
        print 'adding', name
        # Check and adjust sample LENGTH
        trace=pd.read_csv(root+'/'+name,sep='None',engine='python')
        samp=trace.as_matrix()
        samp=samp.transpose()
        L=int(samp.shape[1])
        print 'length:',L
        # estandarizar el largo de la senal para cada clase (o sub-clases)
        class_row=np.zeros(n)
        if L>=n:
            class_row=np.resize(samp,(1,n))
        else:
            class_samples=np.resize(class_samples,(rows,L))
            class_row=samp
            n=L
        class_samples[i][:]=class_row
        i=i+1
        print 'n:', n
    # TRAIN with all samples in matrix for this class
    class_mean.append(class_samples.mean(axis=0))
    n_class=n_class+1
    
    
    
    
        
## Probar clasificador(es)

#Leer traza completa y marker
#La idea es: Probar la traza para cada largo e ir midiendo la distancia contra
#todas las classes
Loc = r'/home/leo/Desktop/traces--2MHz-sampling/powertrace-seed-3048297489193367074/powertrace-complete.txt'
print 'Reading full power trace...'
fulltrace=pd.read_csv(Loc,sep='None',engine='python')
full=fulltrace.as_matrix()
N=len(fulltrace)
print 'Finished reading full trace. Length:', N

print 'Reading ground truth traces...'
actualtrace=pd.read_csv(path+'/actual_trace.txt',header=None)
actual=actualtrace.values.tolist()
print 'Finished reading ground truth traces, # of traces:', len(actual)


marker=pd.read_csv(path+'/marker.txt')
mrkr=marker.as_matrix()
right=0
wrong=0
flag=0
count=0
i=0

while i<len(full):
    if i>1:
        if mrkr[i]!=mrkr[i-1]: # If marker[i] ~= marker[i-1] make it count
            #flag=~(flag)
            count=count+1
            print 'counter:',count
            print 'index:', i
            
    if count>=8 and mrkr[i]!=mrkr[i-1]:
        #Take signal until the length of every class and calculate distance:
        distances=[]
        distances_bylength=[]
        for n in range(len(class_mean)):
            L=len(class_mean[n])
            print 'Class number:',n #,'class length:',L
            mean_vect=class_mean[n][:,np.newaxis]
            class_dist,m=LB_Keogh(full[i:i+L],mean_vect,100)
            #print 'DTW distance:',class_dist # use DTW for distance scores
	    norm_dist=class_dist/L
            norm_dist2=class_dist/m
            rtio=L/m
            print 'DTW Dist/length: %.3f' % norm_dist
            #print 'DTW Dist/m: %.3f' % norm_dist2
            #print 'L=%d,m=%d,ratio=%.2f' % (L, m, rtio)
            distances.append(class_dist)
            distances_bylength.append(norm_dist)
        #Make classification and determine new starting point:
        min_dist=min(distances)
        min_dist_bylength=min(distances_bylength)
        result=distances_bylength.index(min_dist_bylength)
        #print 'minimum DTW Distance:',min_dist
        print 'minimum DTW/Length: %.3f' % min_dist_bylength
        print 'class index:', result
        print 'class label:', class_names[0][result]
        print 'Ground truth:', actual[count-8][0], 'Index:', class_names[0].index(actual[count-8][0])
        if class_names[0][result]==actual[count-8][0]:
            print bcolors.OKGREEN + 'Correct!' + bcolors.ENDC
            right += 1
        else:
            print bcolors.WARNING + 'Wrong.' + bcolors.ENDC
            wrong += 1
        #raw_input("Press Enter to continue...")
        #i=i+len(class_mean[result])+200
        
            
    
    if count>50:
        precision = 100*right/(right+wrong)
        print 'Clasification precision:', precision, '%'
        break
            
    i+=1
        

        #When marker switches again, make classification.


        
    
                            
                            
