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

path=r'/home/leo/Desktop/traces--2MHz-sampling/powertrace-seed-3048297489193367074'

#LBKeogh Distance function
def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):
        
        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        
        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    
    return sqrt(LB_sum)
    

from cesium import featurize

features_to_use = ['amplitude',
                   'maximum',
                   'max_slope',
                   'median',
                   'median_absolute_deviation',
                   'percent_close_to_median',
                   'minimum',
                   'skew',
                   'std']

## ENTRENAR CLASIFICADOR(ES)
#Leer cada directorio para entrenar cada clase
n_class=0
class_mean=[]
dataset_list=[]
class_names=[]
n_features=len(features_to_use)
for root, dirs, files in os.walk(path):
    #print dirs
    if root==path:
        class_names.append(dirs)
        #class_names=sorted(class_names[0][:])
        continue
    print root #Label
    #assign LABEL to Class
    
    n=10000
    rows=len(files)
    class_samples=np.zeros
((rows,n))
    class_dataset=np.zeros((rows,n_features))
    i=0
    files=sorted(files)
    for name in files:
        print 'featurizing', name
        # Check and adjust sample LENGTH
        trace=pd.read_csv(root+'/'+name,sep='None',engine='python')
        samp=trace.as_matrix()
        samp=samp.transpose()
        L=int(samp.shape[1])
        #print 'length:',L

        # Cesium featurization
        t1=range(samp.shape[1])
        fset_cesium = featurize.featurize_time_series(times=t1,
                                              values=samp,
                                              errors=None,
                                              features_to_use=features_to_use)
        #print(fset_cesium)
	feat_row=fset_cesium.values
	#print(feat_row)

	class_dataset[i][:]=feat_row
	

        

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
        
    # TRAIN with all samples in matrix for this class
    print 'n:', n
    class_mean.append(class_samples.mean(axis=0))
    print 'finished featurizing class:', class_names[0][n_class]
    print 'class_dataset shape:' , class_dataset.shape
    #ADD LABEL
    labels=np.full((rows,1),n_class)
    class_dataset=np.hstack((class_dataset,labels))
    dataset_list.append(class_dataset)
    n_class=n_class+1
    #raw_input("Press Enter to continue...")

# I GOT THE LIST, BUILD THE MATRIX
# np.concatenate(list[0],list[1],list[2]...)
final_dataset=np.vstack(dataset_list)
print 'final dataset shape:', final_dataset.shape
print 'Class names:', class_names
raw_input("Press Enter to continue...")

#DO SVM, NN and print results
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

X_train = final_dataset[:, :n_features]  
y_train = final_dataset[:, n_features]


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
print 'Training SVM Classifier w/ Linear Kernel...'
svc = SVC(kernel='linear', C=C).fit(X_train, y_train)
print 'Finished training.'
print 'Training SVM Classifier with RBF Kernel...'
rbf_svc = SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
print 'Finished training.'
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
print 'Training K-Neighbors classifier...'
n_neighbors=11
knn = KNeighborsClassifier(n_neighbors, weights='uniform').fit(X_train,y_train)
print 'Finished training'


#Featurize complete powertrace as test set
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
X_test=[]
y_test=[]

while i<len(full):
    if i>1:
        if mrkr[i]!=mrkr[i-1]: # If marker[i] ~= marker[i-1] make it count
            #flag=~(flag)
            count=count+1
            print 'Adding trace to test dataset'
            print 'counter:',count
            print 'index:', i
            
    if count>=8 and mrkr[i]!=mrkr[i-1]:
        #Take signal until the length of every class and calculate distance:
        distances=[]
        distances_bylength=[]
        n_classes=len(class_names[0])
        trace_features=np.zeros((n_classes,n_features))

        #Take features from TRUE signal length (TRICK)
        true_idx=class_names[0].index(actual[count-8][0])
        L=len(class_mean[true_idx])
        t1=range(L)
        fset_cesium = featurize.featurize_time_series(times=t1,
                                              values=full[i:i+L].T,
                                              errors=None,
                                              features_to_use=features_to_use)
            
	feat_row=fset_cesium.values
        X_test.append(feat_row)
        y_test.append(true_idx)

        """ Take features from ALL 15 class lengths
        for n in range(n_classes):
            L=len(class_mean[n])
            print 'Class number:',n #,'class length:',L
            
            #mean_vect=class_mean[n][:,np.newaxis]
            #class_dist=LB_Keogh(full[i:i+L],mean_vect,100)

            #X_test=featurize(trace) x 15 (all possible lengths)
            # Cesium featurization
            t1=range(L)
            fset_cesium = featurize.featurize_time_series(times=t1,
                                              values=full[i:i+L].T,
                                              errors=None,
                                              features_to_use=features_to_use)
            
	    feat_row=fset_cesium.values
	    

	    trace_features[n][:]=feat_row
            

            #print 'DTW distance:',class_dist # use DTW for distance scores
	    #norm_dist=class_dist/L
            #print 'DTW Dist / length:', norm_dist
            #distances.append(class_dist)
            #distances_bylength.append(norm_dist)
        

        #Make classification:

        Y_test_linear=svc.predict(feat_row)
        Y_test_rbf=rbf_svc.predict(feat_row)
        Y_test_knn=knn.predict(feat_row)

        print 'Prediction by SVC (linear Kernel):'
        print Y_test_linear
        print 'Prediction by SVC (RBF Kernel):'
        print Y_test_rbf
        print 'Prediction by kNN with %d neighbors:' %n_neighbors
        print Y_test_knn
        print 'Ground truth:', actual[count-8][0], 'Index:', true_idx


        #Take the most voted of the 15:
        #result=np.bincount(int(Y_test_linear)).argmax()

        
        
        
        print 'class index:', result
        print 'class label:', class_names[0][result]
        if class_names[0][result]==actual[count-8][0]:
            print bcolors.OKGREEN + 'Correct!' + bcolors.ENDC
            right += 1
        else:
            print bcolors.WARNING + 'Wrong.' + bcolors.ENDC
            wrong += 1

        
        raw_input("Press Enter to continue...")
        """
        
            
    
    if count>1000:
        #precision = 100*right/(right+wrong)
        #print 'Clasification precision:', precision, '%'
        break
        
    i+=1



X_test=np.vstack(X_test)
y_test=np.vstack(y_test)
y_test=np.squeeze(y_test)


#DO COMPARISON
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = 1  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

figure = plt.figure(figsize=(27, 9))

i=1

X=np.vstack((X_train,X_test))
y=np.concatenate((y_train,y_test))

X = StandardScaler().fit_transform(X)
#DO PCA and take PC1 and PC2 as features (for plotting purposes)
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(X)
X_pc=pca.transform(X)
X_pc=X_pc[:100][:]
y_plot=np.random.randint(2, size=100)

X_train_pc, X_test_pc, y_train_plot, y_test_plot = \
        train_test_split(X_pc, y_plot, test_size=.25, random_state=42)

x_min, x_max = X_pc[:, 0].min() - .5, X_pc[:, 0].max() + .5
y_min, y_max = X_pc[:, 1].min() - .5, X_pc[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(1, len(classifiers) + 1, i)
#if ds_cnt == 0:
ax.set_title("Input data")
# Plot the training points
ax.scatter(X_train_pc[:, 0], X_train_pc[:, 1], c=y_train_plot, cmap=cm_bright)
# and testing points
ax.scatter(X_test_pc[:, 0], X_test_pc[:, 1], c=y_test_plot, cmap=cm_bright, alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
i += 1



# iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(1, len(classifiers) + 1, i)
    print 'Training classifier:', name
    clf.fit(X_train, y_train)
    print 'Finished training.'
    score = clf.score(X_test, y_test)
    print 'Classifier Score:', score
    print 'Training with first 2 PCs...'
    clf.fit(X_train_pc, y_train_plot)
    print 'Finished training.'

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 0]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(X_train_pc[:, 0], X_train_pc[:, 1], c=y_train_plot, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test_pc[:, 0], X_test_pc[:, 1], c=y_test_plot, cmap=cm_bright,
               alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    i += 1

plt.tight_layout()
plt.show()



