from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import csv

#parameter
trainFile='D:\\2.Cao hoc\\Ky 2\\Hoc may\\1-training-data.csv'
testFile='D:\\2.Cao hoc\\Ky 2\\Hoc may\\Nguyen-Duy-Cuong-test.csv'

lcavol=[]
lweight=[]
age=[]
lbph=[]
svi=[]
lcp=[]
gleason=[]
pgg45=[]
lpsa=[]
with open(trainFile, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=str(','))
    for row in spamreader:
        lcavol.append(row[0])
        lweight.append(row[1])
        age.append(row[2])
        lbph.append(row[3])
        svi.append(row[4])
        lcp.append(row[5])
        gleason.append(row[6])
        pgg45.append(row[7])
        lpsa.append(row[8])

del lcavol[0]
del lweight[0]
del age[0]
del lbph[0]
del svi[0]
del lcp[0]
del gleason[0]
del pgg45[0]
del lpsa[0]

one = np.ones((np.array(lcavol).shape[0], 1))
Xbar = np.concatenate((one,np.array([lcavol]).T,np.array([lweight]).T,np.array([age]).T,np.array([lbph]).T,np.array([svi]).T,np.array([lcp]).T,np.array([gleason]).T,np.array([pgg45]).T,), axis = 1)

#print( u'Predict weight of person with height 155 cm: %.2f (kg), real number: 52 (kg)'  %(y1) )
#print( u'Predict weight of person with height 160 cm: %.2f (kg), real number: 56 (kg)'  %(y2) )

reg = linear_model.Ridge (alpha = 0.4)
reg.fit (Xbar, lpsa) 

lcavol=[]
lweight=[]
age=[]
lbph=[]
svi=[]
lcp=[]
gleason=[]
pgg45=[]
lpsa=[]
with open(testFile, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=str(','))
    for row in spamreader:
        lcavol.append(row[0])
        lweight.append(row[1])
        age.append(row[2])
        lbph.append(row[3])
        svi.append(row[4])
        lcp.append(row[5])
        gleason.append(row[6])
        pgg45.append(row[7])
        lpsa.append(row[8])

for i in range(0,5,1):
    del lcavol[0]
    del lweight[0]
    del age[0]
    del lbph[0]
    del svi[0]
    del lcp[0]
    del gleason[0]
    del pgg45[0]
    del lpsa[0]

one = np.ones((np.array(lcavol).shape[0], 1))
Xbar = np.concatenate((one,np.array([lcavol]).T,np.array([lweight]).T,np.array([age]).T,np.array([lbph]).T,np.array([svi]).T,np.array([lcp]).T,np.array([gleason]).T,np.array([pgg45]).T,), axis = 1)

Xbar = Xbar.astype(float) 
y_plot = reg.predict(Xbar)

# Compare two results
print( 'Solution found by scikit-learn  : ', reg.coef_ )