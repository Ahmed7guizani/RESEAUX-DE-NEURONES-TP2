# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:38:02 2021

@author: User
"""
# question 1 a
import numpy as np
data = np.loadtxt('C:/Users/User/OneDrive/桌面/TP_RN2/dataset.dat')
#print(data)

X = data[:,0:2]
y = data[:,2]
y = y.astype(int)

#visualiser les données
from matplotlib import pyplot
colors= np.array([x for x in "rgbcmyk"])
pyplot.scatter(X[:, 0], X[:, 1], color=colors[y].tolist(), s=10)
pyplot.show()

#partition des données
from sklearn import model_selection
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.3, random_state=42)

print('la dimension = ',X.ndim)


print("Le nombre d'exemple d'apprentissage = " ,y_train.size)
print("Le nombre d'exemple test = " ,y_test.size)


# b c 
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings("ignore")

def mono_couche(A, B):
    clf1=SGDClassifier(loss='perceptron', eta0=A, max_iter=B, learning_rate='constant', verbose=1)
    clf1.fit(X_train,y_train)

    print("le taux de reconnaissance sur les bases d’apprentissage : ", clf1.score(X_train, y_train))
    print("le taux de reconnaissance sur les bases de test : ", clf1.score(X_test, y_test))
    print("\n\n")
    

    x_min, x_max = X[:, 0].min()*1.1, X[:, 0].max()*1.1 
    y_min, y_max = X[:, 1].min()*1.1, X[:, 1].max()*1.1 
    x_h = (x_max - x_min)/50
    y_h = (y_max - y_min)/50
    xx, yy = np.meshgrid(np.arange(x_min, x_max, x_h), 
                         np.arange(y_min, y_max, y_h))
    Y = clf1.predict(np.c_[xx.ravel(), yy.ravel()]) 
    Y = Y.reshape(xx.shape)

    # les frontières de décisions et les données de test

    pyplot.contourf(xx, yy, Y, cmap=pyplot.cm.Paired, alpha=0.8)
    pyplot.scatter(X_test[:, 0], 
                   X_test[:, 1], 
                   cmap=pyplot.cm.Paired, 
                   color=colors[y_test].tolist()
                  ) 
    pyplot.xlim(xx.min(), xx.max())
    pyplot.ylim(yy.min(), yy.max())
    pyplot.show()
    
    
A = 0.000001
B = 1
mono_couche(A, B)
A = 1
B = 1
mono_couche(A, B)
A = 0.000001
B = 10
mono_couche(A, B)
A = 1
B = 10
mono_couche(A, B)

# question 2 a b
from sklearn.neural_network import MLPClassifier
import numpy as np

C = 5

clf2 = MLPClassifier(hidden_layer_sizes=C, activation='tanh', solver='sgd', 
                     batch_size=1, alpha=0, learning_rate='constant', max_iter=100, momentum=0, verbose=True)
clf2.fit(X_train, y_train)

print("le nombre d’itérations effectuées : ", clf2.n_iter_)

clf2 = MLPClassifier(hidden_layer_sizes=C, activation='tanh', solver='sgd', 
                     batch_size=1, alpha=0, learning_rate='constant', learning_rate_init = 0.01, max_iter=100, momentum=0, verbose=True)
clf2.fit(X_train, y_train)

print("le nombre d’itérations effectuées : ", clf2.n_iter_)

clf2 = MLPClassifier(hidden_layer_sizes=C, activation='tanh', solver='adam', 
                     batch_size=1, alpha=0, learning_rate='constant', learning_rate_init = 0.01,max_iter=100, momentum=0, verbose=True)
clf2.fit(X_train, y_train)

print("le nombre d’itérations effectuées : ", clf2.n_iter_)


# c

def multi_couche(C):
    taux_reconnaissance_apprentissage = []
    taux_reconnaissance_test = []
    for i in range(1,11):
        clf2 = MLPClassifier(hidden_layer_sizes=C, activation='tanh', solver='adam', 
        batch_size=1, alpha=0, learning_rate='constant', learning_rate_init = 0.01,max_iter=100, momentum=0)
        clf2.fit(X_train, y_train)
        taux_reconnaissance_apprentissage.append(clf2.score(X_train, y_train))
        taux_reconnaissance_test.append(clf2.score(X_test, y_test))
    print("C = ", C)
    print("le taux de reconnaissance sur les bases d’apprentissage : ", taux_reconnaissance_apprentissage)
    print("le taux de reconnaissance sur les bases de test : ", taux_reconnaissance_test)
    print("\n")
    print("la moyenne de taux en apprentissage : ", np.mean(taux_reconnaissance_apprentissage))
    print("l'écrat type de taux en apprentissage", np.std(taux_reconnaissance_apprentissage,ddof=1))
    print("\n")
    print("la moyenne de taux en test : ", np.mean(taux_reconnaissance_test))
    print("l'écrat type de taux en test", np.std(taux_reconnaissance_test,ddof=1))
    print("\n")
    
multi_couche(1)
multi_couche(2)
multi_couche(5)
multi_couche(10)
multi_couche(20)
multi_couche(50)
multi_couche(100)




