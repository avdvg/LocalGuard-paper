import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def classifier(aaa, label):

    y_true = label
    x_train, x_test, y_train, y_test = train_test_split(aaa, y_true, train_size=0.8, shuffle=False)

    clf0 = MLPClassifier(random_state=10, max_iter=500)
    clf0.fit(x_train, y_train)
    p0 = clf0.score(x_test, y_test)
    p0 = round(p0, 3)
    print('mlp finished')

    clf1 = GaussianNB()
    clf1.fit(x_train, y_train)
    p1 = clf1.score(x_test, y_test)
    p1 = round(p1, 3)
    print('gnb')

    clf2 = svm.SVC(random_state=10, max_iter=200)
    clf2.fit(x_train, y_train)
    p2 = clf2.score(x_test, y_test)
    p2 = round(p2, 3)
    print('svm')

    clf3 = GradientBoostingClassifier(random_state=10)
    clf3.fit(x_train, y_train)
    p3 = clf3.score(x_test, y_test)
    p3 = round(p3, 3)
    print('gb')

    print('mlp',p0, 'lr',p1,'svm',p2,'gb',p3)
