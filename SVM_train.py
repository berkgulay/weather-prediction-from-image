#author Berk Gulay

from sklearn import svm

train


#try gamma = 3 or 2 and 0.2 or 0.1 , kernel= polynomial with degree 3,5,7,10 coef0 as 0,10,100
#try C as 4 and 2 and 0.5
#try divided tol by 1/10
#try decision_function_shape as 'ovo'
#try class_weight as default
#try with no shrink (default with shrink change it)
svc = svm.SVC(kernel='rbf',gamma=0.5,C=1.2,class_weight='balanced',max_iter=500,decision_function_shape='ovr',tol=0.001,cache_size=1000,probability=True)

svc.fit()

predictions = svc.predict()
