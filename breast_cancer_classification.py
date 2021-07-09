import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

breast_cancer = load_breast_cancer()

features = breast_cancer["data"]
target = breast_cancer["target"]

df = pd.DataFrame(features, columns=breast_cancer.feature_names)
df["target"] = target

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.25, random_state = 123)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print(knn.predict_proba(X_test))

print("_" * 70)
print("KNeighbors Classifier Accuracy : \n{}".format(accuracy_score(y_pred_knn, y_test)))
print("KNeighbors Classifier Confusion Matrix : \n{}".format(confusion_matrix(y_pred_knn, y_test)))
print("KNeighbors Classifier Classification Report : \n{}".format(classification_report(y_pred_knn, y_test)))
print("KNeighbors Classifier roc_auc_score : \n{}".format(roc_auc_score(y_pred_knn, y_test)))
print("KNeighbors Classifier roc_curve : \n{}".format(roc_curve
(y_pred_knn, y_test)))

# plot roc curve
y_pred_proba = knn.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.savefig("breast_cancer_knn.png")


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print("_" * 70)
print("Logistic Regression score: \n{}".format(accuracy_score(y_pred_logreg, y_test)))
print("Logistic Regression Confusion Matrix: \n{}".format(confusion_matrix(y_pred_logreg, y_test)))
print("Logistic Regression Classification Report:\n{}".format(classification_report(y_pred_logreg, y_test)))


# plot roc curve
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.savefig("breast_cancer_logreg.png")


tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
print("_" * 70)
print("Decision Tree Accuracy : \n{}".format(accuracy_score(y_pred_tree, y_test)))
print("Decision Tree Confusion Matrix : \n{}".format(confusion_matrix(y_pred_tree, y_test)))
print("Decision Tree Classification Report : \n{}".format(classification_report(y_pred_tree, y_test)))


#plot roc curve
y_pred_proba = tree.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.savefig("breast_cancer_tree.png")


rand = RandomForestClassifier(n_estimators=500)
rand.fit(X_train, y_train)
y_pred_rand = rand.predict(X_test)
print("_" * 70)
print("Random Forest Accuracy : \n{}".format(accuracy_score(y_pred_rand, y_test)))
print("Random Forest Confusion Matrix : \n{}".format(confusion_matrix(y_pred_rand, y_test)))
print("Random Forest Classification Report : \n{}".format(classification_report(y_pred_rand, y_test)))

#plot roc curve
y_pred_proba = rand.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
print(fpr, tpr, _)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.savefig("breast_cancer_rand.png")



ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
print("_" * 70)
print("AdaBoost Classifier Accuracy : \n{}".format(accuracy_score(y_pred_ada, y_test)))
print("AdaBoost Classifier Confusion Matrix : \n{}".format(confusion_matrix(y_pred_ada, y_test)))
print("AdaBoost Classifier Classification Report : \n{}".format(classification_report(y_pred_ada, y_test)))

# plot roc curve
y_pred_proba = ada.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.savefig("breast_cancer_ada.png")

grad = GradientBoostingClassifier()
grad.fit(X_train, y_train)
y_pred_grad = grad.predict(X_test)

print("_" * 70)
print("Gradient Boosting Accuracy : \n{}".format(accuracy_score(y_pred_grad, y_test)))
print("Gradient Boosting Confusion Matrix : \n{}".format(confusion_matrix(y_pred_grad, y_test)))
print("Gradient Boosting Classification Report : \n{}".format(classification_report(y_pred_grad, y_test)))

# plot roc curve
y_pred_proba = grad.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.savefig("breast_cancer_grad.png")

svc = SVC()
svc.fit(X_train, y_train)
y_pred_svc = grad.predict(X_test)
print("_" * 70)
print("SVC Accuracy : \n{}".format(accuracy_score(y_pred_svc, y_test)))
print("SVC Confusion Matrix : \n{}".format(confusion_matrix(y_pred_svc, y_test)))
print("SVC Classification Report : \n{}".format(classification_report(y_pred_svc, y_test)))


lin_svc = LinearSVC()
lin_svc.fit(X_train, y_train)
y_pred_linsvc = grad.predict(X_test)
print("_" * 70)
print("Linear SVC Accuracy : \n{}".format(accuracy_score(y_pred_linsvc, y_test)))
print("Linear SVC Confusion Matrix : \n{}".format(confusion_matrix(y_pred_linsvc, y_test)))
print("Linear SVC Classification Report : \n{}".format(classification_report(y_pred_linsvc, y_test)))





