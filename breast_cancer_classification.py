import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def print_accuracy_result(name, y_pred):
    print(name, "score: \n{}".format(accuracy_score(y_pred, y_test)))
    print(name, "Confusion Matrix: \n{}".format(confusion_matrix(y_pred, y_test)))
    print(name, "Classification Report:\n{}".format(classification_report(y_pred, y_test)))


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print_accuracy_result("KNeighbors Classifier", y_pred_knn)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print_accuracy_result("Logistic Regression",  y_pred_logreg)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
print_accuracy_result("Decision Tree", y_pred_tree)

rand = RandomForestClassifier(n_estimators=500)
rand.fit(X_train, y_train)
y_pred_rand = rand.predict(X_test)
print_accuracy_result("Random Forest", y_pred_rand)

ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
print_accuracy_result("AdaBoost Classifier", y_pred_ada)

grad = GradientBoostingClassifier()
grad.fit(X_train, y_train)
y_pred_grad = grad.predict(X_test)
print_accuracy_result("Gradient Boosting", y_pred_grad)

svc = SVC()
svc.fit(X_train, y_train)
y_pred_svc = grad.predict(X_test)
print_accuracy_result("SVC", y_pred_svc)

lin_svc = LinearSVC()
lin_svc.fit(X_train, y_train)
y_pred_linsvc = grad.predict(X_test)
print_accuracy_result("Linear SVC", y_pred_linsvc)

mods = [knn, logreg, tree, rand, ada, grad, svc, lin_svc]
mod_names = ['knn', 'logreg', 'tree', 'rand', 'ada', 'grad', 'svc', 'lin_svc']
y_preds = [y_pred_knn, y_pred_logreg, y_pred_tree, y_pred_rand, y_pred_ada, y_pred_grad, y_pred_svc, y_pred_linsvc]

#plot cf_matrix
def cf_matrix(model_name, y_pred):
    plt.figure()
    cf_m = confusion_matrix(y_pred, y_test)
    sns.heatmap(cf_m, annot=True)
    plt.xlabel("predicted_label")
    plt.ylabel("True_label")
    plt.savefig("breast_cancer_cf_{}.png".format(model_name))
    

# plot roc curve
def auc_roc(model, model_name):
    plt.figure()
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    ns_probs = [0 for _ in range(len(y_test))]
    ns_auc = roc_auc_score(y_test, ns_probs)
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(fpr,tpr, linestyle='dotted', label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.title("Receiver Operating Characteristic Curve(ROC)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("breast_cancer_{}.png".format(model_name))


for i in range(len(mods)):
    cf_matrix(mod_names[i],y_preds[i])

for i in range(len(mods)-2):
    auc_roc(mods[i], mod_names[i])


