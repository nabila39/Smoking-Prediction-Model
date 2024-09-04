import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from clean import dataFrame
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from C45 import C45Classifier
import numpy as np
# Show the distribution of the class label (Smoker) and indicate any
plt.figure(figsize=(8, 6))
sns.countplot(x='Smoker', data=dataFrame, palette={'1': 'pink', '0': 'gray'})
plt.title('Distribution of Smoker Class Label')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Smoker', 'Non Smoker'])
plt.show()
#. Show the density plot for the age.
plt.figure(figsize=(8, 4))
sns.kdeplot(dataFrame['Age'], shade=True, color='gray')
plt.title('Density Plot for Age')
plt.show()
#Show the density plot for the BMI.
plt.figure(figsize=(8, 4))
sns.kdeplot(dataFrame['BMI'], shade=True,color='pink')
plt.title('Density Plot for BMI')
plt.show()
#. Visualise the scatterplot of data and split based on Region attribute.
#plt.figure(figsize=(10, 8))
#correlation_matrix = dataFrame.corr()
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
#plt.title('Correlation Between Features')

sns.set_palette(['pink', 'gray']) #n 0 , s 1
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Age', y='Insurance Charges', hue='Region', data=dataFrame)
plt.title('Scatterplot of Data Split by Region')
plt.xlabel('Age')
plt.ylabel('Insurance Charges')
plt.show()

#Split the	dataset	into training (80%) and	test (20%)
features = dataFrame.drop('Smoker', axis=1)
labels = dataFrame['Smoker']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
#
#scaler = StandardScaler()
#imputer = SimpleImputer(strategy='mean')
#pipeline = Pipeline([('imputer', imputer), ('scaler', scaler)])
#X_train_scaled = pipeline.fit_transform(X_train)
#X_test_scaled = pipeline.transform(X_test)

# KNN Models
k_values = [3, 5, 13]
for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)

    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_knn)
    roc_auc_knn = auc(fpr, tpr)
    conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

    sensitivity_knn = recall_score(y_test, y_pred_knn)
    tn, fp, fn, tp = conf_matrix_knn.ravel()
    specificity_knn = tn / (tn + fp)
    f1_score_knn = f1_score(y_test, y_pred_knn)

    print(f"KNN (K={k}):")
    print(f"  Accuracy: {accuracy_knn}")
    print(f"  ROC AUC: {roc_auc_knn}")
    print("  Confusion Matrix:")
    print(conf_matrix_knn)
    print(f"  Sensitivity: {sensitivity_knn}")
    print(f"  Specificity: {specificity_knn}")
    print(f"  F1 Score: {f1_score_knn}")
    print("**********************************************")

# Decision Trees (C4.5)

dt_model = DecisionTreeClassifier(criterion='entropy',random_state=np.random.seed(86))
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, y_pred_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

sensitivity_dt = recall_score(y_test, y_pred_dt)
tn, fp, fn, tp = conf_matrix_dt.ravel()
specificity_dt = tn / (tn + fp)
f1_score_dt = f1_score(y_test, y_pred_dt)

print("Decision Trees:")
print(f"  Accuracy: {accuracy_dt}")
print(f"  ROC AUC: {roc_auc_dt}")
print("  Confusion Matrix:")
print(conf_matrix_dt)
print(f"  Sensitivity: {sensitivity_dt}")
print(f"  Specificity: {specificity_dt}")
print(f"  F1 Score: {f1_score_dt}")
print("**********************************************")

# Naive Bayes (NB)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_test, y_pred_nb)
roc_auc_nb = auc(fpr_nb, tpr_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)

sensitivity_nb = recall_score(y_test, y_pred_nb)
tn, fp, fn, tp = conf_matrix_nb.ravel()
specificity_nb = tn / (tn + fp)
f1_score_nb = f1_score(y_test, y_pred_nb)

print("Naive Bayes (NB):")
print(f"  Accuracy: {accuracy_nb}")
print(f"  ROC AUC: {roc_auc_nb}")
print("  Confusion Matrix:")
print(conf_matrix_nb)
print(f"  Sensitivity: {sensitivity_nb}")
print(f"  Specificity: {specificity_nb}")
print(f"  F1 Score: {f1_score_nb}")
print("**********************************************")

# ANN with a single hidden layer
ann_model = MLPClassifier(hidden_layer_sizes=(200,), max_iter=500, activation='logistic', random_state=42)
ann_model.fit(X_train, y_train)
y_pred_ann = ann_model.predict(X_test)

accuracy_ann = accuracy_score(y_test, y_pred_ann)
fpr_ann, tpr_ann, thresholds_ann = roc_curve(y_test, y_pred_ann)
roc_auc_ann = auc(fpr_ann, tpr_ann)
conf_matrix_ann = confusion_matrix(y_test, y_pred_ann)

sensitivity_ann = recall_score(y_test, y_pred_ann)
tn, fp, fn, tp = conf_matrix_ann.ravel()
specificity_ann = tn / (tn + fp)
f1_score_ann = f1_score(y_test, y_pred_ann)
print("Artificial Neural Network (ANN):")
print(f"  Accuracy: {accuracy_ann}")
print(f"  ROC AUC: {roc_auc_ann}")
print("  Confusion Matrix:")
print(conf_matrix_ann)
print(f"  Sensitivity: {sensitivity_ann}")
print(f"  Specificity: {specificity_ann}")
print(f"  F1 Score: {f1_score_ann}")
print("**********************************************")