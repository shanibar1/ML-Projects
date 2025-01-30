import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split

df = pd.read_csv("XY_train.csv", encoding='iso-8859-1')

# ======== MISSION 0 ========
# split input dataset to training dataset & testing dataset

# remove irrelevant columns
df_imputed = df.drop(columns=['ID', 'POSTAL_CODE'])

# convert values to numerical
df_imputed['GENDER'] = df_imputed['GENDER'].map({'male': 0, 'female': 1})
df_imputed['EDUCATION'] = df_imputed['EDUCATION'].map({'none': 0, 'high school': 1, 'university': 2})
df_imputed['VEHICLE_TYPE'] = df_imputed['VEHICLE_TYPE'].map({'sedan': 0, 'sports car': 1})
df_imputed['VEHICLE_YEAR'] = df_imputed['VEHICLE_YEAR'].map({'before 2015': 0, 'after 2015': 1})
df_imputed['INCOME'] = df_imputed['INCOME'].map({'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3})

df_imputed['AGE'] = pd.cut(df_imputed['AGE'], bins=[0, 20, 30, 50, 70, 100], labels=[0, 1, 2, 3, 4])
df_imputed['DRIVING_EXPERIENCE'] = pd.cut(df_imputed['DRIVING_EXPERIENCE'], bins=[0, 1, 3, 10, 20, 100],
                                          labels=[0, 1, 2, 3, 4])

# remove rows with missing values
df_imputed = df_imputed.dropna()

# save the imputed dataset
df_imputed.to_csv('XY_train_imputed.csv', index=False)

# define X (features) and Y (target variable for prediction)
# target variable is 'OUTCOME'

X = df_imputed.drop(columns=['OUTCOME'])
Y = df_imputed['OUTCOME']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# ======== MISSION 1 ========

# define a new DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)


def print_dtc_score(dtc, X_train, Y_train, X_test, Y_test):
    # determining the prediction accuracy can be done with
    # DecisionTreeClassifier.score or
    # sklearn.metrics.roc_auc_score

    print("=================== MODEL ACCURACY SCORES ===================")

    # ===== accuracy score =====
    y_train_pred = dtc.predict(X_train)

    # calculate accuracy
    score = accuracy_score(Y_train, y_train_pred)
    # print the accuracy of the model for the training dataset
    print(f"Model accuracy score for TRAINING dataset: {score}")
    y_test_pred = dtc.predict(X_test)
    # print the accuracy of the model for the test dataset
    score = accuracy_score(Y_test, y_test_pred)
    print(f"Model accuracy score for TEST dataset: {score}")


print_dtc_score(dtc, X_train, Y_train, X_test, Y_test)

# ======== MISSION 3 ========
# we need to do hyperparameter tuning to improve the model's performance
# we can use GridSearchCV to find the best hyperparameters

# plotting the leaves amount with help of kfold to check accuracy using AUC-ROC
Min_Sample_Leaves_Amount = np.arange(1, 1350, 50)
TreeModel = DecisionTreeClassifier(random_state=42)
SKFold_CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
ParamatersDict = {'min_samples_leaf': Min_Sample_Leaves_Amount}

# using grid search to find min leaves amount that we can starting our paraments from:
grid = GridSearchCV(estimator=TreeModel, param_grid=ParamatersDict, scoring='accuracy',
                    cv=SKFold_CV, n_jobs=-1, refit=True, return_train_score=True)
grid.fit(X_train, Y_train)
cv_results = pd.DataFrame(grid.cv_results_)
Selected_Leaves = ['mean_test_score',
                   'mean_train_score', 'param_min_samples_leaf']

DF_Selected_Values = cv_results[Selected_Leaves]
DF_Selected_Values = DF_Selected_Values.sort_values(
    'param_min_samples_leaf', ascending=True)

plt.figure(figsize=(13, 4))
plt.plot(DF_Selected_Values['param_min_samples_leaf'],
         DF_Selected_Values['mean_train_score'], marker='x', markersize=4)
plt.plot(DF_Selected_Values['param_min_samples_leaf'],
         DF_Selected_Values['mean_test_score'], marker='o', markersize=4)
plt.title('min samples leaves Amount AUC score')
plt.legend(['Train accuracy', 'Validation accuracy'])
plt.xlabel('min_samples_leaf')
plt.xticks([int(x) for x in DF_Selected_Values['param_min_samples_leaf']])
plt.ylabel('AUC score')
#plt.show()

# define the hyperparameters to tune
max_depth_list = np.arange(1, 20)
leaf_samples = np.arange(50, 300, 5)
params_dt = {
    'max_depth': max_depth_list,
    'criterion': ['entropy', 'gini'],
    'class_weight': ['balanced', None],
    'min_samples_leaf': leaf_samples,
}
# define the GridSearchCV
TreeModel = DecisionTreeClassifier(random_state=42)
SKFold_CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# use grid tree to tune the parameters
Grids_Tree = GridSearchCV(estimator=TreeModel, param_grid=params_dt, scoring='accuracy',
                          cv=SKFold_CV, n_jobs=-1, refit=True, return_train_score=True)
Grids_Tree.fit(X_train, Y_train)

best_hyperparameters = Grids_Tree.best_params_
best_TreeModel = Grids_Tree.best_estimator_

# plot the best hyperparameters using matplotlib

cv_results = pd.DataFrame(Grids_Tree.cv_results_)
Selected_Leaves = ['std_test_score', 'mean_test_score', 'mean_train_score',
                   'param_max_depth', 'param_criterion', 'param_class_weight', 'param_min_samples_leaf']
DF_Selected_Values = cv_results[Selected_Leaves]
DF_Selected_Values = DF_Selected_Values.sort_values(
    'mean_test_score', ascending=False).head(10)
DF_Selected_Values['mean_test_score'] = DF_Selected_Values['mean_test_score'].round(
    4)
DF_Selected_Values['mean_train_score'] = DF_Selected_Values['mean_train_score'].round(
    4)
DF_Selected_Values['std_test_score'] = DF_Selected_Values['std_test_score'].round(
    4)
column_names = {
    'mean_test_score': 'Mean test score',
    'mean_train_score': 'Mean train score',
    'param_max_depth': 'Max depth',
    'param_criterion': 'Criterion',
    'param_class_weight': 'Class weight',
    'param_min_samples_leaf': 'Min samples leaf',
    'std_test_score': 'std test score'
}

# plot the results
DF_Selected_Values = DF_Selected_Values.rename(columns=column_names)
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')
ax.set_title('Grid Search results', y=1.1)
table = ax.table(cellText=DF_Selected_Values.values,
                 colLabels=DF_Selected_Values.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.25, 1.25)
#plt.show()

# features importance:

importance = pd.DataFrame({'Feature_name': X_train.columns,
                           'Importance': best_TreeModel.feature_importances_.round(4)})

# sort the DataFrame by importance in descending order
importance = importance.sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(9, 10))
ax.axis('off')
table = ax.table(cellText=importance.values,
                 colLabels=importance.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(0.6, 1.5)
#plt.show()

# ======== MISSION 4 ========
best_estimator = Grids_Tree.best_estimator_
best_params = Grids_Tree.best_params_

# print the accuracy scores for the best estimator model
print_dtc_score(best_estimator, X_train, Y_train, X_test, Y_test)

# plot the tree for the best estimator model
plt.figure(figsize=(20, 16))
plot_tree(best_estimator, filled=True, max_depth=2, feature_names=X_train.columns, class_names=['0', '1'], fontsize=6)
plt.savefig('decision_tree.png')

# print the feature importance for the best estimator model (ordered by importance)
feature_importances = best_estimator.feature_importances_
feature_importances = [(feature_name, importance) for feature_name, importance in
                       zip(X_train.columns, feature_importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

print("=================== FEATURE IMPORTANCE SCORES ===================")
for feature_name, importance in feature_importances:
    print(f"{feature_name}: {importance}")
