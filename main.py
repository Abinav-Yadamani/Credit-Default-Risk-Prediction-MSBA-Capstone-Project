# importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Setting the warnings to be ignored
warnings.filterwarnings('ignore')


# reading the application_train files and displaying the shape

application_train = pd.read_csv("application_train.csv")


application_train['CREDIT_TERM'] = application_train['AMT_ANNUITY'] / application_train['AMT_CREDIT']

application_train['DAYS_EMPLOYED_PERCENT'] = application_train['DAYS_EMPLOYED'] / application_train['DAYS_BIRTH']

train = application_train[['TARGET','EXT_SOURCE_3','EXT_SOURCE_2', 'EXT_SOURCE_1', 'AMT_GOODS_PRICE', 'CREDIT_TERM','DAYS_EMPLOYED_PERCENT' ]]
#print(train.head())


# # missing values
# missing_value = (application_train.isnull().mean() * 100).round()

# # Create a new DataFrame to display the results
# missing_info = pd.DataFrame({'Column Name': missing_value.index, 'Missing Percentage': missing_value.values})

# cor = pd.DataFrame(application_train.corr())
# cor = cor.sort_values('TARGET',ascending=False)
# # extracting the column that shows only the correlation with the target variable
# cor = cor.iloc[1:,1]

# # Displaying column missing value % along with its correlation with the target variable
# cor = pd.DataFrame(cor)
# missing_cor_df = pd.merge(missing_info, cor, left_on='Column Name', right_on = cor.index)
# missing_cor_df[missing_cor_df['Missing Percentage']>0].sort_values('Missing Percentage',ascending=False).head(4)


# # Extracting columns with missing value percent greater than 48
# missing_cor_df = missing_cor_df[missing_cor_df['Missing Percentage'] >= 48]
# missing_cor_columns = missing_cor_df[missing_cor_df['Column Name'] != 'EXT_SOURCE_1']['Column Name']
# # dropping columns with missing value percent greater than 48
# application_train_clean = application_train.drop(columns=missing_cor_columns, axis=1)
# application_train_clean.shape


# for column in application_train_clean.columns:
#     if application_train_clean[column].dtype == 'object':
#         application_train_clean[column].fillna(application_train_clean[column].mode()[0], inplace=True)

# for column in application_train_clean.columns:
#     if application_train_clean[column].dtype == 'float':
#         application_train_clean[column].fillna(application_train_clean[column].median(), inplace=True)


# selected_columns = []

# # displaying columns along with a proportion of the unique values
# for column in application_train_clean.columns:
#     unique_values = application_train_clean[column].value_counts()
#     total_count = len(application_train_clean)
#     if len(unique_values) < 3: # Unique values less than 3 in a column
#         proportions = unique_values / total_count
#         if any(proportions > 0.98):  # Check if any proportion is above 0.95
#             column_info = {
#                 "Column": column,
#                 "Proportions": proportions
#             }
#             selected_columns.append(column_info['Column'])

# # Dropping the selected columns with less variability
# application_train_clean = application_train_clean.drop(columns=selected_columns, axis=1)
# application_train_clean.shape


train = pd.get_dummies(train)

from sklearn.model_selection import train_test_split

X = train.loc[:, train.columns != 'TARGET']
y = train['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print('X_train shape:', X_train.shape)
#print('y_train shape:', y_train.shape)
#print('X_test shape:', X_test.shape)
#print('y_test shape:', y_test.shape)


##########################################################################
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# XGBoost classifier Initializing with the specified parameters
xgb_model = XGBClassifier(colsample_bytree = 0.9,
                          learning_rate = 0.1,
                          max_depth = 5,
                          n_estimators = 200,
                          subsample = 1)

# Fit the model to the training data
xgb_model.fit(X_train, y_train)

y_train_pred_prob = xgb_model.predict_proba(X_train)[:, 1]

# Make predictions on the test set
xgb_y_pred = xgb_model.predict_proba(X_test)[:, 1]  # Use predict_proba to get probability scores for the positive class
y_pred = xgb_model.predict(X_test)

xgb_train_roc_auc = roc_auc_score(y_train, y_train_pred_prob)

# Calculate the ROC-AUC score on the test set
xgb_roc_auc = roc_auc_score(y_test, xgb_y_pred)
xgb_accuracy = accuracy_score(y_test, y_pred)

# pickle the model
# import pickle
# with open('model.pkl', 'wb') as file:
#     pickle.dump(xgb_model, file)

# Save the trained model to a file
xgb_model.save_model('xgboost_model.json')




print(f'Tuned XGB Train ROC-AUC Score: {xgb_train_roc_auc}')
print(f'Tuned XGB Test ROC-AUC Score: {xgb_roc_auc}')
print(f'Tuned XGB Accuracy Score: {xgb_accuracy}')