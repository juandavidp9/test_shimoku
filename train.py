import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,  confusion_matrix


print('importing datasets')
leads = pd.read_csv(r'data\leads.csv')
offers = pd.read_csv(r'data\offers.csv')


def clean_data(leads ,offers):

    offers.drop_duplicates(inplace=True)
    leads.drop_duplicates(inplace=True)

    filtered_leads = leads[leads['Id'].notna()]
    df = pd.merge(filtered_leads, offers, how='left', on=['Id', 'Use Case'])

    #Use case
    df['Use Case'] = df['Use Case'].fillna('Others')

    use_case_mapping = {
        'Concerts and festivals': 'Others',
        'Wedding Planning': 'Others'
    }
    df['Use Case'] = df['Use Case'].replace(use_case_mapping)

    #Source
    source_distribution = df['Source'].value_counts(dropna=False, normalize=True)
    df['Source'].fillna(pd.Series(np.random.choice(source_distribution.index, p=source_distribution.values, size=len(df))), inplace=True)

    #Status
    df = df.rename(columns={'Status_x': 'Status'})
    df.loc[df['Status'] == 'Qualified', 'Status'] = df['Status_y']

    status_mapping = {
        'Called': 'Contacted',
        'Meeting': 'Contacted',
        'Recicled': 'Contacted',
        'Email': 'Contacted',
        'Call': 'Contacted',
        'Demo 1': 'Offer',
        'Demo 2': 'Offer',
        'Negotiation': 'Offer', 
        'Checkbox': 'Offer',
        'New': 'Contacted',
        'Closed Won': 'Offer',
        'Closed Lost': 'Offer'
    }

    df['Status'] = df['Status'].replace(status_mapping)
    df['Status'] = df['Status'].fillna('Other')


    #Reason 
    df = df.rename(columns={'Discarded/Nurturing Reason': 'Reason'})

    def custom_logic(row):
        # if pd.isna(row['Reason']) and row['Status_y'] != 'Closed Won':
        #     loss_reason = row['Loss Reason'] if pd.notna(row['Loss Reason']) else 'Unknown'
        #     return 'loss at end'
        # elif row['Status_y'] == 'Closed Won':
        #     return 'Success'
        if pd.isna(row['Reason']) and (row['Status_y'] == 'Closed Won' or row['Status_y'] == 'Closed Lost'):
            return 'lead converted'
        else:
            return row['Reason']
    df['Reason'] = df.apply(custom_logic, axis=1)


    reason_mapping = {
        'Unreachable': 'Unreachable',
        'No show': 'Unreachable',
        'Not the right moment': 'Not fit',
        'Not feeling': 'Not fit',
        'Not Fit': 'Not fit',
        'Competitor': 'Not Target',
        'Wrong Data': 'Not Target',
        'Black list': 'Not Target',
        'Duplicate/Test': 'Not Target',
        'Already Customer': 'Not Target'
    }
    df['Reason'] = df['Reason'].replace(reason_mapping)
    df['Reason'] = df['Reason'].fillna('Other')

    #Acquisition Campaign
    df['Acquisition Campaign'] = df['Acquisition Campaign'].notna().astype(int)

    #City
    region_mapping = {
        'Chicago': 'Midwest',
        'San Francisco': 'West',
        'San Diego': 'West',
        'Jacksonville': 'South',
        'Washington': 'East',
        'San Jose': 'West',
        'New York': 'Northeast',
        'Seattle': 'West',
        'San Antonio': 'South',
        'Indianapolis': 'Midwest',
        'Columbus': 'Midwest',
        'Phoenix': 'West',
        'Dallas': 'South',
        'Charlotte': 'South',
        'Denver': 'West',
        'Los Angeles': 'West',
        'Austin': 'South',
        'Philadelphia': 'Northeast',
        'Houston': 'South',
        'Fort Worth': 'South',
        
    }
    df['Region'] = df['City'].map(region_mapping)
    df['Region'] = df['Region'].fillna('Unknown')


    #Time to Offer
    df['Created Date_x'] = pd.to_datetime(df['Created Date_x'])
    df['Created Date_y'] = pd.to_datetime(df['Created Date_y'])
    df['Time to Offer'] = np.where(
        (df['Created Date_y'] - df['Created Date_x']).dt.days >= 0,
        (df['Created Date_y'] - df['Created Date_x']).dt.days,
        0
    )
    df['Time to Offer'] = df['Time to Offer'].fillna(0)

    #Time to Close
    df['Close Date'] = pd.to_datetime(df['Close Date'])
    df['Time to Close'] = np.where(
        (df['Close Date'] - df['Created Date_y']).dt.days >= 0,
        (df['Close Date'] - df['Created Date_y']).dt.days,
        df['Time to Offer']
    )
    df['Time to Close'] = df['Time to Close'].fillna(0)

    #Price
    df['Price'] = df['Price'].fillna(0)

    #Discount Code
    df['Discount code'] = df['Discount code'].notna().astype(int)

    #Pain
    df['Pain'] = df['Pain'].fillna('Not applicable')

    #Target
    df['Target'] = (df['Status_y'] == 'Closed Won').astype(int)

    return df[['Use Case', 'Source', 'Status', 'Reason', 'Acquisition Campaign', 'Converted',
         'Price', 'Discount code', 'Pain', 'Region', 'Time to Offer', 'Time to Close', 'Target']]
print()
print('dataset processed')
df = clean_data(leads ,offers)
print('training Random Forest model')
categorical_cols = df.select_dtypes(include=['object']).columns
df_model = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(df_model.drop('Target', axis=1), df_model['Target'], test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=40, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClasiffication Report")
print(classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

y_probs = model.predict_proba(X_test)[:, 1]  # Probabilidades de la clase positiva

X_test_limited = X_test.iloc[:8797].reset_index(drop=True)
formatted_probs = [f'{prob:.3f}' for prob in y_probs]
results_df = pd.DataFrame({
        'Predicted Probability': formatted_probs,
        'Predictiom': model.predict(X_test_limited),
        'True Label': y_test,
        **X_test_limited.to_dict(orient='list')  # Agrega las demás columnas
    })

print('exporting final dataset with predictions')
results_df.to_parquet('data_output/final_datset_predictions')



print('saving model')

with open('models_binary/random_forest.pkl', 'wb') as file:
    pickle.dump(model, file)

print('operation complete.')
