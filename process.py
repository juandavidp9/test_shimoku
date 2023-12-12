import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


print('importing leads dataset')
leads = pd.read_csv(r'data\leads.csv')
print('importing offers dataset')
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
print('processing dataset')
df = clean_data(leads ,offers)

print('exporting parquet file')
df.to_parquet('data_output/final_datset')

print('operation complete.')
