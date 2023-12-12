from os import getenv
import shimoku_api_python as Shimoku
import pandas as pd
import pyarrow.parquet as pq


access_token = getenv('SHIMOKU_TOKEN')
universe_id: str = getenv('UNIVERSE_ID')
workspace_id: str = getenv('WORKSPACE_ID')


s = Shimoku.Client(
    access_token=access_token,
    universe_id=universe_id,
    verbosity='INFO'
)
s.set_workspace(uuid=workspace_id)
s.boards.force_delete_board(name='Test App')
# s.menu_paths.delete_menu_path(name = 'Graphs')

s.set_board('Shimoku Test App')
s.set_menu_path('Graphs')
s.plt.set_tabs_index(('Tabs', 'Summary'),order=0)


s.plt.html(
    order=0,
    cols_size=12, 
    padding='0',
    html="<img src='https://drive.google.com/uc?id=1GZYPMXS811N06-XiH1nWPurFLyRLXbGR' style='width:100%; height:100%;'>"
)


s.plt.change_current_tab('Original Data')

offers = pd.read_csv(r'..\data\offers.csv')
leads = pd.read_csv(r'..\data\leads.csv')
offers.drop_duplicates(inplace=True)
leads.drop_duplicates(inplace=True)
filtered_leads = leads[leads['Id'].notna()]
df = pd.merge(filtered_leads, offers, how='left', on=['Id', 'Use Case'])

s.plt.html(
    html=(
        "<h2>Descriptive Graphs of the original dataset</h2>" 
        "<p style='color:var(--color-grey-600);'>Left join Leads + Offers</p>"
    ),
    order=0, rows_size=1, cols_size=12,
)


temp_df = pd.Series(df['Use Case'].value_counts() / len(df))
temp_df = temp_df.sort_values(ascending=False)
temp_df  = temp_df.to_frame().reset_index()
['Use Case', 'Source', 'Status_x',
       'Discarded/Nurturing Reason', 'Acquisition Campaign', 'Converted', 'City', 'Status_y',
       'Pain', 'Loss Reason']

s.plt.bar(
    title='Use Case',
    data=temp_df, x='index',
    y='Use Case',
    order=1, rows_size=2, cols_size=4
)
###
temp_df2 = pd.Series(df['Source'].value_counts() / len(df))
temp_df2 = temp_df2.sort_values(ascending=False)
temp_df2  = temp_df2.to_frame().reset_index()
s.plt.bar(
    title='Source',
    data=temp_df2, x='index',
    y='Source',
    order=2, rows_size=2, cols_size=4
)


temp_df3 = pd.Series(df['Status_x'].value_counts() / len(df))
temp_df3 = temp_df3.sort_values(ascending=False)
temp_df3  = temp_df3.to_frame().reset_index()
s.plt.bar(
    title='Status',
    data=temp_df3, x='index',
    y='Status_x',
    order=3, rows_size=2, cols_size=4
)


temp_df4 = pd.Series(df['Discarded/Nurturing Reason'].value_counts() / len(df))
temp_df4 = temp_df4.sort_values(ascending=False)
temp_df4  = temp_df4.to_frame().reset_index()

s.plt.bar(
    title='Discarded/Nurturing Reason',
    data=temp_df4, x='index',
    y='Discarded/Nurturing Reason',
    order=4, rows_size=2, cols_size=4
)

temp_df5 = pd.Series(df['Acquisition Campaign'].value_counts() / len(df))
temp_df5 = temp_df5.sort_values(ascending=False)
temp_df5  = temp_df5.to_frame().reset_index()
s.plt.bar(
    title='Acquisition Campaign',
    data=temp_df5, x='index',
    y='Acquisition Campaign',
    order=5, rows_size=2, cols_size=4
)


temp_df6 = pd.Series(df['City'].value_counts() / len(df))
temp_df6 = temp_df6.sort_values(ascending=False)
temp_df6  = temp_df6.to_frame().reset_index()
s.plt.bar(
    title='City',
    data=temp_df6, x='index',
    y='City',
    order=7, rows_size=2, cols_size=4
)


temp_df7 = pd.Series(df['Converted'].value_counts() / len(df))
temp_df7 = temp_df7.sort_values(ascending=False)
temp_df7  = temp_df7.to_frame().reset_index()
s.plt.pie(
    data=temp_df7,
    names='index', values='Converted',
    order=6, rows_size=2, cols_size=4,
    title='Converted',
)



temp_df8 = pd.Series(df['Status_y'].value_counts() / len(df))
temp_df8 = temp_df8.sort_values(ascending=False)
temp_df8  = temp_df8.to_frame().reset_index()
s.plt.bar(
    title='Status 2',
    data=temp_df8, x='index',
    y='Status_y',
    order=8, rows_size=2, cols_size=4
)


temp_df9 = pd.Series(df['Pain'].value_counts() / len(df))
temp_df9 = temp_df9.sort_values(ascending=False)
temp_df9  = temp_df9.to_frame().reset_index()
s.plt.bar(
    title='Pain',
    data=temp_df9, x='index',
    y='Pain',
    order=9, rows_size=2, cols_size=4
)

temp_df10 = pd.Series(df['Loss Reason'].value_counts() / len(df))
temp_df10 = temp_df10.sort_values(ascending=False)
temp_df10  = temp_df10.to_frame().reset_index()
s.plt.bar(
    title='Loss Reason',
    data=temp_df10, x='index',
    y='Loss Reason',
    order=10, rows_size=2, cols_size=5
)


s.plt.change_current_tab('Transformed Data')

s.plt.html(
    html=(
        "<h2>Descriptive Graphs after some feature engineering</h2>" 
        "<p style='color:var(--color-grey-600);'>features used in the ML classification algorithm</p>"
    ),
    order=0, rows_size=1, cols_size=12,
)

parquet = '../data_output/final_datset'
tabla_parquet = pq.read_table(parquet)
df = tabla_parquet.to_pandas()

temp_df = pd.Series(df['Use Case'].value_counts() / len(df))
temp_df = temp_df.sort_values(ascending=False)
temp_df  = temp_df.to_frame().reset_index()
['Use Case', 'Source', 'Status', 'Reason', 'Acquisition Campaign',
       'Converted', 'Price', 'Discount code', 'Pain', 'Region',
       'Time to Offer', 'Time to Close', 'Target']
s.plt.bar(
    title='Use Case',
    data=temp_df, x='index',
    y='Use Case',
    order=1, rows_size=2, cols_size=4
)


temp_df2 = pd.Series(df['Source'].value_counts() / len(df))
temp_df2 = temp_df2.sort_values(ascending=False)
temp_df2  = temp_df2.to_frame().reset_index()
s.plt.pie(
    data=temp_df2,
    names='index', values='Source',
    order=2, rows_size=2, cols_size=4,
    title='Source',
)




temp_df3 = pd.Series(df['Status'].value_counts() / len(df))
temp_df3 = temp_df3.sort_values(ascending=False)
temp_df3  = temp_df3.to_frame().reset_index()
s.plt.bar(
    title='Status',
    data=temp_df3, x='index',
    y='Status',
    order=3, rows_size=2, cols_size=4
)


temp_df4 = pd.Series(df['Reason'].value_counts() / len(df))
temp_df4 = temp_df4.sort_values(ascending=False)
temp_df4  = temp_df4.to_frame().reset_index()
s.plt.bar(
    title='Reason',
    data=temp_df4, x='index',
    y='Reason',
    order=4, rows_size=2, cols_size=4
)


temp_df5 = pd.Series(df['Acquisition Campaign'].value_counts() / len(df))
temp_df5 = temp_df5.sort_values(ascending=False)
temp_df5  = temp_df5.to_frame().reset_index()
s.plt.pie(
    data=temp_df5,
    names='index', values='Acquisition Campaign',
    order=5, rows_size=2, cols_size=4,
    title='Acquisition Campaign'
)



temp_df6 = pd.Series(df['Converted'].value_counts() / len(df))
temp_df6 = temp_df6.sort_values(ascending=False)
temp_df6  = temp_df6.to_frame().reset_index()
s.plt.pie(
    title='Converted',
    data=temp_df6, names='index',
    values='Converted',
    order=8, rows_size=2, cols_size=4
)


temp_df7 = pd.Series(df['Discount code'].value_counts() / len(df))
temp_df7 = temp_df7.sort_values(ascending=False)
temp_df7  = temp_df7.to_frame().reset_index()
s.plt.pie(
    title='Discount code',
    data=temp_df7, names='index',
    values='Discount code',
    order=7, rows_size=2, cols_size=4
)



temp_df8 = pd.Series(df['Pain'].value_counts() / len(df))
temp_df8 = temp_df8.sort_values(ascending=False)
temp_df8  = temp_df8.to_frame().reset_index()
s.plt.bar(
    title='Pain',
    data=temp_df8, x='index',
    y='Pain',
    order=6, rows_size=2, cols_size=4
)


temp_df9 = pd.Series(df['Region'].value_counts() / len(df))
temp_df9 = temp_df9.sort_values(ascending=False)
temp_df9  = temp_df9.to_frame().reset_index()
s.plt.bar(
    title='Region',
    data=temp_df9, x='index',
    y='Region',
    order=9, rows_size=2, cols_size=4
)


temp_df10 = pd.Series(df['Target'].value_counts() / len(df))
temp_df10 = temp_df10.sort_values(ascending=False)
temp_df10  = temp_df10.to_frame().reset_index()
s.plt.pie(
    title='Target',
    data=temp_df10, names='index',
    values='Target',
    order=10, rows_size=2, cols_size=4
)
s.plt.pop_out_of_tabs_group()