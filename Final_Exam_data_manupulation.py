# Packages
import pandas as pd
from matplotlib import pyplot as plt


# Read excel
path = "C:/Users/Vahid/OneDrive - University of Toronto/MMA - 2022 -/Jupyter files/"
df_raw = pd.read_excel(path+'Data/UBER.xlsx', sheet_name ='Switchbacks' )
df = df_raw

# Read csv
data = pd.read_csv('attrition.csv')

# Check columns
df_raw.columns

# Extract day name
df_raw['day_name'] = df_raw['period_start'].dt.day_name()

# Extract date
df_raw['date'] = df_raw['Created On'].dt.date

# Extract hour
df_raw['time'] = df_raw['Created On'].dt.hour

# Convert a column to time using lambda x
df_raw['time'] = df_raw['period_start'].apply(lambda x: x.time())

# Group by count()
df_raw.groupby(['day_name','wait_time'])['treat'].count().reset_index().rename(columns={'treat':'num_observation'}).sort_values(by='day_name')

# Aggregation
summary = df_raw.groupby(['wait_time'])['rider_cancellations','trips_pool','trips_express'].agg('mean')
summary_day = df_raw.groupby(['day']).agg(Number_orders=('Order', 'nunique'))

df.groupby('column1').agg({'column2': 'mean', 'column3': ['min', 'max']})
grouped = df.groupby(['column1', 'column2'])
mean_agg = grouped['column3'].agg('mean')  # Returns a Series
mean_direct = grouped['column3'].mean()  # Returns a DataFrame

# Pivot and unstack
summary = df_raw.groupby(['day_name','time','wait_time'])['wait_time'].count()
summary.unstack(["time",'wait_time'])

# Cast column to certian type
df_raw['SKU'] = df_raw['SKU'].astype('object')

# Describe dataframe
df_raw.describe(include = 'object')

# Check dataframe information
df_raw.info()

# List unique values
df_raw['Description'].unique()

# Check the total number of null values
df_raw.isnull().sum()

# Drop missing rows
df_raw.dropna(subset = ['text'],inplace=True)

# Drop columns
to_drop = ['REVIEW ID']
df_raw.drop(to_drop, inplace=True, axis=1)

# Remove rows that are not in a list
df_raw = df_raw [~ df_raw['Description'].isin(['CFLs,cfls,7832553,2,,Channa Dal,'])]

# Sort based on columns
df_raw.sort_values(['lift','support'],ascending=False).reset_index(drop=True)

# Merge
df1 = df
df2 = df

merged_df = df1.merge(df2, on='common_column')
merged_df = df1.merge(df2, left_on='df1_column', right_on='df2_column')

merged_df = pd.merge(df1, df2, left_index=True, right_index=True)

# Rename columns
df = df.rename(columns={'old_name1': 'new_name1', 'old_name2': 'new_name2'})

# Drop duplicates
df_raw.drop_duplicates(subset=['REVIEW BY','REVIEW SUBJECT'] ,keep='first', inplace=True)

# Reindexing
df_raw.set_index('REVIEW BY', inplace=True)

# Crosstab
pd.crosstab(df['A'], df['B'])

# Round values
print('Cost of emplyee replacement: $', df.sum(axis=0)['cost_replacement'].round(2))


