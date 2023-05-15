# Packages
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd ,seaborn as sns, numpy as np ,matplotlib.pyplot as plt

path = "C:/Users/Vahid/OneDrive - University of Toronto/MMA - 2022 -/Jupyter files/"
df_raw = pd.read_excel(path+'Data/UBER.xlsx', sheet_name ='Switchbacks' )
df = df_raw


# Histograms with two curves in one plot
sns.set(rc = {'figure.figsize':(20,10)})
sns.set(font_scale = 1)

g = sns.displot(data=df_raw, x='trips_pool', hue='wait_time', kind='kde', fill=True, palette=sns.color_palette('bright')[:2], height=5, aspect=1.5)
g.figure.subplots_adjust(top=0.9);
g._legend.set_title('Wait Time')
g.set(xlabel='Number of Pool Trips', ylabel='Density')


# Bar charts - Horizontal
summary = df_raw

fig, ax = plt.subplots(figsize=(12, 5))
sns.set_style("whitegrid")

sns.barplot(y='Description', x='# Orders', data=summary.iloc[0:10,])
plt.title('Top Ten Popular Products Among Baskets', fontsize=18)
plt.ylabel(' ', fontsize=18)
plt.xlabel('# Orders', fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()


# Bar charts - Vertical
summary_time = df_raw.groupby('time').agg(Number_orders=('Order', 'nunique'))

fig, ax = plt.subplots(figsize=(12, 5))
sns.set_style("whitegrid")

summary_time['Number_orders'].plot(kind="bar",ax=ax)
plt.title('Total Number of Item Solds Time-Wise', fontsize=18)
plt.xlabel('Time', fontsize=18)
plt.ylabel('# Items sold', fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()


# Line Chart with two lines
wait_time_london =[]
wait_time_toronto = []
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(1,2,.1), wait_time_toronto, label = "Toronto")
ax.plot(np.arange(1,2,.1), wait_time_toronto, label = "London")
ax.set_ylabel("Average Waiting Time(days) ", fontsize=14)
ax.set_xlabel("The CV of Arrival Rate", fontsize=14)
plt.legend(bbox_to_anchor=(1, .5))
plt.title('Effect of Increasing in Coefficient of Variation of Service Time on Wait List')
plt.show()  


# Box plot with many categories
result_london = pd.DataFrame()
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(data=result_london, x='sim_time', y='utilization')
ax.set_ylabel("Utilization ", fontsize=14)
ax.set_xlabel("Simulation Time", fontsize=14)
plt.legend(bbox_to_anchor=(1.6, .5))
plt.title('The utilization in London with 50 number of trails per simulation time ')
plt.show() 


# Countplot
fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(x='Attrition',hue='JobLevel',data=df)


# Multiple Scatter Plots
for col in df.describe().columns:    
    fig = plt.figure(figsize=(14, 5))
    plt.scatter(df[col], df['MonthlyIncome'])
    plt.ylabel("MonthlyIncome", fontsize=18)
    plt.xlabel(col, fontsize=18)
    plt.title("Monthly Income vs. {}".format(col), fontsize=18)
    plt.show()


# ------------------------------------------------
####   Bar chart  ####

# seaborn bar chart
ax = sns.countplot(df["Churn?"])
ax.set_title("Churning Customers")

# matplotlib bar chart
df["Churn?"].value_counts().plot(kind='bar', title= "Churning Customers")

# ------------------------------------------------
####  Stacked Bar chart and Crosstab  ####


# pandas crosstab
churn_crosstab = pd.crosstab(df["Churn?"], df["Int'l Plan"], margins=False)

# seaborn stacked bar charts
sns.countplot(x="Int'l Plan", hue="Churn?", data=df)

churn_crosstab = churn_crosstab.transpose()
churn_crosstab.plot(kind = 'bar', stacked = True)

# stacked bar chart normalized
churn_crosstab_norm = churn_crosstab.div(churn_crosstab.sum(axis=1),axis=0)
churn_crosstab_norm.plot(kind = 'bar', stacked=True)

# ------------------------------------------------
####  Histograms  ####

# pandas histogram
df.hist(figsize=(14,10))
df.plot(figsize=(14,10), kind='density', subplots=True, layout=(4,4), sharex=False)

# seaborn histogram
sns.kdeplot(df["CustServ Calls"])

# matplotlib histogram
churn_csc_T = df[df["Churn?"] == "True."]["CustServ Calls"]
churn_csc_F = df[df["Churn?"] == "False."]["CustServ Calls"]

plt.hist([churn_csc_T, churn_csc_F], bins = 10, stacked = True)
plt.legend(['Churn = True', 'Churn = False'])
plt.title('Histogram of Customer Service Calls with Churn Overlay')
plt.xlabel('Customer Service Calls')
plt.ylabel('Frequency')
xlabels = np.arange(10)  # the labels
xpos = [x*0.9+0.45 for x in xlabels]  # the label locations
plt.xticks(xpos, xlabels)

# matplotlib normalized histogram
(n, bins, patches) = plt.hist([churn_csc_T, churn_csc_F], bins = 10, stacked = True)
n[1] = n[1] - n[0]
n_table = np.column_stack((n[0], n[1]))
n_norm = n_table / n_table.sum(axis=1)[:, None]
ourbins = np.column_stack((bins[0:10], bins[1:11]))

plt.bar(x = ourbins[:,0], height = n_norm[:,0], width = ourbins[:, 1] - ourbins[:, 0])
plt.bar(x = ourbins[:,0], height = n_norm[:,1], width = ourbins[:, 1] - ourbins[:, 0], bottom = n_norm[:,0])

plt.legend(['Churn = True', 'Churn = False'])
plt.title('Normalized Histogram of Customer Service Calls with Churn Overlay')
plt.xlabel('Customer Service Calls')
plt.ylabel('Proportion')
xpos = [x-0.45 for x in xpos]
plt.xticks(xpos, xlabels)

# ------------------------------------------------
####  Box Plots  ####

# pandas boxplot
df.boxplot(figsize=(14,10))
plt.xticks(rotation=90)

# matplotlib individual boxplot
df.plot(figsize=(14,10), kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)

# seaborn boxplot
sns.boxplot(x = df["Churn?"], y = df["CustServ Calls"], data = df)

# ------------------------------------------------
####  Scatter Plots  ####

# seaborn scatter plot
sns.scatterplot(x = "Day Mins", y = "Eve Mins", hue="Churn?", data = df)


# seaborn scatter matrix
from pandas.plotting import scatter_matrix

scatter_matrix(df)

# matplotlib scatter plot
for column in df.columns:
    if column != 'Median_income':
        df.plot(kind = "scatter", x = 'Median_income', y = column, alpha=0.1)

# ------------------------------------------------
####  Correlation Plots  ####

# pandas correlation matrix
correlations = df.corr()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)

ticks = range(0,16,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
numeric_columns = df.select_dtypes(include='number')
ax.set_xticklabels(numeric_columns, rotation=90)
ax.set_yticklabels(numeric_columns)


# seaborn correlation heatmap
sns.heatmap(df.corr(method='pearson'))

# seaborn correlation heatmap
plt.figure(figsize=(25, 15))
plt.suptitle('Correlations', fontsize = 30, color= 'teal')
corr = df.corr()
sns.heatmap(corr, annot=True)
plt.show()

# ------------------------------------------------
####  KMeans Plots  ####

# visualize silhouette_sample_score
fig, ax = plt.subplots(7, 2, figsize=(10,30))
for k in range(2,15):
    
    np.random.seed(84) 

    km = KMeans(n_clusters=k)
    q, mod = divmod(k, 2)

    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(scaled_df) 

# visualize inertias
ks = range(1,20)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    
    np.random.seed(seed)
    
    model = KMeans(n_clusters = k)
    
    # Fit model to samples
    model.fit(scaled_df)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# visualize silhouette line plot
ks = range(2,15)
silhouette_score_list = []
for k in ks:
    
    np.random.seed(seed)
    
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)
    
    # Fit model to samples
    model.fit_predict(scaled_df)
    
    # Append the inertia to the list of inertias
    s_score = silhouette_score(scaled_df, model.labels_, metric='euclidean')
    silhouette_score_list.append(model.inertia_)
    
    
plt.plot(ks, silhouette_score_list, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('silhouette score')
plt.xticks(ks)
plt.show()