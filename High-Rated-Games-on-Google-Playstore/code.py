# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data = pd.read_csv(path)

data.hist(column='Rating', bins=25)
plt.show()

data = data[data['Rating']<=5]

data.hist(column='Rating', bins=25)
plt.show()
#Code ends here


# --------------
# code starts here
total_null = data.isnull().sum()

percent_null = (total_null/data.isnull().count())

missing_data = pd.concat([total_null, percent_null], keys=['Total','Percent'], axis=1)
print(missing_data)

data.dropna(inplace=True)

total_null_1 = data.isnull().sum()

percent_null_1 = (total_null/data.isnull().count())

missing_data_1 = pd.concat([total_null_1, percent_null_1], keys=['Total','Percent'], axis=1)
print(missing_data_1)
# code ends here


# --------------

#Code starts here
plt.figure(figsize = (10,10))
sns.catplot(x="Category", y="Rating", data=data, kind="box", height=10)

plt.xticks(rotation=90)
plt.title('Rating vs Category [BoxPlot]')

plt.show()
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
print(data['Installs'].value_counts())

data['Installs'] = data['Installs'].apply(lambda x: int(str(x)[:-1].replace(',','')))

le = LabelEncoder()

data['Installs'] = le.fit_transform(data['Installs'])

plt.figure(figsize = (10,10))
sns.regplot(x="Installs", y="Rating", data=data)

plt.title('Rating vs Installs [RegPlot]')
plt.show()
#Code ends here





# --------------
#Code starts here
print(data['Price'].value_counts())

data['Price']=data['Price'].str.replace('$','')

data['Price'] = data['Price'].astype(float)

plt.figure(figsize = (10,10))
sns.regplot(x="Price", y="Rating", data=data)

plt.title('Rating vs Price [RegPlot]')
plt.show()
#Code ends here


# --------------

#Code starts here
print( len(data['Genres'].unique()) , "genres")

data['Genres'] = data['Genres'].apply(lambda x:x.split(';')[0])

gr_mean  = data[['Genres','Rating']].groupby('Genres', as_index=False).mean()

print(gr_mean.describe())

gr_mean = gr_mean.sort_values(by='Rating')

print(gr_mean.head(1))
print(gr_mean.tail(1))
#Code ends here


# --------------

#Code starts here
data['Last Updated'] = pd.to_datetime(data['Last Updated'])

max_date = max(data['Last Updated'])

data['Last Updated Days'] = (max_date - data['Last Updated']).dt.days

plt.figure(figsize = (10,10))

sns.regplot(x="Last Updated Days", y="Rating", data=data)

plt.title('Rating vs Last Updated [RegPlot]')
plt.show()
#Code ends here


