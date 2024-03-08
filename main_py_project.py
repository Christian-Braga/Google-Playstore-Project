import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st

###EXPLORATORY DATA ANALYSIS###
#Data Exploration
#################
google_df = pd.read_csv('/Users/christianbraga/Desktop/programming_final_project/sample_google_df.csv')
first_5_rows = google_df.head(5)
first_5_rows[['Currency','Size','Minimum Android','Developer Id']]
last_5_rows = google_df.tail(5)
df_info = google_df.info()
df_columns = google_df.columns
df_shape = google_df.shape
columns_null_values = google_df.isnull().sum()
df_description = google_df.describe().T

#Data Wrangling
###############
#i'm going to work on a copy of our dataset
copy_df = google_df.copy()

#now i'm going to deal with the rows and columns of the dataset
#each row represents an application so for convenience i'm going to put the App Name column as index of the dataset
#and drop the respective column. Then i want to change the column labels in a way that is easier to work with
copy_df.index = copy_df['App Name']
index_adjusted = list(map(lambda x : x.lower().replace(' ','_'),copy_df.index))
copy_df.index = index_adjusted
copy_df.index.name = 'app_name'
copy_df.drop('App Name',axis=1,inplace = True)
new_column_labels = list(map(lambda x : x.lower().replace(' ','_'),copy_df.columns))
copy_df.columns = new_column_labels

#remove all columns that are useless for the purpose of the analysis
copy_df.drop(['app_id','developer_id','minimum_installs','maximum_installs','minimum_android','currency','developer_website','developer_email','privacy_policy','last_updated','editors_choice','scraped_time'],axis=1,inplace=True)

#I will fill the null values of the rating and rating count columns with their average value for the specific category.
#Function to find the mean rating and ratign count for each category.
def mean_rating_and_rating_count_for_category():
    avg_rating_and_rating_count_for_category =  copy_df.groupby('category')[['rating','rating_count']].mean()
    avg_df = pd.DataFrame(avg_rating_and_rating_count_for_category).T
    return avg_df.round(1)

#Function to fill the null values in the rating column with the average rating value for the specific category
def fillna_rating_with_mean_for_category():
    copy_df['category'].astype(str)
    categories = copy_df['category'].value_counts().index.tolist()
    subset = copy_df[copy_df['rating'].isnull()]
    for i in categories:
        musk = subset['category'] == i
        subset.loc[musk,'rating'] = mean_rating_and_rating_count_for_category()[i][0].astype(float)
        copy_df.update(subset)

#Function to fill the null values in the rating_count column with the average rating value for the specific category
def fillna_rating_count_with_mean_for_category():
    copy_df['category'].astype(str)
    categories = copy_df['category'].value_counts().index.tolist()
    subset = copy_df[copy_df['rating_count'].isnull()]
    for i in categories:
        musk = subset['category'] == i
        subset.loc[musk,'rating_count'] = mean_rating_and_rating_count_for_category()[i][0].astype(float)
        copy_df.update(subset)

#now i fill the null value of the column installs with the mean of this column
#before that I need to adjust all the values in this column by removing the + and turning all the values into float
instal_column = copy_df['installs'].astype(str)
list_simbol_removed = list(map(lambda x : x.replace('+','').replace(',',''),instal_column))
copy_df['installs'] = list_simbol_removed

#fill the null value of the column installs with the mean of this columns
installs_float_column = copy_df['installs'].astype(float)
installs_float_column.fillna(installs_float_column.mean(),inplace = True)

#fill the null value of the column size with the mean of this column
#before that I need to adjust all the values in this column by removing all the dimension symbols and
#turning all the values into float
#to remove all the dimension symbol i will remove from each value of the column all the characters that are not a digit
new_column_values = []
for i in copy_df['size'].astype(str):
    new_string = '0' #allows me to handle any empty strings that would make difficult
    #to transform the data to float later (this 0 character will be removed)
    for characters in i:
        if characters.isnumeric():
            new_string += characters
    new_column_values.append(new_string)
#i put this new list as value of the column size
copy_df['size'] = new_column_values
size_float_column = copy_df['size'].astype(float)
#the values 0.0 will be one of the previous null values or one of the previous values that was not composed
#of numbers, so we replace them with the average value of the size column.
mean_size = size_float_column.mean().round(1)
for i in size_float_column:
    if i == 0.0:
        i = mean_size
#adjust the value in a proper way
copy_df['size'] = size_float_column

#i drop all rows that have a null value in the released column since I cannot replace it with other data
copy_df.dropna(inplace = True)

#modify the released column to keep only the year, and changing the column values to int
released_years = list(map(lambda x : x[-4:],copy_df['released'].astype(str)))
copy_df['released'] = released_years
copy_df['released'] = copy_df['released'].astype(int)
copy_df.head(3)

#change the data in the category column so that it is easier to work with it
category_values_adjusted = list(map(lambda x : x.lower().replace(' ','_'),copy_df['category']))
copy_df['category'] = category_values_adjusted

#to apply the model i need to codify the boolean values of the columns into number
#i create a function to transform the value True in 1 and the value False in 0 for the input columns
def codify_boeolean_values(column_name):
    codify_values = []
    for i in copy_df[column_name].astype(str):
        if i == True:
            i = 1
            codify_values.append(i)
        else:
            i = 0
            codify_values.append(i)
    copy_df[column_name] = codify_values
    copy_df[column_name] = copy_df[column_name].astype(int)
codify_boeolean_values('in_app_purchases')
codify_boeolean_values('ad_supported')
codify_boeolean_values('free')

#now since each rows rappresent an application i check for duplicates
duplicate_apps = copy_df.index.value_counts()
duplicate_df = pd.DataFrame(duplicate_apps[duplicate_apps > 1])
duplicate_df.index #duplicate rows

#I handle these duplicate rows by keeping only the row with the most recent year of release
def delete_duplicate_rows():
    value_count_apps = copy_df.index.value_counts()
    duplicate_dataframe = pd.DataFrame(value_count_apps[value_count_apps > 1])
    list_duplicate_apps = duplicate_df.index.astype(str) #duplicate apps
    for app in list_duplicate_apps:
        duplicated_app = copy_df.loc[app]
        most_recent_year = duplicated_app['released'].astype(int).max()
        copy_df.drop(duplicated_app[duplicated_app['released'] != most_recent_year].index,inplace = True)

#to be sure that all the columns are considered with the right type
copy_df['free'] = copy_df['free'].astype(int)
copy_df['ad_supported'] = copy_df['ad_supported'].astype(float)
copy_df['in_app_purchases'] = copy_df['in_app_purchases'].astype(float)
copy_df['installs'] = copy_df['installs'].astype(int)
copy_df['category'] = copy_df['category'].astype(str)
copy_df['content_rating'] = copy_df['content_rating'].astype(str)
        
#PLOTS
######
#to avoid warning messages
warnings.filterwarnings('ignore')

#mean rating for category
category = list(mean_rating_and_rating_count_for_category().loc['rating'].sort_values(ascending = True).index)
mean_rating = list(mean_rating_and_rating_count_for_category().loc['rating'].sort_values(ascending = True))
plt.figure(figsize=(5,7)) #5,7
plt.title('mean rating for category')
plt.xlabel('mean rating')
plt.ylabel('cateogories')
plt.barh(category,mean_rating, color = 'green')
plt.yticks(fontsize = 6)
#plt.show()

#top 10 categories for mean rating
mean_rating_top_10_category = mean_rating_and_rating_count_for_category().loc['rating'].sort_values(ascending=False)[0:10]
top_10_categories_label = mean_rating_and_rating_count_for_category().loc['rating'].sort_values(ascending=False)[0:10].index
plt.figure(figsize=(10,3))
plt.title('top 10 category for mean rating')
plt.xlabel('categories')
plt.ylabel('mean rating')
bar_chart = plt.bar(top_10_categories_label,mean_rating_top_10_category,color = 'coral',width = 0.6)
bar_chart[0].set_hatch('o')
plt.xticks(rotation = 45,fontsize=8)
#plt.show()

#downloads for years for the apps in the 3 categories with the highest mean rating
def number_of_installs_in_category(category):
    category_apps = copy_df[copy_df['category']==category]
    list_of_years = sorted(list(category_apps['released'].value_counts().index)) #total years
    number_of_installs_for_years = [] #total number of downloads in the category for year
    for i in list_of_years:
        subset_for_years = category_apps[category_apps['released'] == i]
        installs_for_years = subset_for_years['installs'].astype(int).sum()
        number_of_installs_for_years.append(installs_for_years)
    return number_of_installs_for_years 
def list_of_years_in_category(category):
    category_apps = copy_df[copy_df['category']==category]
    list_of_years = sorted(list(category_apps['released'].value_counts().index))
    return list_of_years
plt.figure(figsize=(10,4))
plt.title('downloads for years for the apps in the 3 categories with the highest mean rating')
plt.plot(list_of_years_in_category('casino'),number_of_installs_in_category('casino'),color='blue',marker = 'o', label='casino')
plt.plot(list_of_years_in_category('role_playing'),number_of_installs_in_category('role_playing'),color='green',marker = 'o', label='role_playing')
plt.plot(list_of_years_in_category('racing'),number_of_installs_in_category('racing'),color='red',marker = 'o', label='racing')
plt.xlabel('years')
plt.ylabel('downloads')
plt.legend()
plt.grid(linestyle='--')
plt.xticks(fontsize=8)
#plt.show()

#downloads over the year of the top 3 category for mean rating
fig,ax = plt.subplots(1,3,figsize=(23,7))
ax[0].set_title('download for years category casino')
ax[0].set_xlabel('years')
ax[0].set_ylabel('download')
ax[0].legend()
ax[0].set_xticks(list_of_years_in_category('casino'))
ax[0].grid(linestyle='--')
ax[0].plot(list_of_years_in_category('casino'),number_of_installs_in_category('casino'),color='blue',marker = 'o', label='casino')
ax[1].set_title('download for years category role_playing')
ax[1].set_xlabel('years')
ax[1].set_ylabel('download')
ax[1].set_xticks(list_of_years_in_category('role_playing'))
ax[1].grid(linestyle='--')
ax[1].plot(list_of_years_in_category('role_playing'),number_of_installs_in_category('role_playing'),color='green',marker = 'o', label='role_playing')
ax[2].set_title('download for years category racing')
ax[2].set_xlabel('years')
ax[2].set_ylabel('download')
ax[2].set_xticks(list_of_years_in_category('racing'))
ax[2].grid(linestyle='--')
ax[2].plot(list_of_years_in_category('racing'),number_of_installs_in_category('racing'),color='red',marker = 'o', label='racing')
fig.suptitle('downloads over the year of the top 3 category for mean rating')
#fig.show()

#Percentage of apps in top 10 categories by average rating
top_10_categories_label #top 10 categories for mean rating
number_of_apps_in_top_10_categories = []
for i in top_10_categories_label:
    number_of_apps_in_top_10_categories.append(copy_df[copy_df['category'] == i].shape[0])
number_of_apps_in_top_10_categories
plt.figure(figsize=(5,7))
plt.title('Apps distribution in the top 10 categories by average rating')
explosion = (0.0,0.0,0.0,0.3,0.0,0.0,0.0,0.0,0.0,0.0)
plt.pie(number_of_apps_in_top_10_categories,labels=top_10_categories_label,autopct='%.1f%%',shadow=True,startangle=90,explode=explosion,pctdistance=0.80)
#plt.show()

#Number of apps in top 10 categories by average rating
plt.figure(figsize=(7,4))
plt.title('Number of apps in top 10 categories by average rating')
plt.xlabel('number of app')
plt.ylabel('category')
plt.barh(top_10_categories_label,number_of_apps_in_top_10_categories,color='purple')
plt.yticks(fontsize=7)
#plt.show()

#scatter plot rating of the app based on the price and
#scatter plot rating of the app based on the size
prices = list(copy_df['price'])
ratings = list(copy_df['rating'])
sizes = list(copy_df['size'])
colors = np.arange(len(ratings))
fig,ax = plt.subplots(1,2,figsize=(13,5))
ax[0].set_title('relationship between app rating and price')
ax[0].set_xlabel('price')
ax[0].set_ylabel('rating')
ax[0].scatter(prices,ratings,c=colors,cmap='plasma')
ax[1].set_title('relationship between app rating and size')
ax[1].set_xlabel('size')
ax[1].set_ylabel('rating')
ax[1].scatter(sizes,ratings,c=colors,cmap='rainbow')
fig.suptitle('RATING IN RELATION TO PRICE AND SIZE')
#fig.show()

#distribution of the ratings
plt.figure(figsize=(5,4))
#deal with outlaiers
musk_different_from_min = copy_df['rating'] != copy_df['rating'].min()
df_without_outliers = copy_df[musk_different_from_min]
sns.distplot(df_without_outliers['rating'],hist = True)
plt.xticks()
#plt.show()

#heatmap correlation of the dataset
plt.figure(figsize=(8,4))
df_for_heatmap = copy_df[['rating','rating_count','price','size','released','installs']]
sns.heatmap(df_for_heatmap.corr(),annot=True,fmt='.2f')
#plt.show()

#correlation of the top 3 most downloaded categories
df_group_category = copy_df.groupby('category').sum()
download_for_category = df_group_category['installs'].sort_values(ascending=False)
top_3_most_dowloaded_categories = list(download_for_category.index[0:3])
fig,axs = plt.subplots(1,3,figsize= (20,6))
#first category
axs[0].set_title(top_3_most_dowloaded_categories[0])
apps_category_1 = copy_df[copy_df['category'] == top_3_most_dowloaded_categories[0]]
heatmap_category_1 = apps_category_1[['rating','rating_count','price','size','released','installs']]
sns.heatmap(heatmap_category_1.corr(),annot=True,fmt='.2f',ax=axs[0])
#second category
axs[1].set_title(top_3_most_dowloaded_categories[1])
apps_category_2 = copy_df[copy_df['category'] == top_3_most_dowloaded_categories[1]]
heatmap_category_2 = apps_category_2[['rating','rating_count','price','size','released','installs']]
sns.heatmap(heatmap_category_2.corr(),annot=True,fmt='.2f',ax=axs[1])
#third category
axs[2].set_title(top_3_most_dowloaded_categories[2])
apps_category_3 = copy_df[copy_df['category'] == top_3_most_dowloaded_categories[2]]
heatmap_category_3 = apps_category_3[['rating','rating_count','price','size','released','installs']]
sns.heatmap(heatmap_category_3.corr(),annot=True,fmt='.2f',ax=axs[2])
fig.suptitle('correlation in the top 3 categories for downloads')
#fig.show()

#MODEL
######
#I am going to build a multiple linear regression to predict the rating of an application based on its features.
#encoding the data:most of the data encoding has been done in the data wrangling part, we still need to encode the category and content_rating column.

#encoding of the categories
copy_df['category'].unique() #checking of the unique categories

#i will use the class LabelEncoder
label_encoder = LabelEncoder()
encoded_data = list(label_encoder.fit_transform(copy_df['category']))
encoded_series = pd.Series(encoded_data, name = 'new label')
decoded_data = label_encoder.inverse_transform(encoded_data)
decoded_series = pd.Series(decoded_data, name = 'old label')
new_category_label = pd.DataFrame({'new_label':encoded_series,'old_label':decoded_series})

#i do the same for the content_rating column
copy_df['content_rating'].unique() #checking of the unique content_rating labels
label_encoder2 = LabelEncoder()
encoded_data2 = list(label_encoder2.fit_transform(copy_df['content_rating']))
encoded_series2 = pd.Series(encoded_data2, name = 'new label')
decoded_data2 = label_encoder2.inverse_transform(encoded_data2)
decoded_series2 = pd.Series(decoded_data2, name = 'old label')
new_content_rating_label = pd.DataFrame({'new_label':encoded_series2,'old_label':decoded_series2})

#apply the modifications to our dataset
copy_df['category'] = encoded_data
copy_df['content_rating'] = encoded_data2

#exporting the cleaned dataset in a csv format
copy_df.to_csv('cleaned_df.csv',index = False)

#model building
X = copy_df.drop('rating',axis=1) #all the features of the dataset without the column rating
Y = copy_df['rating'] #the rating column

#cross validation
#split the data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#print the shape of our train and testing sets
#print('training set shape:',X_train.shape, Y_train.shape)
#print('testing set shape:',X_test.shape, Y_test.shape)

#build the model
lr = LinearRegression()

#train the model
lr.fit(X_train,Y_train)

#predict on the testing set
Y_pred = lr.predict(X_test)

#evaluate the perfomance of the model
r2 = r2_score(Y_test,Y_pred)
mse = mean_squared_error(Y_test,Y_pred)
mae = mean_absolute_error(Y_test,Y_pred)
intercept = lr.intercept_
#print('R2 score',r2)
#print('mean squared error',mse)
#print('mean absolute error',mae)
#print('intercept',intercept)
#this model is not accurate, i try with another one

#PLOT MULTILINEAR REGRESSION: ACTUAL VS PREDICTED
plt.figure(figsize=(9,5))
colors = np.linspace(0, 1, len(Y_test))
plt.scatter(Y_test,Y_pred,c=colors,cmap='viridis',alpha=0.8,label='Data Points')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Multilinear Regression: Actual vs. Predicted")
z1 = np.polyfit(Y_test, Y_pred, 1)
p1 = np.poly1d(z1)
plt.plot(Y_test, p1(Y_test), color='red', label='Trendline',linestyle='--')
color_bar = plt.colorbar()
color_bar.set_label('Color Intensity')
#print('R2 score',r2)
plt.text(5, 0.4, f'R2 score:{r2.round(4)}', fontsize=10, color='red', ha='right', va='bottom')
plt.legend()
#plt.show()

#RANDOM FOREST REGRESSION
x_train,x_test,y_train,y_test = train_test_split(X,Y)
#print(x_train.shape)
#print(y_train.shape)

rf = RandomForestRegressor()
rf.fit(x_train,y_train)

#random forest regression prediction
y_pred = rf.predict(x_test)

#evaluate the perfomance of the model
r2_random_for = r2_score(y_test,y_pred)
mse_random_for = mean_squared_error(y_test,y_pred)
mae_random_for = mean_absolute_error(y_test,y_pred)
#print('R2 score',r2_random_for)
#print('mean squared error',mse_random_for)
#print('mean absolute error',mae_random_for)
#radom forest regression perform better

#PLOT OF RANDOM FOREST REGRESSION
colors = np.linspace(0, 1, len(y_test))
plt.figure(figsize=(9,5))
plt.scatter(y_test, y_pred, c=colors, cmap='plasma', alpha=0.8, label='Data Points')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest Regression: Actual vs. Predicted")
z2 = np.polyfit(y_test, y_pred, 1)
p2 = np.poly1d(z2)
plt.plot(y_test, p2(y_test), color='red', label='Trendline',linestyle='--')
color_bar_2 = plt.colorbar()
color_bar_2.set_label('Color Intensity')
plt.text(max(y_test), min(y_pred), f'R2 score: {r2_random_for.round(4)}', fontsize=10, color='green', ha='right', va='bottom')
plt.legend()
#plt.show()

#determination of residual errors and plot
errors = y_test - y_pred
plt.figure(figsize=(9,5))
plt.scatter(y_test, errors,c=colors, cmap='plasma', alpha=0.8)
plt.axhline(y=0, color='red', linestyle='--', label='Errore nullo')
color_bar_3 = plt.colorbar()
color_bar_3.set_label('Color Intensity')
plt.title('Random Forest Regression: Errori Residui')
plt.legend()
plt.xlabel('Valori Effettivi')
plt.ylabel('Errori Residui')
#plt.show()

#testing the model on a row of the dataset
row1 = x_test.iloc[-2]
#row1
row1_reshaped = row1.values.reshape(1,-1)
row_test_name = row1.name
#row_test_name
copy_df.loc[row_test_name]
ypredict_test = rf.predict(row1_reshaped)
ypredict_test[0].round(4)

#function to predict the rating based on the user's values.
def rating_prediction():
    data = {}
    category = input('category: ')
    category_labels_df = new_category_label[new_category_label['old_label'] == str(category)]
    category_encoded = category_labels_df['new_label'].iloc[0]
    data['category'] = float(category_encoded)
    rating_count = input('rating count: ')
    data['rating_count'] = float(rating_count)
    installs = input('installs: ')
    data['installs'] = float(installs)
    free = input('free: ')
    data['free'] = float(free)
    price = input('price: ')
    data['price'] = float(price)
    size = input('size: ')
    data['size'] = float(size)
    released = input('released year: ')
    data['released'] = float(released)
    content_rating = input('content rating: ')
    content_rating_labels_df = new_content_rating_label[new_content_rating_label['old_label'] == content_rating]
    content_rating_encoded = content_rating_labels_df['new_label'].iloc[0] 
    data['content_rating'] = float(content_rating_encoded)
    ad_supp = input('ad supported: ')
    data['ad_supported'] = float(ad_supp)
    in_app_purch = input('in app purchases: ')
    data['in_app_purchases'] = float(in_app_purch)
    values_to_predict = pd.DataFrame(data,index=[0]) 
    prediction = rf.predict(values_to_predict)
    return prediction[0].round(4)
#rating_prediction()











