import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import io

google_df = pd.read_csv('Project_files/sample_google_df.csv')
cleaned_df = pd.read_csv('Project_files/cleaned_df.csv')

copy_df = google_df.copy()
def home():
    title = st.title('Google Playstore Project')
    st.write('')
    st.image('Project_files/goog.webp')
    st.write('')
    st.write('')
    st.subheader('**DESCRIPTION OF THE PROJECT**')
    st.write('')
    st.write('The goal of this project is to build a machine learning model that can predict the rating of a google playstore application based on its characteristics. To do that, I first explored the dataset to understand how it is composed and what kind of information we have. Then I performed some data cleaning operations to correct possibly null values and encode the data to fit the model. Then I made some graphs to better understand the information in our dataset and finally I built a machine learning model that can predict the rating of an application.')
    st.write('')
    st.write('the dataset I worked on can be found on Kaggle.com at the following link: https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps')
    st.markdown('**- You can use the sidebar to navigate through the website.**')

def data_exploration_page():
    st.title('Data Exploration')
    st.write('')
    st.subheader('The project was carried out on a subset of 10000 rows taken randomly from the original dataset.')
    st.write('')
    st.subheader('The row data:')
    st.write('')
    st.dataframe(google_df)
    buffer = io.StringIO()
    google_df.info(buf = buffer)
    info_str = buffer.getvalue()
    st.write('')
    st.write('')
    st.subheader('Information of the row dataset:')
    st.write('')
    st.text(info_str)
    st.write('')
    st.write('')
    st.write('''
- Each row of the dataset represents an application of the google play store.
- In some columns the data contain symbols that we have to remove
- In some columns we have data in a label format or in a Boolean format that we need to encode
- We have some null values that we must deal with''')
    st.write('')
    st.write('')
    st.subheader('Description of the row dataset:')
    st.write('')
    st.text(google_df.describe().T)
    st.write('')
    st.write('')
    st.subheader('Shape of the row dataset:')
    st.write('')
    st.text(google_df.shape)

def data_wrangling():
    st.title('Data Wrangling')
    st.write('')
    st.subheader('The row data:')
    st.write('')
    st.dataframe(copy_df)
    st.write('')
    st.write('')
    st.subheader('Steps of the data cleaning: ')
    st.write('')
    st.write('''
- Adjust the index of the dataset and the name of the columns
- Remotion of useless columns
- Fill the null values of the rating and rating count columns with their average value for the specific category.
- Fill the null value of the column installs with the mean of this column, and elimination of all the symbol in the column's values
- Fill the null value of the column size with the mean of this column, and elimination of all the symbol in the column's values
- Modify the released column to keep only the year, and changing the column values to int
- Encoding the boolean value of the columns for the application of the model
- I handle the duplicate rows by keeping only the app with the most recent year of release
- Cast the values of each column
- Encoding the columns category and content rating''')
    options = ['index and column management','remove usless columns','fill the null values of rating and rating count columns','fill the null value of the column installs','column size, fill null value and encoding of the data','manage the released column','encoding the boolean values','handle rows duplicates','casting column\'s value','encoding category and content rating column']
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    select_options = st.selectbox('**Select the options that you want to see**: ', options)
    if 'index and column management' in select_options:
        st.write('Each row represents an application so for convenience i\'m going to put the App Name column as index of the dataset and drop the respective column. Then i want to change the column labels in a way that is easier to work with')
        code_col_ind = '''copy_df.index = copy_df['App Name']
                       index_adjusted = list(map(lambda x : x.lower().replace(' ','_'),copy_df.index))
                       copy_df.index = index_adjusted
                       copy_df.index.name = 'app_name'
                       copy_df.drop('App Name',axis=1,inplace = True)
                       new_column_labels = list(map(lambda x : x.lower().replace(' ','_'),copy_df.columns))
                       copy_df.columns = new_column_labels'''
        st.code(code_col_ind)
    elif 'remove usless columns' in select_options:
        st.write('remove all columns that are useless for the purpose of the analysis')
        code_remove_col = '''copy_df.drop(['app_id','developer_id','minimum_installs','maximum_installs','minimum_android','currency','developer_website','developer_email','privacy_policy','last_updated','editors_choice','scraped_time'],axis=1,inplace=True)'''
        st.code(code_remove_col)
    elif 'fill the null values of rating and rating count columns' in select_options:
        st.write('I will fill the null values of the rating and rating count columns with their average value for the specific category.')
        code_fill_rating_rating_count = ''' #Function to find the mean rating and ratign count for each category.
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
        copy_df.update(subset)'''
        st.code(code_fill_rating_rating_count)
    elif 'fill the null value of the column installs' in select_options:
        fill_installs_code = ''' 
#now i fill the null value of the column installs with the mean of this column
#before that I need to adjust all the values in this column by removing the + and turning all the values into float
instal_column = copy_df['installs'].astype(str)
list_simbol_removed = list(map(lambda x : x.replace('+','').replace(',',''),instal_column))
copy_df['installs'] = list_simbol_removed

#fill the null value of the column installs with the mean of this columns
installs_float_column = copy_df['installs'].astype(float)
installs_float_column.fillna(installs_float_column.mean(),inplace = True)'''
        st.code(fill_installs_code)
    elif 'column size, fill null value and encoding of the data' in select_options:
        code_size_column = '''
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
copy_df['size'] = size_float_column'''
        st.code(code_size_column)
    elif 'manage the released column' in select_options:
        released_code = '''
#modify the released column to keep only the year, and changing the column values to int
released_years = list(map(lambda x : x[-4:],copy_df['released'].astype(str)))
copy_df['released'] = released_years
copy_df['released'] = copy_df['released'].astype(int)'''
        st.code(released_code)
    elif 'encoding the boolean values' in select_options:
        boolean_code = '''
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
codify_boeolean_values('free')'''
        st.code(boolean_code)
    elif 'handle rows duplicates' in select_options:
        duplicate_code = '''
#I handle these duplicate rows by keeping only the row with the most recent year of release
def delete_duplicate_rows():
    value_count_apps = copy_df.index.value_counts()
    duplicate_dataframe = pd.DataFrame(value_count_apps[value_count_apps > 1])
    list_duplicate_apps = duplicate_df.index.astype(str) #duplicate apps
    for app in list_duplicate_apps:
        duplicated_app = copy_df.loc[app]
        most_recent_year = duplicated_app['released'].astype(int).max()
        copy_df.drop(duplicated_app[duplicated_app['released'] != most_recent_year].index,inplace = True)  
delete_duplicate_rows()'''
        st.code(duplicate_code)
    elif 'casting column\'s value' in select_options:
        casting_code = '''
#to be sure that all the columns are considered with the right type
copy_df['free'] = copy_df['free'].astype(int)
copy_df['ad_supported'] = copy_df['ad_supported'].astype(float)
copy_df['in_app_purchases'] = copy_df['in_app_purchases'].astype(float)
copy_df['installs'] = copy_df['installs'].astype(int)
copy_df['category'] = copy_df['category'].astype(str)
copy_df['content_rating'] = copy_df['content_rating'].astype(str)'''
        st.code(casting_code)
    elif 'encoding category and content rating column' in select_options:
        encoding_code = '''
#i will use the class LabelEncoder
label_encoder = LabelEncoder()
encoded_data = list(label_encoder.fit_transform(copy_df['category']))
encoded_series = pd.Series(encoded_data, name = 'new label')
decoded_data = label_encoder.inverse_transform(encoded_data)
decoded_series = pd.Series(decoded_data, name = 'old label')
new_category_label = pd.DataFrame({'new_label':encoded_series,'old_label':decoded_series})
new_category_label
#i do the same for the content_rating column
label_encoder2 = LabelEncoder()
encoded_data2 = list(label_encoder2.fit_transform(copy_df['content_rating']))
encoded_series2 = pd.Series(encoded_data2, name = 'new label')
decoded_data2 = label_encoder2.inverse_transform(encoded_data2)
decoded_series2 = pd.Series(decoded_data2, name = 'old label')
new_content_rating_label = pd.DataFrame({'new_label':encoded_series2,'old_label':decoded_series2})
new_content_rating_label
#apply the modifications to our dataset
copy_df['category'] = encoded_data
copy_df['content_rating'] = encoded_data2'''
        st.code(encoding_code)
    st.write('')
    st.write('')
    st.subheader('The cleaned dataset')
    st.write('')
    cleaned_df = pd.read_csv('Project_files/cleaned_df.csv')
    st.dataframe(cleaned_df)

def plots():
    st.title('Plots')
    plots_options = ['Distribution of the categories for mean rating','Top 10 categories for mean rating','Downloads by years for the apps in the 3 categories with the highest mean rating','Apps distribution in the top 10 categories by average rating','Rating relation to price and size','Distribution of the ratings without outliers','Features correlation']
    plots_selected = st.selectbox('**Choose which plot to display**',plots_options)
    if plots_selected =='Distribution of the categories for mean rating':
        st.subheader('Distribution of the categories for mean rating')
        st.write('')
        st.image('Project_files/mean_rating_for_categories.png')
        st.write('')
        st.write('')
    elif plots_selected == 'Top 10 categories for mean rating':
        st.subheader('Top 10 categories for mean rating')
        st.write('')
        st.image('Project_files/top_10.png')
        st.write('')
        st.write('')
    elif plots_selected == 'Downloads by years for the apps in the 3 categories with the highest mean rating':
        st.subheader('Downloads by years for the apps in the 3 categories with the highest mean rating')
        st.write('')
        st.image('Project_files/dowloads_complete.png')
        st.image('Project_files/downloads_top3.png')
        st.write('')
        st.write('')
    elif plots_selected == 'Apps distribution in the top 10 categories by average rating':
        st.subheader('Apps distribution in the top 10 categories by average rating')
        st.write('')
        col1,col2 = st.columns(2)
        with col1:
            st.image('Project_files/apps_distribution.png',use_column_width=True)
        with col2:
            st.image('Project_files/n_apps3.png',use_column_width=True)
            st.write('')
            st.write('')
    elif plots_selected == 'Rating relation to price and size':
        st.subheader('Rating relation to price and size')
        st.write('')
        st.image('Project_files/scatters.png')
        st.write('')
        st.write('')
    elif plots_selected == 'Distribution of the ratings without outliers':
        st.subheader('Distribution of the ratings without outliers')
        st.write('')
        st.image('Project_files/hist.png')
        st.write('')
        st.write('')
    elif plots_selected == 'Features correlation':
        st.subheader('Features correlation')
        st.write('')
        st.image('Project_files/heatmap.png')
        st.write('')
        st.write('')
        st.subheader('Correlation in the top 3 categories by downloads')
        st.write('')
        st.image('Project_files/heatmap2.png')
        st.write('')
        st.write('')

#code to have categories' label
copy_df.index = copy_df['App Name']
index_adjusted = list(map(lambda x : x.lower().replace(' ','_'),copy_df.index))
copy_df.index = index_adjusted
copy_df.index.name = 'app_name'
copy_df.drop('App Name',axis=1,inplace = True)
new_column_labels = list(map(lambda x : x.lower().replace(' ','_'),copy_df.columns))
copy_df.columns = new_column_labels
copy_df.drop(['app_id','developer_id','minimum_installs','maximum_installs','minimum_android','currency','developer_website','developer_email','privacy_policy','last_updated','editors_choice','scraped_time'],axis=1,inplace=True)
#change the data in the category column so that it is easier to work with it
category_values_adjusted = list(map(lambda x : x.lower().replace(' ','_'),copy_df['category']))
copy_df['category'] = category_values_adjusted
list_of_the_categories = list(copy_df['category'].value_counts().index)
list_of_the_content_rating_alternatives = list(copy_df['content_rating'].value_counts().index)


def ml_model():
    st.title('Machine Learning models')
    st.write('')
    st.subheader('Model building')
    st.write('')
    model_options = ['Multiple Linear Regression','Random Forest Regression']
    model_selected = st.selectbox('**Choose which model you want to see:** ',model_options)
    if model_selected == 'Multiple Linear Regression':
        st.write('')
        st.write('')
        st.write('I built a Multi Linear Regression model to predict the rating of an application based on its features.')
        multi_linear_code = '''
        X = copy_df.drop('rating',axis=1) #dataframe without the rating feature
        Y = copy_df['rating'] #rating feature'''
        st.write('')
        st.code(multi_linear_code)
        st.write('')
        st.write('')
        st.write('I splitted the data into training and testing set and i built the model')
        code_splitted = '''
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        lr = LinearRegression()'''
        st.write('')
        st.code(code_splitted)
        st.write('')
        st.write('')
        train_code = '''
        lr.fit(X_train,Y_train)
        Y_pred = lr.predict(X_test)'''
        st.write('I trained the model and make the prediction on the test set')
        st.write('')
        st.code(train_code)
        st.write('')
        st.write('')
        st.write('Graph of the model')
        st.write('')
        st.image('Project_files/multilinear6.png')
        st.markdown('**The model isn\'t very accurate**')
        st.write('')
        st.write('')
        st.markdown('**Performance of the model:**')
        st.write('')
        st.write('R2 score: 0.0685971931487741')
        st.write('Mean Squared Error: 4.0963762472653915')
        st.write('Mean Absolute Error: 1.9035149499186017')
        st.write('Intercept: 519.8973016103193')
    elif model_selected == 'Random Forest Regression':
        st.write('')
        st.write('')
        st.write('I built a Random Forest Regression model to predict the rating of an application based on its features.')
        random_f_code = '''
X = copy_df.drop('rating',axis=1) #dataframe without the rating feature
Y = copy_df['rating'] #rating feature'''
        st.write('')
        st.code(random_f_code)
        st.write('')
        st.write('')
        st.write('I splitted the data into training and testing set and i built the model')
        st.write('')
        code_splitted_rand_f = '''
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
rf = RandomForestRegressor()'''
        st.code(code_splitted_rand_f)
        st.write('')
        st.write('')
        train_code_rand_f = '''
rf.fit(x_train,y_train)
#random forest regression prediction
y_pred = rf.predict(x_test)'''
        st.write('I trained the model and make the prediction on the test set')
        st.write('')
        st.code(train_code_rand_f)
        st.write('')
        st.write('')
        st.write('Graph of the model')
        st.image('Project_files/random_f.png')
        st.markdown('**This model is more accurate**')
        st.write('')
        st.write('')
        st.write('')
        st.markdown('**Performance of the model:**')
        st.image('Project_files/error_res.png')
        st.write('R2 score 0.9449299711465027')
        st.write('Mean Squared Error 0.243337043671237')
        st.write('Mean Absolute Error 0.2649523019985196')
    st.write('')
    st.write('')

#multilinear regression model
X = cleaned_df.drop('rating',axis=1) #dataframe without the rating feature
Y = cleaned_df['rating'] #rating feature
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
lr = LinearRegression()
lr.fit(X_train,Y_train)
Y_pred = lr.predict(X_test)
#random forest regression model
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
rf = RandomForestRegressor()
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)


def web_app_application():
    st.title('Application Rating Predictor')
    st.write('')
    st.write('')
    model_options2 = ['Multiple Linear Regression','Random Forest Regression']
    model_selected2 = st.selectbox('**Choose which model you want to use:** ',model_options2)
    if model_selected2 == 'Random Forest Regression':
        st.write('')
        st.write('')
        st.write('Chose the category of your application from the possible alternatives')
        st.write('')
        st.write('')
        col1,col2,col3 = st.columns(3)
        with col2:
            df_category_encoded = pd.read_csv('Project_files/encoded_category.csv')
            df_category_encoded_no_duplicates = df_category_encoded.drop_duplicates(subset=['old_label'])
            st.dataframe(df_category_encoded_no_duplicates)
        st.write('')
        st.write('')
        category_input = st.text_input('Write here the corresponding new_label value that refers to your app\'s category: ')
        st.write('')
        rating_count_st = st.text_input('Write here the rating count of your application: ')
        st.write('')
        installs_st = st.text_input('Write how many downloads your application has o the expected number: ')
        st.write('')
        free_st = st.text_input('Write 1 if your application is free, write 0 otherwise: ')
        st.write('')
        price_st = st.text_input('Write the price of your app: ')
        st.write('')
        size_st = st.text_input('Write the size of your app: ')
        st.write('')
        released_st = st.text_input('Write the released year of your app: ')
        st.write('')
        st.write('Chose who can rate your app among these alternatives:')
        st.write('')
        st.write('')
        col4,col5,col6 = st.columns(3)
        with col5:
            df_content_rating_encoded = pd.read_csv('Project_files/content_rating.csv')
            df_content_rating_encoded_no_duplicates = df_content_rating_encoded.drop_duplicates(subset=['old_label'])
            st.dataframe(df_content_rating_encoded_no_duplicates)
            st.write('')
            st.write('')
        content_rating_st = st.text_input('Write here the corresponding new_label value that refers to who can rate your app: ')
        st.write('')
        st.write('')
        ad_sup_st = st.text_input('Write 1 if your app is ad supported, 0 otherwise')
        st.write('')
        st.write('')
        purchase_st = st.text_input('Write 1 if your app includes purhcase in app, 0 otherwise: ')
        if st.button('PREDICT'):
            random_forest_data = {}
            random_forest_data['category'] = float(category_input)
            random_forest_data['rating_count'] = float(rating_count_st)
            random_forest_data['installs'] = float(installs_st)
            random_forest_data['free'] = float(free_st)
            random_forest_data['price'] = float(price_st)
            random_forest_data['size'] = float(size_st)
            random_forest_data['released'] = float(released_st)
            random_forest_data['content_rating'] = float(content_rating_st)
            random_forest_data['ad_supported'] = float(ad_sup_st)
            random_forest_data['in_app_purchases'] = float(purchase_st)
            values_to_predict = pd.DataFrame(random_forest_data,index=[0])
            prediction = rf.predict(values_to_predict)
            st.subheader('The predicted rating of your application is: ')
            st.write(prediction[0].round(4))
    elif model_selected2 == 'Multiple Linear Regression':
        st.write('')
        st.write('')
        st.write('Chose the category of your application from the possible alternatives')
        st.write('')
        st.write('')
        col1,col2,col3 = st.columns(3)
        with col2:
            df_category_encoded = pd.read_csv('Project_files/encoded_category.csv')
            df_category_encoded_no_duplicates = df_category_encoded.drop_duplicates(subset=['old_label'])
            st.dataframe(df_category_encoded_no_duplicates)
        st.write('')
        st.write('')
        category_input = st.text_input('Write here the corresponding new_label value that refers to your app\'s category: ')
        st.write('')
        rating_count_st = st.text_input('Write here the rating count of your application: ')
        st.write('')
        installs_st = st.text_input('Write how many downloads your application has o the expected number: ')
        st.write('')
        free_st = st.text_input('Write 1 if your application is free, write 0 otherwise: ')
        st.write('')
        price_st = st.text_input('Write the price of your app: ')
        st.write('')
        size_st = st.text_input('Write the size of your app: ')
        st.write('')
        released_st = st.text_input('Write the released year of your app: ')
        st.write('')
        st.write('Chose who can rate your app among these alternatives:')
        st.write('')
        st.write('')
        col4,col5,col6 = st.columns(3)
        with col5:
            df_content_rating_encoded = pd.read_csv('Project_files/content_rating.csv')
            df_content_rating_encoded_no_duplicates = df_content_rating_encoded.drop_duplicates(subset=['old_label'])
            st.dataframe(df_content_rating_encoded_no_duplicates)
            st.write('')
            st.write('')
        content_rating_st = st.text_input('Write here the corresponding new_label value that refers to who can rate your app: ')
        st.write('')
        st.write('')
        ad_sup_st = st.text_input('Write 1 if your app is ad supported, 0 otherwise')
        st.write('')
        st.write('')
        purchase_st = st.text_input('Write 1 if your app includes purhcase in app, 0 otherwise: ')
        st.write('')
        st.write('')
        if st.button('PREDICT'):
            random_forest_data = {}
            random_forest_data['category'] = float(category_input)
            random_forest_data['rating_count'] = float(rating_count_st)
            random_forest_data['installs'] = float(installs_st)
            random_forest_data['free'] = float(free_st)
            random_forest_data['price'] = float(price_st)
            random_forest_data['size'] = float(size_st)
            random_forest_data['released'] = float(released_st)
            random_forest_data['content_rating'] = float(content_rating_st)
            random_forest_data['ad_supported'] = float(ad_sup_st)
            random_forest_data['in_app_purchases'] = float(purchase_st)
            values_to_predict = pd.DataFrame(random_forest_data,index=[0])
            prediction = lr.predict(values_to_predict)
            st.subheader('The predicted rating of your application is: ')
            st.write(prediction[0].round(4))

page_selection = st.sidebar.radio('Select what you want to visualize: ',['Home','Data Exploration','Data Wrangling','Plots','Machine Learning Model','Application Rating Predictor'])
if page_selection == 'Home':
    home()
elif page_selection == 'Data Exploration':
    data_exploration_page()
elif page_selection == 'Data Wrangling':
    data_wrangling()
elif page_selection == 'Plots':
    plots()
elif page_selection == 'Machine Learning Model':
    ml_model()
elif page_selection == 'Application Rating Predictor':
    web_app_application()

