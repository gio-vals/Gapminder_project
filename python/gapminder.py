## Gapminder Project ##
# importing libraries for data analysis

import pandas as pd
import numpy as np
# importing libraries for data visualisation

import matplotlib.pyplot as plt
import seaborn as sns
# importing library for image processing

import imageio
 
# continents.csv
continents = pd.read_csv("../data/continents.csv", sep =";") 
continents
continents.info()
# checking the unique continents in the dataframe

continents['continent'].unique()
# checking for null

continents.isnull().sum()
continents['continent'].value_counts()
continents["continent"].value_counts().plot(kind="bar", color='teal');
 
# population.csv
# reading the population.csv into a dataframe

population = pd.read_csv("../data/population.csv")
population.head()
population.tail(10)
# inspecting the dataframe
# inspecting the datatypes of the columns

population.dtypes
# inspecting the dataframe with .info

population.info()
# The describe method will give statistical insights for numeric columns

population.describe()
# Renaming the first column to 'country'

population.rename(columns={'Total population':"country"}, inplace=True)
# check the null values

population.isnull()  #the question is: "Is this value a null? False= not a null / True = yes, null value"
# summarize empty values

population.isnull().sum()
# checking for duplicated rows!
population.duplicated().sum()

# if we had duplicates, we should use the method drop_duplicates()
# removing missing data by dropping the observations with missing data

population.dropna(axis=0, inplace=True)
population.isna().sum()
population.info()
# histogram on the 'year' column using matplotlib

plt.hist(population['year'], bins=20);
# histogram on the 'population' column using matplotlib
# use logarithmic transformation to minimize differences (homogenize the data) in datasets with huge differences

plt.hist(population['population'], bins=200, log=True);
# chacking for the lowest population

population['population'].min()
# and identifying it belongs to Pitcairn Islands in the Pacific Ocean for the years 2005-2010

population[population['population']==50]
# filtering the dataset that it begins with the year 1950 with a boolean mask

p_mask1950 = population["year"]>1949
# and saving it as another dataframe

population1950 = population[p_mask1950]
population1950.reset_index(drop=True, inplace=True)
population1950
# making the data persistant by saving the dataset as a .csv file in the data folder

population1950.to_csv('../data/population1950.csv')
 
# Fertility rate csv dataset
fertility = pd.read_csv('../data/fertility_rate.csv')
fertility
# renaming the first column to 'country'

fertility = fertility.rename(columns={'Total fertility rate':"country"})
fertility.info()
fertility.isna().sum()
fertility.dropna(axis=0, inplace=True)
fertility.isna().sum()
f_mask1950 = fertility["year"]>1949
fertility1950 = fertility[f_mask1950]
fertility1950.reset_index(drop=True, inplace=True)
fertility1950.head()
fertility1950.to_csv('../data/fertility1950.csv')
 
# Life expectancy.xls

# it is an excel file. In case is not installed: !pip install xlrd
expectancy = pd.read_excel('../data/life_expectancy.xls')
expectancy
expectancy.rename(columns={'Life expectancy':"country"}, inplace=True)
expectancy.head()
expectancy.info()
expectancy.isna().sum()
expectancy.dropna(axis=0, inplace=True)
expectancy.isna().sum()
e_mask1950 = expectancy["year"]>1949
expectancy1950 = expectancy[e_mask1950]
expectancy1950.reset_index(drop=True, inplace=True)
expectancy1950.info()
expectancy1950.head()
expectancy1950.to_csv('../data/expectancy1950.csv')
# exploring the expectancy1950 dataset

expectancy1950.describe()
# what is the total mean life expectancy?

expectancy1950['life expectancy'].mean()
# what is the total median life expectancy?

expectancy1950['life expectancy'].median()
# what is the total standard deviation of life expectancy?

expectancy1950['life expectancy'].std()
 
# what is the mean vs median life expectancy for the years 1980 and 2000?
# mean life expectancy for the year 2000

expectancy1950[expectancy1950['year']==2000].mean(numeric_only=True)
# mean life expectancy for the year 1980

expectancy1950[expectancy1950['year']==1980].mean(numeric_only=True)
# median life expectancy for the year 2000

expectancy1950[expectancy1950['year']==2000].median(numeric_only=True)
# median life expectancy for the year 1980

expectancy1950[expectancy1950['year']==1980].median(numeric_only=True)
 
# Which countries have the highest and lowest life expectancy worldwide and in which year?
expectancy1950['life expectancy'].max()
expectancy1950['life expectancy'].min()
expectancy1950.describe()
# which country has the max life expectancy?

expectancy1950[expectancy1950["life expectancy"]==84.8]
# with .loc

expectancy1950.loc[expectancy1950["life expectancy"]==84.8]
# which country has the min life expectancy?

expectancy1950[expectancy1950["life expectancy"]==13.2]
# with .loc

expectancy1950.loc[expectancy1950["life expectancy"]==13.2]
# find the 90% quantile of the life expectancy

expectancy1950['life expectancy'].quantile([0.1, 0.9])

# the 90% quantile of the life expectancy is 76.4
 
# Merging 4 dataframes

# continents
# population1950
# fertility1950
# expectancy1950
# merging continents with expectancy1950

cont_expect = continents.merge(expectancy1950)
cont_expect.head()
cont_expect.isnull().sum()
# merging the other two dfs (population1950 and fertility1950)
popul_fert = population1950.merge(fertility1950)
popul_fert.head()
popul_fert.isnull().sum()
# merging all in one dataframe
gapminder_total = cont_expect.merge(popul_fert)
gapminder_total.head()
gapminder_total.info()
# saving the complete dataframe to csv

gapminder_total.to_csv('../data/gapminder_total.csv')
gapminder_total.describe()
# How often does each continent appear in the data set?

gapminder_total['continent'].value_counts()
gapminder_total['continent'].value_counts(normalize=True)
# Which continent has the lowest average fertility rate overall?

agg_dict = {'fertility': ['mean']}

gapminder_total.groupby('continent').agg(agg_dict)

# Europe by far
continent_fertility = gapminder_total.groupby('continent')['fertility'].mean()
continent_fertility
continent_fertility.idxmin()
# What was the average life expectancy in Europe in 2015? 

agg_dict = {'life expectancy': ['mean']}

gapminder_total[gapminder_total['year']==2015].groupby('continent').agg(agg_dict)

# 78.75 years
# How many countries does each continent have in the dataset?

gapminder_total.groupby('continent')['country'].nunique()
# What is the average population of a European country in 1960 compare to 2015?
# Hint: once again filter for the year in question and do each year separately to compare

agg_dict = {'continent': ['value_counts']}

gapminder_total[gapminder_total['year']==1960].groupby('continent').agg(agg_dict)

# double checking that we have the same number of European countries in 1976 (as in 2015 = 40)
gapminder_total[gapminder_total['year']==2015].groupby('continent').agg(agg_dict)

# 40 countries in Europe
agg_dict = {'population': ['mean']}
gapminder_total[gapminder_total['year']==1960][gapminder_total['continent']=='Europe'].groupby('continent').agg(agg_dict)
gapminder_total[gapminder_total['year']==2015][gapminder_total['continent']=='Europe'].groupby('continent').agg(agg_dict)
# for all the continents (not just Europe) in 1960

gapminder_total[gapminder_total['year']==1960].groupby('continent').agg(agg_dict)
# for all the continents (not just Europe) in 2015

gapminder_total[gapminder_total['year']==2015].groupby('continent').agg(agg_dict)
# What is the highest population a continent ever had?

agg_dict = {'population': ['max']}

populations_per_year = gapminder_total.groupby(['continent', 'year']).agg(agg_dict)

populations_per_year
populations_per_year.idxmax()
 
# making masks for the all the continents and saving them as dfs


Europe = gapminder_total[gapminder_total['continent']=='Europe']
Asia = gapminder_total[gapminder_total['continent']=='Asia']
Africa = gapminder_total[gapminder_total['continent']=='Africa']
NorthAmerica = gapminder_total[gapminder_total['continent']=='North America']
SouthAmerica = gapminder_total[gapminder_total['continent']=='South America']
Oceania = gapminder_total[gapminder_total['continent']=='Australia and Oceania']
# creating histograms for the distribution of populations in the different continents for all the years together

listofdfs = [Europe['population'], Asia['population'], Africa['population'], 
             NorthAmerica['population'], SouthAmerica['population'], Oceania['population']] 

plt.hist(x = listofdfs, label =['Europe', 'Asia', 'Africa', 'North America', 'South America', 'Oceania'], log=True, bins = 6, rwidth=1)

plt.legend()
plt.xlabel('Population')
plt.ylabel('Count')
plt.title("Population Distribution per Contintent (1950-2015)");

# rwidth: Set Width for Bars in Bar Plot
gapminder_total
gapminder_total.tail()
gapminder_total.groupby('continent')['country'].nunique()
# Boxplots for the distribution of populations in every country per continent 
# (figsize adjusted to the number of countries)
sns.set_style("darkgrid")

# Africa (49 countries)
plt.figure(figsize = (14,24))
sns.boxplot(y = 'country', x = 'population', data=Africa).set(title='Distribution of African populations');
# Asia (39 countries)
plt.figure(figsize = (14,20))
sns.boxplot(y = 'country', x = 'population', data=Asia).set(title='Distribution of Asian populations');
# Europe (40 countries)
plt.figure(figsize = (14,20))
sns.boxplot(y = 'country', x = 'population', data=Europe).set(title='Distribution of European populations');
# North America (19 countries)
plt.figure(figsize = (14,10))
sns.boxplot(y = 'country', x = 'population', data=NorthAmerica).set(title='Distribution of North American populations');
# South America (12 countries)
plt.figure(figsize = (14,6))
sns.boxplot(y = 'country', x = 'population', data=SouthAmerica).set(title='Distribution of South American populations');
# Oceania (9 countries)
plt.figure(figsize = (14,5))
sns.boxplot(y = 'country', x = 'population', data=Oceania).set(title='Distribution of Oceanian populations');
Asia
# Borplots with the mean fertility for every country per continent 
# (figsize adjusted to the number of countries)

# Africa 

plt.figure(figsize = (14,8))
sns.barplot(x="country", y ="fertility", palette="tab10", data=Africa).set(title='Fertility rate in African countries (1950-2015)');
plt.xticks(rotation=90);
gapminder_total.groupby('continent')['country'].nunique()
# Asia (39 countries)

plt.figure(figsize = (12,8))
sns.barplot(x ="country", y="fertility", palette="tab10", data=Asia).set(title='Fertility rate in Asian countries (1950-2015)');
plt.xticks(rotation=90);
# Europe 

plt.figure(figsize = (12,8))
sns.barplot(x ="country", y="fertility", palette="tab10", data=Europe).set(title='Fertility rate in European countries (1950-2015)');
plt.xticks(rotation=90);
plt.figure(figsize = (6,8))
sns.barplot(x ="country", y="fertility", palette="tab10", data=NorthAmerica).set(title='Fertility rate in North American countries (1950-2015)');
plt.xticks(rotation=90);
plt.figure(figsize = (5,8))
sns.barplot(x ="country", y="fertility", palette="tab10", data=SouthAmerica).set(title='Fertility rate in South American countries (1950-2015)');
plt.xticks(rotation=90);
plt.figure(figsize = (4,8))
sns.barplot(x ="country", y="fertility", palette="tab10", data=Oceania).set(title='Fertility rate in Oceanian countries (1950-2015)');
plt.xticks(rotation=90);
# Life expectancy in every country per continent (bar plots)
# Africa

plt.figure(figsize = (14,8))
sns.barplot(x ="country", y="life expectancy", palette="tab10", data=Africa).set(title='Life expectancy in African countries (1950-2015)');
plt.xticks(rotation=90);
# Asia

plt.figure(figsize = (12,8))
sns.barplot(x ="country", y="life expectancy", palette="tab10", data=Asia).set(title='Life expectancy in Asian countries (1950-2015)');
plt.xticks(rotation=90);
# Europe

plt.figure(figsize = (12,8))
sns.barplot(x ="country", y="life expectancy", palette="tab10", data=Europe).set(title='Life expectancy in European countries (1950-2015)');
plt.xticks(rotation=90);
# North America

plt.figure(figsize = (6,8))
sns.barplot(x ="country", y="life expectancy", palette="tab10", data=NorthAmerica).set(title='Life expectancy in North American countries (1950-2015)');
plt.xticks(rotation=90);
# South America

plt.figure(figsize = (5,8))
sns.barplot(x ="country", y="life expectancy", palette="tab10", data=SouthAmerica).set(title='Life expectancy in South American countries (1950-2015)');
plt.xticks(rotation=90);8
# Oceania 

plt.figure(figsize = (4,8))
sns.barplot(x ="country", y="life expectancy", palette="tab10", data=Oceania).set(title='Life expectancy in Oceanian countries (1950-2015)');
plt.xticks(rotation=90);
 
gapminder_total
# Creating a subset of data for one year eg. 1950

df_subset = gapminder_total.loc[gapminder_total['year'] == 1950]
df_subset
# Step 7
# Setting the axes to display the origin also gives us 
# a more complete picture of what is going on.


plt.figure(figsize=(8, 6))
plt.title('Life Expectancy vs Fertility Rate')
plt.ylabel('Fertility Rate')
plt.xlabel('Life Expectancy')
plt.axis([0, 85, 0, 9.5])

g = sns.scatterplot(
                    x='life expectancy', 
                    y='fertility', 
                    data = df_subset, 
                    hue= 'continent',
                    size = 'population',
                    sizes = (40, 250),
                    s=120,
                    palette="tab10",
                    alpha = 0.7
                    );

h,l=g.get_legend_handles_labels()

plt.legend(h[0:7],l[0:7],bbox_to_anchor=(1.02, 0.8),loc=2, 
frameon= False, fontsize=10);
# saving the plot for one year

sns.set_style("darkgrid")

plt.subplots(figsize=(8, 6))
plt.title('Life Expectancy vs Fertility Rate)')
plt.ylabel('Fertility Rate (children)')
plt.xlabel('Life Expectancy (years)')
plt.axis([0, 85, 0, 9.5])

g = sns.scatterplot(
                    x='life expectancy', 
                    y='fertility', 
                    data = df_subset, 
                    hue= 'continent',
                    size = 'population',
                    sizes = (40, 250),
                    s=120,
                    palette="tab10",
                    alpha = 0.7
                    );

h,l=g.get_legend_handles_labels()

plt.legend(h[0:7],l[0:7],bbox_to_anchor=(1.02, 0.8),loc=2, 
frameon= False, fontsize=10)

plt.annotate('Brazil',(df_subset[df_subset['country']=="Brazil"]['life expectancy'],    
                      df_subset[df_subset['country']=="Brazil"]['fertility']+0.1));  
plt.annotate('New Zealand',(df_subset[df_subset['country']=="New Zealand"]['life expectancy'],    
                      df_subset[df_subset['country']=="New Zealand"]['fertility']+0.1)); 
plt.annotate('Botswana',(df_subset[df_subset['country']=="Botswana"]['life expectancy'],    
                      df_subset[df_subset['country']=="Botswana"]['fertility']+0.1)); 
plt.annotate('Honduras',(df_subset[df_subset['country']=="Honduras"]['life expectancy'],    
                      df_subset[df_subset['country']=="Honduras"]['fertility']+0.1)); 
plt.annotate('Serbia',(df_subset[df_subset['country']=="Serbia"]['life expectancy'],    
                      df_subset[df_subset['country']=="Serbia"]['fertility']+0.1)); 
plt.annotate('Iran',(df_subset[df_subset['country']=="Iran"]['life expectancy'],    
                      df_subset[df_subset['country']=="Iran"]['fertility']+0.1)); 


PATH = (f'../images')
filename = (f'plot_image1950.png')
plt.savefig(f'{PATH}/{filename}', bbox_inches="tight")
plt.close()
 
# with a for loop, creating graphs for each year and saving them in the folder 'images'

PATH =(f'../images')
for year in range(1950, 2016):
    df_subset = gapminder_total[gapminder_total['year']==year]
    #MY PLOT
    sns.set_style("darkgrid")

    plt.subplots(figsize=(8, 6))
    plt.title(f'Life Expectancy vs Fertility Rate ({year})')
    plt.ylabel('Fertility Rate (children)')
    plt.xlabel('Life Expectancy (years)')
    plt.axis([0, 85, 0, 9.5])

    g = sns.scatterplot(
                    x='life expectancy', 
                    y='fertility', 
                    data = df_subset, 
                    hue= 'continent',
                    size = 'population',
                    sizes = (40, 250),
                    s=120,
                    palette="tab10",
                    alpha = 0.7
                    );

    h,l=g.get_legend_handles_labels()

    plt.legend(h[0:7],l[0:7],bbox_to_anchor=(1.02, 0.8),loc=2, 
    frameon= False, fontsize=10)

    plt.annotate('Brazil',(df_subset[df_subset['country']=="Brazil"]['life expectancy'],    
                      df_subset[df_subset['country']=="Brazil"]['fertility']+0.1));  
    plt.annotate('New Zealand',(df_subset[df_subset['country']=="New Zealand"]['life expectancy'],    
                      df_subset[df_subset['country']=="New Zealand"]['fertility']+0.1)); 
    plt.annotate('Botswana',(df_subset[df_subset['country']=="Botswana"]['life expectancy'],    
                      df_subset[df_subset['country']=="Botswana"]['fertility']+0.1)); 
    plt.annotate('Honduras',(df_subset[df_subset['country']=="Honduras"]['life expectancy'],    
                      df_subset[df_subset['country']=="Honduras"]['fertility']+0.1)); 
    plt.annotate('Bulgaria',(df_subset[df_subset['country']=="Bulgaria"]['life expectancy'],    
                      df_subset[df_subset['country']=="Bulgaria"]['fertility']+0.1)); 
    plt.annotate('Iran',(df_subset[df_subset['country']=="Iran"]['life expectancy'],    
                      df_subset[df_subset['country']=="Iran"]['fertility']+0.1)); 


    
    filename = f'plot_{year}.png'
    plt.savefig(f'{PATH}/{filename}', bbox_inches="tight")
    plt.close()
# making an animated gif from the images saved in 'images' and saving the gif in the folder 'gifs'

PATH =(f'../images')
PATH2 =(f'../gifs')

images = []

for year in range(1950, 2016):
    filename = f'{PATH}/plot_{year}.png'
    images.append(imageio.imread(f'{filename}'))

imageio.mimsave(f'{PATH2}/the_gif_dur02.gif', images, duration=0.2)
 
### Making a gif with the European populations over the years. Step1

PATH =(f'../images')
for year in range(1960, 2016):


    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize = (10,12))
    plt.title(f'Population of European countries ({year})')
    plt.ylabel('Countries')
    plt.xlabel('Population')
    ax.set(xlim=(0, 90000000))

    sns.barplot(x ="population", y="country", errorbar= None, estimator=np.max, ax=ax, data=Europe[Europe['year']==year]);    
    
    filename = f'plot_Europe_{year}.png'
    plt.savefig(f'{PATH}/{filename}', bbox_inches="tight")
    plt.close()
### Making a gif with the European populations over the years. Step2

PATH =(f'../images')
PATH2 =(f'../gifs')

images = []

for year in range(1960, 2016):
    filename = f'{PATH}/plot_Europe_{year}.png'
    images.append(imageio.imread(f'{filename}'))

imageio.mimsave(f'{PATH2}/the_European_populations_1960-2015.gif', images, duration=0.2)
### Making a gif with the Asian populations over the years. Step1
PATH =(f'../images')
for year in range(1960, 2016):


    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize = (10,12))
    plt.title(f'Population of Asian countries ({year})')
    plt.ylabel('Countries')
    plt.xlabel('Population')
    ax.set(xlim=(0, 1500000000))

    sns.barplot(x ="population", y="country", errorbar= None, estimator=np.max, ax=ax, data=Asia[Asia['year']==year]);    
    
    filename = f'plot_Asia_{year}.png'
    plt.savefig(f'{PATH}/{filename}', bbox_inches="tight")
    plt.close()
### Making a gif with the Asian populations over the years. Step2

PATH =(f'../images')
PATH2 =(f'../gifs')

images = []

for year in range(1960, 2016):
    filename = f'{PATH}/plot_Asia_{year}.png'
    images.append(imageio.imread(f'{filename}'))

imageio.mimsave(f'{PATH2}/the_Asian_populations_1960-2015.gif', images, duration=0.2)
### Making a gif with the African populations over the years. Step1
PATH =(f'../images')
for year in range(1960, 2016):


    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize = (10,12))
    plt.title(f'Population of African countries ({year})')
    plt.ylabel('Countries')
    plt.xlabel('Population')
    ax.set(xlim=(0, 200000000))

    sns.barplot(x ="population", y="country", errorbar= None, estimator=np.max, ax=ax, data=Africa[Africa['year']==year]);    
    
    filename = f'plot_Africa_{year}.png'
    plt.savefig(f'{PATH}/{filename}', bbox_inches="tight")
    plt.close()
### Making a gif with the African populations over the years. Step2

PATH =(f'../images')
PATH2 =(f'../gifs')

images = []

for year in range(1960, 2016):
    filename = f'{PATH}/plot_Africa_{year}.png'
    images.append(imageio.imread(f'{filename}'))

imageio.mimsave(f'{PATH2}/the_African_populations_1960-2015.gif', images, duration=0.2)