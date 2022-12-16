
# Predicting Rain in Australia

## Introduction:
The Australian Commonwealth Bureau of Meteorology provided a year's worth of daily weather observations gathered from the Canberra airport in Australia, which were then processed to produce this sample dataset for demonstrating data mining using Python.

Next data processing, the goal variable RainTomorrow (which indicates whether rain will fall the following day—No/Yes) and the risk variable RISK MM were created (how much rain recorded in millimetres). On the raw data, various transformations were carried out. The dataset is fairly little and only helpful for demonstrating different data science procedures in a reproducible manner. RainTomorrow is the target variable to predict. It means -- did it rain the next day, Yes or No? This column is Yes if the rain for that day was 1mm or more.

The Australian Commonwealth Bureau of Meteorology owns the uderlying dataset, which is distributed as part of the rattling package with permission.

## Importance of current study:
Develop a predictive classifier to predict the next-day rain on the target variable Rain Tomorrow. This dataset contains about 10 years of daily weather observations from many locations across Australia.

The goal variable to be predicted is RainTomorrow. Does it indicate that it rained the following day, Yes or No? If the amount of rain that day was 1 mm or greater, the column is set to Yes.

## Goal of the Project
Rain has a significant impact on a variety of plans, including crop growth for farmers, weekend getaways for families, and airline logistics. For the benefit of many stakeholders, a weather app can employ an accurate prediction model to determine if it will rain.

## Problem Type
Classfication Problem Using Machine Learning(Predict next-day rain by training classification models on the target variable RainTomorrow)

## Dataset Source & Acknowledgements
* Observations were drawn from numerous weather stations. The daily observations are available from http://www.bom.gov.au/climate/data.

* An example of latest weather observations in Canberra: http://www.bom.gov.au/climate/dwo/IDCJDW2801.latest.shtml

* Definitions adapted from http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml

* Data source: http://www.bom.gov.au/climate/dwo/ and http://www.bom.gov.au/climate/data.

* Copyright Commonwealth of Australia 2010, Bureau of Meteorology.

* This dataset contains about 10 years of daily weather observations from many locations across Australia.

## Features
* Date: The date of observation
* Location: The common name of the location of the weather station
* RainToday: Boolean: 1 if precipitation (mm) in the 24 hours to 9am
* exceeds 1mm, otherwise 0
* WindDir3pm: Direction of the wind at 3pm
* WindDir9am: Direction of the wind at 9am
* WindGustDir: The direction of the strongest wind gust in the 24 hours to midnight
* MinTemp: The minimum temperature in degrees celsius
* MaxTemp: The maximum temperature in degrees celsius
* Rainfall: The amount of rainfall recorded for the day in mm
* Evaporation: The so-called Class A pan evaporation (mm) in the 24 hours to 9am
* Sunshine: The number of hours of bright sunshine in the day.
* WindGustSpeed: The speed (km/h) of the strongest wind gust in the 24 hours to midnight
* WindSpeed9am: Wind speed (km/hr) averaged over 10 minutes prior to 9am
* WindSpeed3pm: Wind speed (km/hr) averaged over 10 minutes prior to 3am
* Humidity9am: Humidity (percent) at 9am
* Humidity3pm: Humidity (percent) at 3pm
* Pressure9am: Atmospheric pressure (hpa) reduced to mean sea level at 9am
* Pressure3pm: Atmospheric pressure (hpa) reduced to mean sea level at 3pm
* Cloud9am: Fraction of sky obscured by cloud at 9am. This is measured in "oktas", which are a unit of eigths. It records how many eigths of the sky are obscured by cloud. A 0 measure indicates completely clear sky whilst an 8 indicates that it is completely overcast
* Cloud3pm: Fraction of sky obscured by cloud (in "oktas": eighths) at 3pm. See Cload9am for a description of the values
* Temp9am: Temperature (degrees C) at 9am
* Temp3pm: Temperature (degrees C) at 3pm

## Target Variable
Chance of Rainfall Tomorrow is my target variable. The question I would like to answer is "Did it rain tomorrow?" (1 = yes, 0 = no )

## Analysis
* Extraction data from dataset and converting into pandas dataframe.
* Print the shape and size of the converted dataframe.
* Describe various features of dataset using describe function.
* Clean and tranform the dataframe uing various machine learning library like sklearn,numpy,pandas.
* Plot various plots to understand relationship between various features of dataset.
* After applying all the pre-processing steps in our dataset we have to create our machine learning column.
* Split data into training and testing data.
* Choose an approprite machine learning model using K-fold cross validation Algorithm (k-fold cross validation is a procedure used to estimate the skill of the model on new data).
* Apply hyperparameter tunning in all those model who last preformed best in K-fold cross validation algorithm.
* Identify the potential features in our dataset.
* The model that perform best in hyperparameter tunning will be chosen as our machine learning model.

## Methods and Techniques
### Data Acquisition
The Australian Commonwealth Bureau of Meteorology provided a year's worth of daily weather observations gathered from the Canberra airport in Australia, which were then processed to produce this sample dataset for demonstrating data mining using Python.​

### Cleaning and Normalization
After acquiring the dataset, we will remove null values or redundant rows, drop repeated columns, and perform outlier analysis and treatment. Data and Rainfall are both eliminated from our dataset. SkLearn's LabelEncoder is used and the RISK_MM variable is obtained to predict the rain. Many machine learning estimators need dataset standardization: if the individual features do not resemble standard normally distributed data, they may perform the necessary steps to achieve the same. Variables WindGustDir, WindDir9am, and WindDir3pm (each is split into two variables containing the cosines and sines of the wind direction angles).
The samples that had “NA” were replaced by their mode.

### Exploratory Data Analysis
During this stage, many data mining techniques are used to find hidden trends in the dataset and figure out the relationship between variables. Multiple graphs are also made to find trends in rainfall and figure out the many things that cause it. Several Python libraries have been used, such as NumPy, seaborn, matplotlib, etc. Various plots have been shown to better visualize the data. Box plots and scatterplots are used to show correlations.

### Machine Learning Models
Now we will train multiple machine learning models on our dataset and utilize validation techniques to check for overall fit. Finally, we will present the best model for rainfall prediction. K Nearest Neighbor, Decision Tree Regression, and Random Forest Regression are implemented and evaluated based on accuracies and RMSE scores.

## Implimentation
### Dataset
This dataset contains about 10 years of daily weather observations from numerous Australian weather stations, It has around 23 columns, 145460 rows.
"RainTomorrow" is the target variable to predict. It means -- did it rain the next day, Yes or No? This column is Yes if the rain for that day was 1mm or more. The number of columns is 23, while rows are 145460, there are 22 Independent Columns while 1 coloumn is dependent.

### Evaluation Metrics
The scores for accuracy and RMSE are used for evaluation. For classification problems, accuracy is a common way to measure success. It is the number of right predictions divided by the total number of predictions. RMSE is one of the most common ways to measure how accurate continuous data is. Since RMSE gives more weight to big mistakes than MAE, it should be more useful when large errors are not what you want.

### Model Training and Evaluation
Our Machine Learning algorithms are LogisticRegression, K Nearest Neighbor, and Random Forest. Below are the accuracies of each model.

* The accuracy score of the LogisticRegression is 0.83
* The accuracy score of the KNeighborsClassifier is 0.83
* The accuracy score of the RandomForestClassifier is 0.85

## Cross Validation 
We repeat the model several times with different samples in our database and the mean of that is calculated to check for accuracy percentage of our model. Models are compared using the cross validation mathod. 

* Accuracy % of the LogisticRegression 83.74 
* Accuracy % of the KNeighborsClassifier 81.78
* Accuracy % of the RandomForestClassifier() 84.16

## Result Discussion 
Data analysis helped us understand several underlying trends in rainfall over the years. Coming to the performance of the three machine learning models - Among all the trained models, Random Forest has the highest accuracy. This is because Random forest is very good in execution Speed & model performance. Random forest had an accuracy of 84.16%, followed by Logistic Regression and K-Nearest Neighbors with 83.74% and 81.78 % respectively. Based on accuracy we choose Random Forest as our choosen model.

## Conclusion
This research, it was examined whether machine learning techniques could be used to solve the issue of rainfall forecasting in the particular situation of Australia. This kind of technique has been used in earlier research with various kinds of monthly, annual, and other time period datasets in locations other than Australia. The regions of Victoria and Sydney were the study venues. 
The potential benefits of using machine learning techniques as tools to replace traditional rain forecasting techniques (they also have some advantages over classical forecasting methods, such as the possibility of estimating the reliability of the results using the Indicators, Performance Key, or the possibility of adjusting the performance of the algorithms by manipulating their input parameters, which allows them to be adapted to particular cases).

## Future Work
* This project can be further improvised by combining multiple data sets related to rainfall and weather conditions and performing in-depth analysis.
* Some statistical tests- hypothesis testing can be performed which can extract valuable insights

## Reference 
* https://rdrr.io/cran/rattle.data/man/weather.html
* https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package
* [1]. V. Rao and J. Sachdev, "A machine learning approach to classify news articles based on location", 2017 International Conference on Intelligent Sustainable Systems (ICISS), pp. 863-867, 2017.
* [2]. V. Kumar and S. Minz, "Poem classification using machine learning approach", Proceedings of the Second International Conference on Soft Computing for Problem Solving (SocProS 2012), pp. 675-682, December 28–30, 2012.
* [3]. B.Pang, L. Lee and S. Vaithyanathan, "Thumbs up? Sentiment classification using machine learning techniques" in Language Processing (EMNLP), Philadelphia, pp. 79-86, July 2002.
* [4]. B. Stehman, "Selecting and interpreting measures of thematic classification accuracy", Remote Sensing of Environment, 1997.
