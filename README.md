# Project 2 -How lifestyle factors impact Diabetes Risk


## Project Objectives
1.Determine whether lifestyle factors influence the likelihood of developing Type II diabetes.

2.Identify which lifestyle factors have the greatest impact on the risk of developing Type II diabetes.
### Team

The following team members were all original contributors to project
- Casandra Murray
- Armando Zamora
- Jessica McCarty
- Alex King
- Sarath Arja

## Project Files
The following files are the core code for the project
- ```'Resources/diabetes_data.csv''Resources/diabetes.csv'`` - data file backing the project dataframe
- ```data_outcomes.py``` - python file for basic data processing and cleaning
- ```data_jessica.ipynb``` - notebook for running the models and building the test and train data sets

## Data Sources
Data sources for this analysis come from existing publicly available sources on Kaggel https://www.kaggle.com/datasets/prosperchuks/health-dataset.  This data was pulled down as ```'Resources/diabetes_data.csv'``` in the existing project. 




## Data Processing
In this project, we extended our previous work, which was focused on women, by applying a similar approach to a new dataset. To maintain consistency with our prior analysis, we filtered the dataset to exclude male data, ensuring that our analysis remained centered exclusively on female-related information.

During preprocessing, we also evaluated the impact of removing certain features, specifically those related to stroke and heart attack. However, after testing, we found that excluding these features did not significantly improve the model's performance. As a result, we decided to retain these features in the dataset for further analysis, ensuring that our model maintained robustness and accuracy.

## Data Outcomes
All data outcomes are processed in the (file name)




## Outcomes/Conclusions
 
 One conclusion that we found is that the models ran better with Female only data compared to Male only data. The accuracy rate for male data was around 73% and 76% for female only data. 

