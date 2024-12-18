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
- ```'Resources/diabetes_data.csv`` - data file backing the project dataframe
- ```data_outcomes.py``` - python file that loads and cleans the data 
- ```diabetes_data.ipynb``` - notebook for running the models and building the test and train data sets
- ```data_models.py``` - python file that is used for running the models 
- ```interface.py```- python file that is used for our UI and prediction of new data.

## Data Sources
Data sources for this analysis come from existing publicly available sources on Kaggel https://www.kaggle.com/datasets/prosperchuks/health-dataset.  This data was pulled down as ```'Resources/diabetes_data.csv'``` in the existing project. 


## Data Processing
In this project, we extended our previous work, which was focused on women, by applying a similar approach to a new dataset. To maintain consistency with our prior analysis, we filtered the dataset to exclude male data, ensuring that our analysis remained centered exclusively on female-related information.

During preprocessing, we also evaluated the impact of removing certain features, specifically those related to stroke and heart attack. However, after testing, we found that excluding these features did not significantly improve the model's performance. As a result, we decided to retain these features in the dataset for further analysis, ensuring that our model maintained robustness and accuracy.


## Data Outcomes
All data outcomes are processed in the (```data_outcome.py```)

In our project, we explored various machine learning models to identify the best fit for our data. After rigorous evaluation, Random Forest and Gradient Boosting emerged as the top-performing models. Both models highlighted three key lifestyle factors that significantly impact outcomes: Body Mass Index (BMI), heavy alcohol consumption, and age. These factors consistently influenced predictions, indicating their importance in understanding the patterns within our dataset.


## Outcomes/Conclusions
One conclusion that we found is that the models ran better with Female only data compared to Male only data. The accuracy rate for male data was around 73% and 76% for female only data. 

Our analysis found that while all lifestyle factors affect diabetes risk, some—such as healthy eating, smoking, cholesterol, and physical activity—had a smaller direct impact than anticipated. However, these factors may still indirectly influence primary factors like BMI, illustrating the complex interplay of lifestyle choices on health outcomes. This underscores the need for a holistic approach to lifestyle improvements for effective diabetes prevention.