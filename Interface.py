#Code uses streamlit to provide a simply GUI interface/
# To run execute --> streamlit run Interface.py from a command line.  It will
# open a web page to the default localhost:8505 page for execution
# Current code does not perform input checks due to time constraints.  



import streamlit as st
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import data_models
import data_outcomes as data


# def collect_user_data():
    # questions = [
        # "How old are you?",
        # "Have you had a cholesterol check in the last 5 years?"
        # "Do you have high cholesterol?",
        # "Do you consume alcohol often?",
        # "What is you BMI?"
        # "Have you smoked at least 100 cigarettes in your entire life??",
        # "Do you have coronary heart disease (CHD) or myocardial infarction (MI)?",
        # "Have you been physical activity in past 30 days?",
        # "Consume Vegetables 1 or more times per day?",
        # "Consume Fruit 1 or more times per day?"
    # ]


    #  for question in questions:
        # while True:
            # answer = input(f"{question} (yes or no): ").strip().lower()
            # if answer in ['yes', 'no']:
                # responses.append(1 if answer == 'yes' else 0)
                # break
            # else:
                # print("Please enter 'yes' or 'no'.")

    # return responses

# def predict_diabetes(classifier, user_responses):
    # prediction = classifier.predict([user_responses])
    # return "Likely to have diabetes" if prediction == 1 else "Unlikely to have diabetes"

# user_responses = collect_user_data()
# diabetes_prediction = predict_diabetes(model, user_responses)

# print(diabetes_prediction)



st.title("Diabetes Analysis Page")

def getAgeCategory(num):
    print(f'age {num}')
    if (num ==''):
        num = 18
    else:
        num = int(num)
    print(f'age2 {num}')
    num_cat = 1
    if (num >=25 and num <=29):
        num_cat = 2
    if (num >=30 and num <=34):
        num_cat = 3
    if (num >=35 and num <=39):
        num_cat = 4
    if (num >=40 and num <=44):
        num_cat = 5
    if (num >=45 and num <=49):
        num_cat = 6
    if (num >= 50 and num <=54):
        num_cat = 7
    if (num >=55 and num <=59):
        num_cat = 8
    if (num >=60 and num <=64):
        num_cat = 9
    if (num >=65 and num <=69):
        num_cat = 10
    if (num >=70 and num <=74):
        num_cat = 11
    if (num >=75 and num <=79):
        num_cat = 12
    if (num >=80):
        num_cat = 13
        
    return num_cat

def getBool(x):
    bln = 0
    if(x == 'Yes'):
        bln = 1
    else:
        bln = 0
    return bln
        
def getSex(x):
    sex = 0
    if(x == "Male"):
        sex = 1
    return sex

def getHeavyDrinker(drinks, sex):
    if(drinks == ''):
        drinks = 0
    else:
        drinks = int(drinks)
    heavy = 0
    if(sex == 0):
        if (drinks >= 7):
            heavy = 1
    else:
        if(drinks >= 14):
            heavy = 1
    return heavy

def getGeneralHealth(hlth):
    h_val = 1
    if hlth == 'Excellent':
        h_val = 1
    if hlth == 'Very Good':
        h_val = 2
    if hlth == 'Good':
        h_val = 3
    if hlth == 'Fair':
        h_val = 4
    if hlth == 'Poor':
        h_val = 5
    return h_val
        

age = 18
age = getAgeCategory(st.text_input("How old are you?:"))

sex = 0
sex = getSex(st.selectbox("Are you biologically female or male?", ('Female', 'Male')))
highChol = getBool(st.selectbox("Do you have High Cholesterol?", ('Yes', 'No')))
cholCheck = getBool(st.selectbox("Have you had a cholesterol check in the last 5 years??", ('Yes', 'No')))
bmi = st.text_input("What is your BMI?")
smoker = getBool(st.selectbox("Have you smoked at least 100 cigarettes in your entire life?", ('Yes', 'No')))
#heartdisease = getBool(st.selectbox("Do you have coronary heart disease (CHD) or myocardial infarction (MI)?", ("Yes", "No")))
physactivity = getBool(st.selectbox("Have you been physical activity in past 30 days?", ("Yes", 'No')))
fruites = getBool(st.selectbox("Consume Fruites 1 or more times per day?", ('Yes', 'No')))
veggies = getBool(st.selectbox("Consume Vegetables 1 or more times per day?", ("Yes", "No")))
alcohol = getHeavyDrinker(st.text_input("Do you consume alcohol often (enter drinks per week?"), sex)
genhealth = getGeneralHealth(st.selectbox("How would you say your health is?", ('Execellent', 'Very Good', 'Good', 'Fair', 'Poor')))
#menthealth = st.text_input('Days of poor mental health over the month',)
physhlth = st.text_input('How many day did you have a physical illness or injury over the past 30 days')
#diffwalk = getBool(st.selectbox('Do you have serious difficulty walking or climbing stairs', ('Yes', 'No')))
#stroke = getBool(st.selectbox('have you every had a stroke?', ('Yes', 'No')))
Highbp = getBool(st.selectbox('Do you have high bp?', ('Yes', "No")))
name = st.text_input("Enter your name:")

if st.button("Submit"):

    d = {
        'Age': age,
        'Sex': sex,
        'HighChol': highChol,
        'CholCheck': cholCheck,
        'BMI': bmi,
        'Smoker': smoker,
        #'HeartDiseaseorAttack': heartdisease,
        'PhysActivity': physactivity,
        'Fruits':fruites,
        'Veggies':veggies,
        'HvyAlcoholConsump':alcohol,
        'GenHlth': genhealth,
        #'MentHlth': menthealth,
        'PhysHlth': physhlth,
        #'DiffWalk': diffwalk,
        #'Stroke': stroke,
        'HighBP':Highbp
    }

    df = pd.DataFrame(d, index=[0])
    df1 = data.getDiabetesBehaviorDataframe()

    gradientboost = GradientBoostingClassifier(random_state=42, n_estimators=200)
    X_train, X_test, y_train, y_test, pipeline = data_models.model_generator(df1, gradientboost)
    testing_predictions = pipeline.predict(df)

    st.write('Gradient Boost Resulsts')
    st.write(testing_predictions)

    rfc = RandomForestClassifier(random_state=1, n_estimators=100, max_depth=10, class_weight='balanced')
    X_train, X_test, y_train, y_test, pipeline = data_models.retrieve_model(df1, rfc)
    testing_predictions = pipeline.predict(df)

    st.write('Random Forest Resulsts')
    st.write(testing_predictions)
    
    st.write(f"Hello, {name}!")
    st.write(df)