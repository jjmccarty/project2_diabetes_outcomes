def collect_user_data():
    questions = [
        "How old are you?",
        "Have you had a cholesterol check in the last 5 years?"
        "Do you have high cholesterol?",
        "Do you consume alcohol often?",
        "What is you BMI?"
        "Have you smoked at least 100 cigarettes in your entire life??",
        "Do you have coronary heart disease (CHD) or myocardial infarction (MI)?",
        "Have you been physical activity in past 30 days?",
        "Consume Vegetables 1 or more times per day?",
        "Consume Fruit 1 or more times per day?"
    ]


     for question in questions:
        while True:
            answer = input(f"{question} (yes or no): ").strip().lower()
            if answer in ['yes', 'no']:
                responses.append(1 if answer == 'yes' else 0)
                break
            else:
                print("Please enter 'yes' or 'no'.")

    return responses

def predict_diabetes(classifier, user_responses):
    prediction = classifier.predict([user_responses])
    return "Likely to have diabetes" if prediction == 1 else "Unlikely to have diabetes"

user_responses = collect_user_data()
diabetes_prediction = predict_diabetes(model, user_responses)

print(diabetes_prediction)