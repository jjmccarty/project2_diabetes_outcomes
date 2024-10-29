def collect_user_data():
    questions = [
        "How old are you?",
        
        "Do you have high cholesterol?",
        "Are you physically active?",
        "Do you consume alcohol often?",
        "Are you a smoker?",
        "Do you eat vegetables?",
        "Do you eat fruit?"
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