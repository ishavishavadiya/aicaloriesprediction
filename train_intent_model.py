from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
import pickle
from entity_extractor import extract_entities
import joblib

# Load intent model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("intent_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("intent_model")

# Load label mappings
with open("intent_labels.pkl", "rb") as f:
    label2id, id2label = pickle.load(f)

# Load encoders
le_gender = joblib.load('le_gender.pkl')
le_activity = joblib.load('le_activity.pkl')
le_goal = joblib.load('le_goal.pkl')
le_preference = joblib.load('le_preference.pkl')
le_lifestyle = joblib.load('le_lifestyle.pkl')
le_restriction = joblib.load('le_restriction.pkl')
le_health_condition = joblib.load('le_health_condition.pkl')

# --- Intent Classification ---
def classify_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_label = torch.argmax(probs, dim=1).item()
    return id2label[pred_label]

# --- Generate AI Response ---
def generate_response(text, profile):
    intent = classify_intent(text)
    entities = extract_entities(text)

    gender = le_gender.inverse_transform([profile['gender']])[0]
    goal = le_goal.inverse_transform([profile['goal']])[0]
    preference = le_preference.inverse_transform([profile['preference']])[0]

    if intent == "post_workout_food":
        return f"Since you're working on {goal}, try grilled tofu with quinoa or a banana protein smoothie."

    elif intent == "light_meal":
        return f"A light suggestion: moong dal soup with a salad. Great choice for {gender.lower()}s focusing on {goal.lower()}."

    elif intent == "meal_suggestion":
        meal = entities.get("meal_type") or "meal"
        return f"A healthy {meal} idea: oats with almond milk and fruits — works well for a {preference.lower()} diet."

    elif intent == "diet_restricted":
        diet = ", ".join(entities['diet']) if entities['diet'] else "your dietary needs"
        return f"Here's a recipe matching {diet}: quinoa salad with veggies and lemon dressing."

    elif intent == "greeting":
        return "Hi there! What kind of meal would you like help with today?"

    elif intent == "thanks":
        return "You're welcome! Stay strong and healthy 💪"

    else:
        return "Hmm, I didn’t quite catch that. Can you rephrase it?"

# --- Chat Loop ---
if __name__ == "__main__":
    print("👋 Welcome to CoachBot!")
    print("Let’s set up your profile first:")

    # Collect profile info (example, can be replaced by form/frontend input)
    gender = le_gender.transform([input("Gender (Male/Female): ")])[0]
    activity = le_activity.transform([input("Activity Level (Low/Moderate/High): ")])[0]
    goal = le_goal.transform([input("Goal (Weight Loss/Weight Maintenance/Muscle Gain): ")])[0]
    preference = le_preference.transform([input("Diet Preference (Vegetarian/Non-Vegetarian/Vegan): ")])[0]
    lifestyle = le_lifestyle.transform([input("Lifestyle (Sedentary/Active/Very Active): ")])[0]
    restriction = le_restriction.transform([input("Restriction (No Restriction/Dairy Free/Gluten Free): ")])[0]
    health = le_health_condition.transform([input("Health Condition (None/Diabetes/Heart Disease): ")])[0]

    user_profile = {
        "gender": gender,
        "activity": activity,
        "goal": goal,
        "preference": preference,
        "lifestyle": lifestyle,
        "restriction": restriction,
        "health": health
    }

    print("\nCoachBot is ready! Type your meal request. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            print("CoachBot: Bye! Stay healthy 🥗")
            break
        response = generate_response(user_input, user_profile)
        print(f"CoachBot: {response}")
