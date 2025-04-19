from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
import pickle
from entity_extractor import extract_entities

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("intent_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("intent_model")

# Load label mappings
with open("intent_labels.pkl", "rb") as f:
    label2id, id2label = pickle.load(f)

def classify_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_label = torch.argmax(probs, dim=1).item()
    return id2label[pred_label]

def generate_response(text):
    intent = classify_intent(text)
    entities = extract_entities(text)

    if intent == "post_workout_food":
        return "Try grilled tofu with quinoa or a banana protein smoothie."
    
    elif intent == "light_meal":
        return "How about moong dal soup and a salad? Light and under 400 kcal."
    
    elif intent == "meal_suggestion":
        meal = entities.get("meal_type") or "meal"
        return f"A healthy {meal} idea: oats with almond milk and fruits."
    
    elif intent == "diet_restricted":
        diet = ", ".join(entities['diet']) if entities['diet'] else "your dietary needs"
        return f"Here's a recipe matching {diet}: quinoa salad with veggies and lemon dressing."
    
    elif intent == "greeting":
        return "Hi there! What kind of meal would you like help with?"
    
    elif intent == "thanks":
        return "You're welcome! Stay healthy 🙂"
    
    else:
        return "Sorry, I didn’t get that. Can you rephrase?"

# Chat loop
if __name__ == "__main__":
    print("CoachBot is ready! Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("CoachBot: Bye! Stay healthy 🥗")
            break
        response = generate_response(user_input)
        print(f"CoachBot: {response}")
