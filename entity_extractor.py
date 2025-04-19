import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = {
        "meal_type": None,
        "nutrient_focus": [],
        "diet": [],
        "calories": None
    }

    for token in doc:
        # Meal type
        if token.text.lower() in ["breakfast", "lunch", "dinner"]:
            entities["meal_type"] = token.text.lower()

        # Nutrient focus
        if token.text.lower() in ["high", "low"]:
            for child in token.children:
                if child.text.lower() in ["protein", "carb", "fat"]:
                    entities["nutrient_focus"].append(f"{token.text.lower()} {child.text.lower()}")

        # Diet type
        if token.text.lower() in ["vegan", "vegetarian", "gluten-free", "keto"]:
            entities["diet"].append(token.text.lower())

    # Calorie detection (e.g., "under 500 calories", "around 300 cal")
    for ent in doc.ents:
        if ent.label_ == "CARDINAL" and "cal" in text.lower():
            try:
                entities["calories"] = int(ent.text)
                break
            except ValueError:
                continue

    return entities
