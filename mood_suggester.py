from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Training data
training_sentences = [
    "I'm feeling stressed and want something sweet",
    "I'm bored and craving spicy food",
    "I'm tired and need something crunchy",
    "I'm anxious and want something salty",
    "I'm lazy and want chocolate",
    "I'm happy and want something cold",
    "I'm sad and want something warm",
    "I'm energetic and want something fresh",
    "I'm angry and want something chewy",
    "I'm excited and want something fizzy",
    "I'm relaxed and want something creamy",
    "I'm overwhelmed and need comfort food",
    "I'm sleepy and want a light snack",
    "I'm focused and want something with protein",
    "I'm homesick and want something nostalgic",
    "I'm lonely and want something indulgent",
    "I'm motivated and want a healthy treat",
    "I'm confused and want something familiar",
    "I'm adventurous and want something exotic",
    "I'm annoyed and want something crunchy",
    "I'm cold and want something hot",
    "I'm hot and want something frozen",
    "I'm proud and want a treat",
    "I'm disappointed and want something comforting",
    "I'm grateful and want something wholesome",
]

food_suggestions = [
    "Banana with peanut butter or yogurt with honey",
    "Roasted chickpeas or spicy hummus with veggies",
    "Air-popped popcorn or carrot sticks with hummus",
    "Whole grain crackers with cheese or trail mix",
    "Dark chocolate protein bar or smoothie",
    "Frozen berries or fruit smoothie",
    "Oatmeal with cinnamon or soup",
    "Green smoothie or cucumber salad",
    "Beef jerky or granola bar",
    "Sparkling water with citrus or kombucha",
    "Pudding or mashed potatoes",
    "Mac and cheese or grilled cheese sandwich",
    "Rice cakes with almond butter or a piece of fruit",
    "Boiled eggs or Greek yogurt",
    "Apple pie or mashed potatoes",
    "Chocolate lava cake or rich brownies",
    "Avocado toast or fruit bowl",
    "Peanut butter sandwich or cereal",
    "Sushi or Thai curry",
    "Nachos or veggie chips",
    "Frozen grapes or ice cream",
    "Cupcake or fancy pastry",
    "Mashed potatoes or warm brownies",
    "Quinoa salad or roasted veggies",
    "Quinoa salad or roasted veggies",
]

# Model setup
vectorizer = CountVectorizer()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(training_sentences, food_suggestions, test_size=0.2, random_state=42)

# Transform the training and test data
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train the model
clf = MultinomialNB()
clf.fit(X_train_vect, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_vect)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.3f}")

# Single suggestion
def suggest_food(user_text):
    """
    Returns the most likely food suggestion for the given mood input.
    """
    X_test = vectorizer.transform([user_text])
    prediction = clf.predict(X_test)
    return prediction[0]

# Top N suggestions with confidence
def suggest_top_n_foods(user_text, n=3):
    """
    Returns top N food suggestions with confidence scores.
    """
    X_test = vectorizer.transform([user_text])
    probs = clf.predict_proba(X_test)[0]
    top_indices = probs.argsort()[-n:][::-1]
    return [(clf.classes_[i], round(probs[i], 3)) for i in top_indices]

# Example usage
if __name__ == "__main__":
    user_input = input("How are you feeling and what are you craving? ")
    print("Top suggestion:", suggest_food(user_input))
    print("Other suggestions:", suggest_top_n_foods(user_input))
