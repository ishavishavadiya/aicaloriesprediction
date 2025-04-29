from flask import Flask, jsonify, render_template, request
import numpy as np
import joblib
from mood_suggester import suggest_food
from werkzeug.utils import secure_filename
import os
from model_utils import predict_food

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load encoders
le_gender = joblib.load('le_gender.pkl')
le_activity = joblib.load('le_activity.pkl')
le_goal = joblib.load('le_goal.pkl')
le_preference = joblib.load('le_preference.pkl')
le_lifestyle = joblib.load('le_lifestyle.pkl')
le_restriction = joblib.load('le_restriction.pkl')
le_health_condition = joblib.load('le_health_condition.pkl')

# Load model
model = joblib.load('calorie_predictor_model.pkl')

def get_macro_estimates(calories):
    """
    Estimates macros using standard macro ratio:
    - 50% carbs
    - 20% protein
    - 30% fat
    (can adjust these based on goal)
    Returns values in grams.
    """
    carbs_percent = 0.5
    protein_percent = 0.2
    fat_percent = 0.3

    # Calories per gram
    cal_per_g_carbs = 4
    cal_per_g_protein = 4
    cal_per_g_fat = 9

    carbs_g = (calories * carbs_percent) / cal_per_g_carbs
    protein_g = (calories * protein_percent) / cal_per_g_protein
    fat_g = (calories * fat_percent) / cal_per_g_fat

    return {
        'Carbs (g)': round(carbs_g),
        'Protein (g)': round(protein_g),
        'Fat (g)': round(fat_g)
    }

def get_meal_plan(calories, preference_raw):
    preference = preference_raw.lower()

    if preference == 'vegan':
        if calories <= 1500:
            return {
                'Breakfast': 'Poha with peas and lemon, herbal tea',
                'Lunch': 'Rajma with brown rice and salad',
                'Dinner': 'Vegetable khichdi with papad',
                'Snack': 'Cucumber slices with lemon & salt',
                'Recipe Links': ['https://youtu.be/asY7cq6j0xE?si=Tjq2udWTmTdxGihk']
            }
        elif calories <= 1800:
            return {
                'Breakfast': 'Upma with veggies and chutney',
                'Lunch': 'Masoor dal, quinoa, and stir-fried bhindi',
                'Dinner': 'Millet dosa with coconut chutney',
                'Snack': 'Roasted chana',
                'Recipe Links': ['https://youtu.be/cAoYsLXUKGI?si=bdETIrhtLvtcuc9n']
            }
        else:
            return {
                'Breakfast': 'Vegan aloo paratha with mint chutney',
                'Lunch': 'Chana masala, jeera rice, salad',
                'Dinner': 'Sambar with idli and coconut chutney',
                'Snack': 'Dates and almonds',
                'Recipe Links': ['https://youtu.be/_QLLo1b5Zcw?si=OEVzPwr-jPJNYqvN']
            }

    elif preference == 'vegetarian':
        if calories <= 1500:
            return {
                'Breakfast': 'Moong dal chilla with mint chutney',
                'Lunch': 'Vegetable pulao and curd',
                'Dinner': 'Palak paneer with phulka',
                'Snack': 'Buttermilk or a fruit',
                'Recipe Links': ['https://youtu.be/nPi2GD2SqfQ?si=v2T8t30wU-sG5peu']
            }
        elif calories <= 1800:
            return {
                'Breakfast': 'Besan chilla and tomato chutney',
                'Lunch': 'Bhindi sabzi, dal, roti, salad',
                'Dinner': 'Stuffed capsicum with jeera rice',
                'Snack': 'Sprout salad',
                'Recipe Links': ['https://youtu.be/0_g_8FPr3Ag?si=H4XNgMCZrJMhGdLb']
            }
        else:
            return {
                'Breakfast': 'Paneer paratha with curd',
                'Lunch': 'Mixed vegetable curry, dal fry, rice, salad',
                'Dinner': 'Vegetable biryani with boondi raita',
                'Snack': 'Banana shake or roasted makhana',
                'Recipe Links': ['https://youtu.be/mS8uOQh-ue8?si=uvxdQy0So425tMVj']
            }

    else:  # Non-vegetarian or None
        if calories <= 1500:
            return {
                'Breakfast': 'Boiled egg sandwich with chutney',
                'Lunch': 'Grilled chicken salad with lemon dressing',
                'Dinner': 'Fish curry with roti',
                'Snack': 'Coconut water and peanuts',
                'Recipe Links': ['https://youtu.be/CRHOoqPniGI?si=uzkNQcdC2NwvZIVI']
            }
        elif calories <= 1800:
            return {
                'Breakfast': 'Omelette with multigrain toast',
                'Lunch': 'Chicken curry, dal, roti, cucumber salad',
                'Dinner': 'Egg bhurji with jeera rice',
                'Snack': 'Greek yogurt with fruit',
                'Recipe Links': ['https://youtu.be/nJj7mwDWtiY?si=UmKPFS6MwytKwEfE']
            }
        else:
            return {
                'Breakfast': 'Paratha with keema and curd',
                'Lunch': 'Chicken biryani with raita',
                'Dinner': 'Mutton curry with rice',
                'Snack': 'Boiled eggs or milkshake',
                'Recipe Links': ['https://youtu.be/l3WlyvCEJUM?si=ZhXpz9p0BhQzoxDU']
            }

def get_workout_plan(activity, goal):
    activity = activity.lower()
    goal = goal.lower()

    workout_plan = []
    if goal == 'weight loss':
        if activity == 'low':
            workout_plan = ['20-min walk', 'Gentle yoga', 'Stretching']
        elif activity == 'moderate':
            workout_plan = ['30-min brisk walk', 'Bodyweight circuit', 'Light jog']
        else:
            workout_plan = ['HIIT (20 mins)', 'Cardio boxing', 'Spin class']
    elif goal == 'muscle gain':
        if activity == 'low':
            workout_plan = ['Resistance band training', 'Wall sits', 'Bodyweight squats']
        elif activity == 'moderate':
            workout_plan = ['Full-body dumbbell workout', 'Push-ups & pull-ups', 'Lunges and planks']
        else:
            workout_plan = ['Heavy weightlifting', 'Split training (legs/chest/back)', 'Progressive overload routines']
    else:  # weight maintenance
        if activity == 'low':
            workout_plan = ['Daily 20-min walk', 'Yoga or Pilates', 'Light strength training']
        elif activity == 'moderate':
            workout_plan = ['Jogging', 'Swimming', 'Cycling (30 mins)']
        else:
            workout_plan = ['CrossFit', 'Running 5K', 'Strength + Cardio combo']

    workout_video_links = {
        '20-min walk': 'https://youtu.be/Hfdajt9lWFo?si=beWZbpOiequcpNZ3',
        'HIIT (20 mins)': 'https://youtu.be/yXHgcYpUVD4?si=QeUBtDTjaVeSdr72',
        'Resistance band training': 'https://youtu.be/9qqnYOcSpY8?si=4ZO6gqGU1dhFdWaE',
        'Full-body dumbbell workout': 'https://youtu.be/XxuRSjER3Qk?si=fgDJbD3Vwv9hmBLr',
        'Heavy weightlifting': 'https://youtu.be/588-C4bEL28?si=c3kQAXtG87XeC3_V',
        'Jogging': 'https://youtu.be/3XbfW90grUk?si=YNFLQZwyfz6nWYFv',
        'CrossFit': 'https://youtu.be/VFWrXJ_fOao?si=dxP5dzc0UCINq0TA',
    }

    return [(exercise, workout_video_links.get(exercise, '')) for exercise in workout_plan]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # Get form data
        age = int(request.form['age'])
        gender_raw = request.form.get('gender')
        height = int(request.form['height'])
        weight = int(request.form['weight'])
        activity_raw = request.form.get('activity')
        goal_raw = request.form.get('goal')
        preference_raw = request.form.get('preference')  # Make sure this is used only if relevant
        lifestyle_raw = request.form.get('lifestyle')
        restriction_raw = request.form.get('restriction')
        health_condition_raw = request.form.get('health_condition')

        # Validate fields (same as before)
        if any(val in ['', None] for val in [gender_raw, activity_raw, goal_raw]):
            return render_template('error.html', message="Please fill in all required fields.")

        # Encode input
        gender = le_gender.transform([gender_raw])[0]
        activity = le_activity.transform([activity_raw])[0]
        goal = le_goal.transform([goal_raw])[0]
        preference = le_preference.transform([preference_raw])[0] if preference_raw != 'None' else -1
        lifestyle = le_lifestyle.transform([lifestyle_raw])[0]
        restriction = le_restriction.transform([restriction_raw])[0]
        health_condition = le_health_condition.transform([health_condition_raw])[0] if health_condition_raw != 'None' else -1

        # Predict calories with only the features that were used during model training (e.g., 7 features)
        input_data = np.array([[age, gender, height, weight, activity, goal, preference]])  # Use only 7 features
        predicted_calories = model.predict(input_data)[0]

        # Generate plans (same as before)
        macro_estimates = get_macro_estimates(predicted_calories)
        meal_plan = get_meal_plan(predicted_calories, preference_raw)
        workout_plan = get_workout_plan(activity_raw, goal_raw)

        return render_template(
            'result.html',
            calories=int(predicted_calories),
            meal_plan=meal_plan,
            workout_plan=workout_plan,
            macro_estimates=macro_estimates
        )


    except Exception as e:
        return render_template('error.html', message=str(e))

@app.route('/mood', methods=['GET', 'POST'])
def mood_suggester():
    if request.method == 'POST':
        data = request.get_json()
        mood_text = data.get('mood_input', '')
        suggestion = suggest_food(mood_text)
        return jsonify({"suggestion": suggestion})
    
    return render_template('mood.html')

@app.route('/image-upload', methods=['GET', 'POST'])
def image_upload():
    if request.method == 'POST':
        if 'mealImage' not in request.files:
            return render_template('image_upload.html', error="No file part")
        file = request.files['mealImage']
        if file.filename == '':
            return render_template('image_upload.html', error="No selected file")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        prediction_data = predict_food(filepath)
        return render_template(
            'image_upload.html',
            prediction=prediction_data,
            image_path=filepath
        )

    return render_template('image_upload.html')

@app.route('/imagerecog')
def imagerecog():
    return render_template('imagerecog.html')

if __name__ == '__main__':
    app.run(debug=True, port=5050)