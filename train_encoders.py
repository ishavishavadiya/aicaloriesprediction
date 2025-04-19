from sklearn.preprocessing import LabelEncoder
import joblib

# Example data for each field (replace these with actual data)
gender_data = ['Male', 'Female', 'Male', 'Female']
activity_data = ['Low', 'Moderate', 'High']
goal_data = ['Weight Loss', 'Weight Maintenance', 'Muscle Gain']
preference_data = ['Vegetarian', 'Non-Vegetarian', 'Vegan']
lifestyle_data = ['Sedentary', 'Active', 'Very Active']
restriction_data = ['No Restriction', 'Dairy Free', 'Gluten Free']
health_condition_data = ['None', 'Diabetes', 'Heart Disease']

# Initialize LabelEncoders
le_gender = LabelEncoder()
le_activity = LabelEncoder()
le_goal = LabelEncoder()
le_preference = LabelEncoder()
le_lifestyle = LabelEncoder()
le_restriction = LabelEncoder()
le_health_condition = LabelEncoder()

# Fit each encoder on its respective data
le_gender.fit(gender_data)
le_activity.fit(activity_data)
le_goal.fit(goal_data)
le_preference.fit(preference_data)
le_lifestyle.fit(lifestyle_data)
le_restriction.fit(restriction_data)
le_health_condition.fit(health_condition_data)

# Save each encoder as a .pkl file
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_activity, 'le_activity.pkl')
joblib.dump(le_goal, 'le_goal.pkl')
joblib.dump(le_preference, 'le_preference.pkl')
joblib.dump(le_lifestyle, 'le_lifestyle.pkl')
joblib.dump(le_restriction, 'le_restriction.pkl')
joblib.dump(le_health_condition, 'le_health_condition.pkl')

print("Encoders saved successfully!")
