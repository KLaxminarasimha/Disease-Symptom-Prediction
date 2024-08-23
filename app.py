from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load a synthetic dataset (replace it with your dataset)
df = pd.DataFrame({
    'Fever': [1, 0, 1, 0, 1],
    'Cough': [1, 0, 1, 0, 1],
    'Fatigue': [1, 0, 1, 1, 0],
    'Difficulty_Breathing': [1, 1, 0, 0, 1],
    'Disease': ['Flu', 'Cold', 'Flu', 'Cold', 'Asthma']
})

# Encode categorical variables
le = LabelEncoder()
df['Disease'] = le.fit_transform(df['Disease'])

# Features and target variable
X = df.drop('Disease', axis=1)
y = df['Disease']

# Train the DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = {
            'Fever': int(request.form['fever']),
            'Cough': int(request.form['cough']),
            'Fatigue': int(request.form['fatigue']),
            'Difficulty_Breathing': int(request.form['difficulty_breathing'])
        }

        user_df = pd.DataFrame([user_input])
        prediction = model.predict(user_df)[0]
        result = le.inverse_transform([prediction])[0]

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
