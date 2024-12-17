from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

# Load the encoders
encoders = pickle.load(open("encoders.pkl", "rb"))

# Load the model
model = pickle.load(open("autism_model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input JSON data
        data = request.get_json()

        # Define training column names from your dataset
        required_columns = [
            "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score", "A6_Score",
            "A7_Score", "A8_Score", "A9_Score", "A10_Score", "age", "gender",
            "ethnicity", "jaundice", "austim",  "contry_of_res", "used_app_before",
            "result", "relation"
        ]

        # Check if all required fields are present in the input data
        for column in required_columns:
            if column not in data:
                return jsonify({'error': f'Missing required field: {column}'}), 400

        # Standardize column names in input data to match model training columns
        standardized_data = {column: data.get(column) for column in required_columns}

        # Convert input data to a DataFrame
        input_data = pd.DataFrame([standardized_data])

        # Ensure all columns are in the encoder dictionary
        for column in input_data.columns:
            if column in encoders:
                input_data[column] = encoders[column].transform(input_data[column])

        # Make prediction
        prediction = model.predict(input_data)

        # Return prediction as JSON
        return jsonify({'prediction': prediction.tolist()[0]})

    except KeyError as ke:
        return jsonify({'error': f'Missing required data: {ke}'}), 400  # Return error with HTTP status code 400
    except ValueError as ve:
        return jsonify({'error': f'Invalid data format: {ve}'}), 400  # Return error with HTTP status code 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error with HTTP status code 500 for any other exception

if __name__ == '__main__':
    app.run(debug=True)
