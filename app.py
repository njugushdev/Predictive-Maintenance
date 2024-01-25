from flask import Flask, render_template, request, jsonify, json
import joblib
from sklearn.preprocessing import StandardScaler
from nexmo import Client

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

#Load the fitted scaler
sc = joblib.load('scaler.joblib')

# Vonage credentials
VONAGE_API_KEY = '92a47d2e'
VONAGE_API_SECRET = 'TxIcJZ469gzjU57z'
VONAGE_PHONE_NUMBER = 'Vonage APIs'
YOUR_PHONE_NUMBER = '254768022225'

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the input values from the form as a JSON object
            input_data_json = request.form['input_data']
            input_data = json.loads(input_data_json)

            # Extract the required values
            input_values = [
                float(input_data.get('TWC1_Read', 0.0)),
                float(input_data.get('TWC1_Expected', 0.0)),
                float(input_data.get('Variance', 0.0)),
                float(input_data.get('TWC2_Read', 0.0)),
                float(input_data.get('TWC2_Expected', 0.0)),
                float(input_data.get('Variance.1', 0.0)),
                float(input_data.get('TWC3_Read', 0.0)),
                float(input_data.get('TWC3_Expected', 0.0)),
                float(input_data.get('Variance.2', 0.0)),
                float(input_data.get('TWC4_Read', 0.0)),
                float(input_data.get('TWC4_Expected', 0.0)),
                float(input_data.get('Variance.3', 0.0)),
                float(input_data.get('TWCTR_Read', 0.0)),
                float(input_data.get('TWCTR_Expected', 0.0)),
                float(input_data.get('Variance.4', 0.0)),
                float(input_data.get('WTK_Read', 0.0)),
                float(input_data.get('WTK_Expected', 0.0)),
                float(input_data.get('Variance.5', 0.0)),
                float(input_data.get('WTU_Read(Truck)', 0.0)),
                float(input_data.get('WTU_Expected(Truck)', 0.0)),
                float(input_data.get('Variance.6', 0.0)),
                input_data.get('Visual_Inspection_Okay', 'False').lower() == 'true'
            ]

            # Scale the input values
            input_values_scaled = sc.transform([input_values])

            # Make a prediction
            prediction = model.predict(input_values_scaled)

             # Send SMS notification
            send_sms_notification(f"Prediction: {prediction[0]}")

            # Return the prediction to the user
            return render_template('result.html', prediction=prediction[0])

        except Exception as e:
            return jsonify({'error': f'Error processing the request: {str(e)}'}), 400

    return jsonify({'error': 'Invalid request method'}), 400

def send_sms_notification(message):
    # Use the Vonage library to send an SMS
    client = Client(
        key=VONAGE_API_KEY,
        secret=VONAGE_API_SECRET,
    )
    
    # Replace "YOUR_PHONE_NUMBER" with the actual phone number to receive notifications
    response = client.send_message(
        {
            "from": "Vonage APIs",
            "to": "254768022225",
            "text": message,
        }
    )

    # Check if the message was sent successfully
    if response["messages"][0]["status"] == "0":
        print("Message sent successfully")
    else:
        print(f"Message failed with error: {response['messages'][0]['error-text']}")


if __name__ == '__main__':
    app.run(debug=False)
