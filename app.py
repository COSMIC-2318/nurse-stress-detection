import joblib
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable memory-heavy optimizations

# Initialize the Flask application
app = Flask(__name__)

# Load the production assets into memory at startup
try:
    model = load_model('stress_detection_model.keras')
    scaler = joblib.load('stress_scaler.pkl')
except Exception as e:
    print(f"Error loading assets: {e}. Ensure the .keras and .pkl files are in the same folder.")

@app.route('/', methods=['GET'])
def home():
    """Renders the standard input form when the user navigates to the homepage."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Processes the form data, scales it, and returns the prediction, tips, and UI styling."""
    try:
        # 1. Extract data from the HTML form
        eda = float(request.form['eda'])
        hr = float(request.form['hr'])
        temp = float(request.form['temp'])
        shift = int(request.form['shift'])
        
        # 2. Format the input array for exactly 4 features
        input_features = np.array([[eda, temp, hr, shift]])
        
        # 3. Scale the features using the saved distribution metrics
        scaled_features = scaler.transform(input_features)
        
        # 4. Execute model inference
        prediction_probs = model.predict(scaled_features)
        predicted_class = int(np.argmax(prediction_probs, axis=1)[0])
        
        # 5. Define direct messages, tips, and dynamic CSS styling based on the class
        if predicted_class == 0:
            result_status = "You are currently relaxed (No Stress)."
            alert_class = "status-success" # Triggers green UI
            health_tips = [
                "Maintain your current work pace; you are operating efficiently.",
                "Ensure you stay hydrated by drinking a glass of water.",
                "Take brief stretches every 60 minutes to maintain this optimal state."
            ]
        elif predicted_class == 1:
            result_status = "Warning: You are in Stress."
            alert_class = "status-danger" # Triggers red UI
            health_tips = [
                "Stop your current non-critical task and step away for 3 to 5 minutes.",
                "Practice deep breathing: Inhale for 4 seconds, hold for 4 seconds, exhale for 6 seconds.",
                "Communicate with your supervisor if your current workload feels unmanageable."
            ]
        elif predicted_class == 2:
            result_status = "High Physical Activity / Arousal Detected."
            alert_class = "status-warning" # Triggers orange UI
            health_tips = [
                "Your body is working hard. Take a physical micro-break to rest your muscles.",
                "Replenish your fluids immediately to compensate for physical exertion.",
                "Monitor yourself for physical fatigue and sit down if you feel lightheaded."
            ]
        else:
            result_status = "Unknown State"
            alert_class = "status-neutral"
            health_tips = ["Please consult a medical professional if you feel unwell."]
        
        # 6. Render the template, passing the status, tips, and the UI color class
        return render_template('index.html', prediction_text=result_status, tips=health_tips, alert_class=alert_class)
        gc.collect()

    
    except Exception as e:
        # Failsafe to show errors on the webpage
        return render_template('index.html', prediction_text=f'Error processing request: {str(e)}', alert_class="status-danger")

if __name__ == '__main__':
    # Run the Flask development server on localhost
    app.run(debug=True)
