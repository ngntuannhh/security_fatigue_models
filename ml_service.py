from flask import Flask, request, jsonify
import numpy as np
import subprocess
import json
import os
import sys
from suggest_config_for_user import suggest_config_for_user  # Assuming this is your model function

app = Flask(__name__)

MODEL_VER = 'run_tuned_1m'  # or your chosen model version

# Get the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", MODEL_VER, "best_model", "best_model.zip")
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback_buffer.jsonl")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train_agent.py")

def convert_to_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()  # Converts any numpy number to its Python equivalent
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

@app.route('/suggest', methods=['POST'])
def suggest():
    try:
        data = request.get_json()

        # Extract config vector from the JSON payload
        config_vector = data.get("configVector")
        # if config_vector is None:
        #     return jsonify({"error": "Missing config_vector"}), 400

        # Convert to NumPy array
        config_vector = np.array(config_vector)

        #DRL Model Path
        path = MODEL_PATH
        # Get model suggestion
        suggestion = suggest_config_for_user(config_vector, path, 100)
        
        # Convert NumPy types to JSON serializable Python types
        serializable_suggestion = convert_to_serializable(suggestion)

        return jsonify(serializable_suggestion), 200

    except Exception as e:
        # Add logging for better debugging
        print(f"Error in suggestion endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        # Get feedback data from request
        feedback_data = request.get_json()
        
        if not feedback_data or not isinstance(feedback_data, list):
            return "ERROR", 400
        
        # Define the feedback file path
        feedback_file = FEEDBACK_FILE

        # Save the feedback data to a JSONL file
        with open(feedback_file, 'w') as f:
            for entry in feedback_data:
                f.write(json.dumps(entry) + '\n')
    
        print("About to run retraining process directly")
        
        # Import the retrain function directly
        from train_agent import retrain_from_feedback
        
        # Call it directly in the same process
        success = retrain_from_feedback()
        
        if success:
            print("Retraining completed successfully")
            return "OK", 200
        else:
            print("Retraining failed")
            return "ERROR", 500
        

        # # # Use Popen to run the training in the background
        # # subprocess.Popen(["python", 
        # #                TRAIN_SCRIPT, 
        # #                   "--feedback_file", feedback_file, 
        # #                   "--model_path", MODEL_PATH,
        # #                "--retrain_from_feedback1"], 
        # #               shell=True, 
        # #               creationflags=subprocess.CREATE_NO_WINDOW)
        
        # # Return a simple "OK" response
        # return "OK", 200
            
    except Exception as e:
        print(f"Error in retraining endpoint: {e}")
        return "ERROR", 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
