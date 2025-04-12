from flask import Flask, request, jsonify
import numpy as np
from suggest_config_for_user import suggest_config_for_user  # Assuming this is your model function

app = Flask(__name__)

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
        config_vector = data.get("config_vector")
        if config_vector is None:
            return jsonify({"error": "Missing config_vector"}), 400

        # Convert to NumPy array
        config_vector = np.array(config_vector)

        # Get model suggestion
        suggestion = suggest_config_for_user(config_vector, r'C:\Users\Tuan Anh HSLU\OneDrive - Hochschule Luzern\Desktop\HSLU22\Bachelor Thesis\ML Models\models\best_model\best_model.zip', 10)
        
        # Convert NumPy types to JSON serializable Python types
        serializable_suggestion = convert_to_serializable(suggestion)

        return jsonify(serializable_suggestion), 200

    except Exception as e:
        # Add logging for better debugging
        print(f"Error in suggestion endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
