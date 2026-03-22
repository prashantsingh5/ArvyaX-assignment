# -*- coding: utf-8 -*-
"""
Flask Web API for ArvyaX Emotional State Prediction

Run this to start a web server for real-time predictions.
Access at: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import sys
import os
from datetime import datetime

# Fix Unicode output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_model import EmotionalStatePredictor

app = Flask(__name__)

# Load model once at startup
print("Loading trained models...")
predictor = EmotionalStatePredictor()
try:
    predictor.load_models('models')
    print("[OK] Models loaded successfully!")
except Exception as e:
    print(f"[ERROR] Error loading models: {e}")
    print("Please run 'python src/train_model.py' first")
    exit(1)


def generate_supportive_message(state, intensity, action, confidence):
    """Generate human-like supportive message"""

    messages = {
        'overwhelmed': "I can sense you're feeling overwhelmed. Let's take things one step at a time.",
        'anxious': "It sounds like there's some anxiety. Let's work through it together.",
        'restless': "You seem restless. Your energy needs an outlet.",
        'calm': "You're in a calm space. This is a great time to channel this clarity.",
        'focused': "Your focus is strong! Perfect moment for meaningful work.",
        'neutral': "You're in a neutral space. Let's find what resonates with you.",
        'mixed': "I hear complexity in your reflection. That's natural."
    }

    opening = messages.get(state, "I hear you.")

    action_messages = {
        'box_breathing': "Try a short breathing exercise to calm your nervous system.",
        'journaling': "Writing more about this might help clarify what you're feeling.",
        'grounding': "Let's ground yourself in the present moment.",
        'deep_work': "Your mind is ready for focused, meaningful work.",
        'yoga': "Some gentle movement could help release what you're holding.",
        'sound_therapy': "Calming sounds might help you transition.",
        'light_planning': "A bit of planning might ease your mind.",
        'rest': "Your body is telling you it needs rest. Listen to it.",
        'movement': "Your body needs movement to shift this energy.",
        'pause': "Take a moment to pause and check in with yourself.",
        'power_nap': "A short rest could recharge you."
    }

    action_msg = action_messages.get(action, "Let's find what helps you most.")

    if confidence < 0.5:
        return f"{opening} I'm not entirely certain, but {action_msg}"
    else:
        return f"{opening} {action_msg}"


@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""

    try:
        data = request.json
        
        # Validate that JSON was received
        if data is None:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'error_type': 'ValueError'
            }), 400

        # Create input dataframe with proper default values
        input_df = pd.DataFrame([{
            'id': 99999,
            'journal_text': data.get('journal_text', ''),
            'ambience_type': data.get('ambience_type', 'forest'),
            'duration_min': int(data.get('duration_min', 20)),
            'sleep_hours': float(data.get('sleep_hours', 7.0)),
            'energy_level': int(data.get('energy_level', 3)),
            'stress_level': int(data.get('stress_level', 3)),
            'time_of_day': data.get('time_of_day', 'morning'),
            'previous_day_mood': data.get('previous_day_mood', 'calm'),
            'face_emotion_hint': data.get('face_emotion_hint', 'neutral_face'),
            'reflection_quality': data.get('reflection_quality', 'clear')
        }])

        # Make prediction
        predictions, _ = predictor.predict_with_uncertainty(input_df)
        actions, timings = predictor.decide_action(predictions, input_df)

        state = predictions['predicted_state'].values[0]
        intensity = int(predictions['predicted_intensity'].values[0])
        confidence = float(predictions['confidence'].values[0])
        uncertain = int(predictions['uncertain_flag'].values[0])

        # Generate supportive message
        message = generate_supportive_message(state, intensity, actions[0], confidence)

        # Return results
        response = {
            'success': True,
            'prediction': {
                'emotional_state': state,
                'intensity': intensity,
                'confidence': round(confidence, 3),
                'uncertain_flag': uncertain
            },
            'recommendation': {
                'what_to_do': actions[0],
                'when_to_do': timings[0]
            },
            'message': message,
            'context': {
                'text_length': len(data.get('journal_text', '')),
                'word_count': len(data.get('journal_text', '').split())
            }
        }

        return jsonify(response)

    except Exception as e:
        # Log detailed error information for debugging
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"[PREDICTION ERROR] {error_msg}", flush=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.state_model is not None,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🌿 ArvyaX Emotional State Prediction API")
    print("="*80)
    print("\n✓ Server starting...")
    print("\n📱 Access the demo at: http://localhost:5000")
    print("📡 API endpoint: http://localhost:5000/predict")
    print("\n" + "="*80)
    print("\nPress Ctrl+C to stop the server")
    print()

    app.run(debug=True, host='0.0.0.0', port=5000)
