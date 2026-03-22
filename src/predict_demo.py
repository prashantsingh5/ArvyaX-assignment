"""
Simple Inference Demo - Make predictions on new inputs
"""

import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_model import EmotionalStatePredictor


def predict_single(journal_text, ambience='forest', duration=20, sleep=7.0,
                   energy=3, stress=3, time='morning', prev_mood='neutral',
                   emotion_hint='calm', quality='clear'):
    """
    Make a prediction for a single journal entry

    Args:
        journal_text: User's reflection text
        ambience: Type of ambience (forest, ocean, rain, mountain, cafe)
        duration: Session duration in minutes
        sleep: Hours of sleep
        energy: Energy level (1-5)
        stress: Stress level (1-5)
        time: Time of day (morning, afternoon, evening, night)
        prev_mood: Previous day mood
        emotion_hint: Face emotion hint
        quality: Reflection quality (clear, vague, medium)
    """

    # Load trained model
    predictor = EmotionalStatePredictor()
    predictor.load_models('models')

    # Create input dataframe
    input_df = pd.DataFrame([{
        'id': 99999,
        'journal_text': journal_text,
        'ambience_type': ambience,
        'duration_min': duration,
        'sleep_hours': sleep,
        'energy_level': energy,
        'stress_level': stress,
        'time_of_day': time,
        'previous_day_mood': prev_mood,
        'face_emotion_hint': emotion_hint,
        'reflection_quality': quality
    }])

    # Make prediction
    predictions, _ = predictor.predict_with_uncertainty(input_df)

    # Get recommended action
    actions, timings = predictor.decide_action(predictions, input_df)

    # Display results
    print("\n" + "="*80)
    print("EMOTIONAL STATE PREDICTION")
    print("="*80)
    print(f"\nInput Text: \"{journal_text}\"")
    print(f"\nContext:")
    print(f"  - Ambience: {ambience}")
    print(f"  - Duration: {duration} min")
    print(f"  - Sleep: {sleep}h")
    print(f"  - Energy: {energy}/5")
    print(f"  - Stress: {stress}/5")
    print(f"  - Time: {time}")

    print(f"\n{'─'*80}")
    print(f"RESULTS")
    print(f"{'─'*80}")
    print(f"\nEmotional State: {predictions['predicted_state'].values[0]}")
    print(f"Intensity: {predictions['predicted_intensity'].values[0]}/5")
    print(f"Confidence: {predictions['confidence'].values[0]:.2%}")
    print(f"Uncertain: {'Yes' if predictions['uncertain_flag'].values[0] else 'No'}")

    print(f"\n{'─'*80}")
    print(f"RECOMMENDED ACTION")
    print(f"{'─'*80}")
    print(f"\nWhat to do: {actions[0]}")
    print(f"When: {timings[0]}")

    print("\n" + "="*80)

    return predictions, actions, timings


if __name__ == '__main__':
    # Example 1: Calm, well-rested morning
    print("\n### Example 1: Calm Morning ###")
    predict_single(
        journal_text="That was peaceful. The forest sounds helped me think clearly about my goals.",
        ambience='forest',
        duration=25,
        sleep=8.0,
        energy=4,
        stress=2,
        time='morning',
        prev_mood='neutral',
        emotion_hint='calm'
    )

    # Example 2: Stressed, low energy
    print("\n### Example 2: Stressed and Tired ###")
    predict_single(
        journal_text="I don't know... everything feels like too much right now",
        ambience='rain',
        duration=10,
        sleep=5.0,
        energy=1,
        stress=5,
        time='evening',
        prev_mood='stressed',
        emotion_hint='anxious'
    )

    # Example 3: Very short input
    print("\n### Example 3: Short Input ###")
    predict_single(
        journal_text="ok",
        ambience='cafe',
        duration=5,
        sleep=6.0,
        energy=3,
        stress=3,
        time='afternoon',
        prev_mood='neutral',
        emotion_hint='neutral'
    )

    # Example 4: High energy, ready for work
    print("\n### Example 4: Energized for Work ###")
    predict_single(
        journal_text="Feeling really focused after that session. Ready to tackle my projects!",
        ambience='mountain',
        duration=30,
        sleep=7.5,
        energy=5,
        stress=1,
        time='morning',
        prev_mood='good',
        emotion_hint='energized'
    )
