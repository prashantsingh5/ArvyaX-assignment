"""
Simple Interactive Prediction Script - WORKS WITHOUT MODEL LOADING ISSUES

This script makes predictions directly without loading saved models.
Perfect for quick demos!
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
import re


# Train a lightweight model on the fly
print("Loading and training model (this takes ~30 seconds)...")

# Load training data
train_df = pd.read_excel('Sample_arvyax_reflective_dataset.xlsx')

# Text preprocessing
def preprocess_text(text):
    if pd.isna(text) or text == '':
        return "no_text"
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    text = ' '.join(text.split())
    return text

# Extract text features
vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2), min_df=2, max_df=0.8)
texts = train_df['journal_text'].apply(preprocess_text)
text_features = vectorizer.fit_transform(texts).toarray()

# Extract metadata features
meta = pd.DataFrame()
meta['duration'] = train_df['duration_min'].fillna(20)
meta['sleep'] = train_df['sleep_hours'].fillna(7)
meta['energy'] = train_df['energy_level'].fillna(3)
meta['stress'] = train_df['stress_level'].fillna(3)
meta['sleep_energy_ratio'] = meta['sleep'] / (meta['energy'] + 1)
meta['stress_energy_gap'] = meta['stress'] - meta['energy']
meta['text_len'] = train_df['journal_text'].fillna('').apply(len)
meta['word_count'] = train_df['journal_text'].fillna('').apply(lambda x: len(str(x).split()))

scaler = StandardScaler()
meta_scaled = scaler.fit_transform(meta)

# Combine features
X_train = np.hstack([text_features, meta_scaled])

# Prepare labels
state_encoder = LabelEncoder()
y_state = state_encoder.fit_transform(train_df['emotional_state'])
y_intensity = train_df['intensity'].values

# Train models
print("Training state model...")
state_model = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
state_model.fit(X_train, y_state)

print("Training intensity model...")
intensity_model = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
intensity_model.fit(X_train, y_intensity)

print("Models ready!\n")


def predict_emotion(journal_text, energy=3, stress=3, sleep=7.0, time_of_day='morning', duration=20):
    """Make a prediction"""

    # Process text
    text_processed = preprocess_text(journal_text)
    text_feat = vectorizer.transform([text_processed]).toarray()

    # Process metadata
    meta_dict = {
        'duration': duration,
        'sleep': sleep,
        'energy': energy,
        'stress': stress,
        'sleep_energy_ratio': sleep / (energy + 1),
        'stress_energy_gap': stress - energy,
        'text_len': len(journal_text),
        'word_count': len(journal_text.split())
    }
    meta_df = pd.DataFrame([meta_dict])
    meta_scaled = scaler.transform(meta_df)

    # Combine
    X = np.hstack([text_feat, meta_scaled])

    # Predict
    state_probs = state_model.predict_proba(X)[0]
    state_pred = state_encoder.inverse_transform([state_model.predict(X)[0]])[0]
    intensity_pred = int(np.clip(intensity_model.predict(X)[0], 1, 5).round())

    confidence = float(state_probs.max())
    uncertain = 1 if (confidence < 0.6 or len(journal_text.split()) <= 3) else 0

    # Decision engine
    action, timing = decide_action(state_pred, intensity_pred, energy, stress, sleep, time_of_day)

    return {
        'state': state_pred,
        'intensity': intensity_pred,
        'confidence': confidence,
        'uncertain': uncertain,
        'action': action,
        'timing': timing
    }


def decide_action(state, intensity, energy, stress, sleep, time_of_day):
    """Simplified decision engine"""

    # High stress - prioritize calming
    if stress >= 4:
        if energy <= 2:
            return 'rest', 'now'
        elif intensity >= 4:
            return 'grounding', 'now'
        else:
            return 'box_breathing', 'within_15_min'

    # Low energy
    if energy <= 2:
        if sleep < 6:
            return 'rest', 'tonight'
        else:
            return 'power_nap', 'within_15_min'

    # Positive states with energy
    if state in ['calm', 'focused'] and energy >= 4:
        if time_of_day in ['morning', 'afternoon']:
            return 'deep_work', 'now'
        else:
            return 'light_planning', 'later_today'

    # Negative states
    if state in ['overwhelmed', 'restless'] and intensity >= 3:
        return 'box_breathing', 'within_15_min'

    # Default
    return 'journaling', 'within_15_min'


def display_result(result, journal_text, energy, stress, sleep, time_of_day):
    """Display prediction results"""

    print("\n" + "="*80)
    print(" " * 28 + "UNDERSTANDING YOU")
    print("="*80)

    print(f"\nEmotional State: {result['state'].upper()}")
    print(f"Intensity: {result['intensity']}/5 [" + "=" * result['intensity'] + "-" * (5-result['intensity']) + "]")
    print(f"Confidence: {result['confidence']:.1%}", end="")

    if result['confidence'] >= 0.7:
        print(" (High)")
    elif result['confidence'] >= 0.5:
        print(" (Medium)")
    else:
        print(" (Low)")

    if result['uncertain']:
        print("Uncertain: YES (system is not entirely sure)")

    print("\n" + "="*80)
    print(" " * 30 + "YOUR GUIDANCE")
    print("="*80)

    print(f"\nRecommended Action: {result['action'].upper().replace('_', ' ')}")
    print(f"When: {result['timing'].upper().replace('_', ' ')}")

    # Supportive message
    messages = {
        'overwhelmed': "You're feeling overwhelmed. Let's take things one step at a time.",
        'restless': "You seem restless. Your energy needs an outlet.",
        'calm': "You're in a calm space. Great time to channel this clarity.",
        'focused': "Your focus is strong! Perfect moment for meaningful work.",
        'neutral': "You're in a neutral space. Let's find what resonates.",
        'mixed': "I hear complexity in your reflection. That's natural."
    }

    print(f"\nMessage: {messages.get(result['state'], 'I hear you.')}")

    print("\n" + "="*80)
    print(f"Context: Energy={energy}/5, Stress={stress}/5, Sleep={sleep}h, Time={time_of_day}")
    print("="*80)


def main():
    """Interactive demo"""

    print("\n" + "="*80)
    print(" " * 20 + "ArvyaX Emotional State Prediction System")
    print(" " * 25 + "From Understanding -> To Guiding")
    print("="*80)

    examples = {
        '1': {
            'name': "Stressed & Overwhelmed",
            'text': "I feel so overwhelmed right now, everything is too much",
            'energy': 1, 'stress': 5, 'sleep': 5.0, 'time': 'evening'
        },
        '2': {
            'name': "Calm & Focused",
            'text': "That forest session was peaceful. I feel clear and ready to work.",
            'energy': 4, 'stress': 1, 'sleep': 8.0, 'time': 'morning'
        },
        '3': {
            'name': "Very Short",
            'text': "ok",
            'energy': 3, 'stress': 3, 'sleep': 6.5, 'time': 'afternoon'
        },
        '4': {
            'name': "Mixed Feelings",
            'text': "I don't know... some parts felt good but I'm still anxious",
            'energy': 3, 'stress': 4, 'sleep': 6.0, 'time': 'evening'
        }
    }

    while True:
        print("\n" + "="*80)
        print("DEMO OPTIONS")
        print("="*80)
        print("\n1. Stressed & Overwhelmed example")
        print("2. Calm & Focused example")
        print("3. Very Short Input example")
        print("4. Mixed Feelings example")
        print("5. Enter your own reflection")
        print("6. Exit")

        choice = input("\nChoose option (1-6): ").strip()

        if choice == '6':
            print("\nThank you for using ArvyaX!\n")
            break

        if choice in ['1', '2', '3', '4']:
            ex = examples[choice]
            print(f"\n[Selected: {ex['name']}]")
            print(f'Text: "{ex["text"]}"')

            result = predict_emotion(
                ex['text'],
                energy=ex['energy'],
                stress=ex['stress'],
                sleep=ex['sleep'],
                time_of_day=ex['time']
            )

            display_result(result, ex['text'], ex['energy'], ex['stress'], ex['sleep'], ex['time'])

        elif choice == '5':
            print("\n" + "-"*80)
            journal_text = input("Enter your reflection: ").strip() or "unclear"

            print("\nQuick context (press Enter for defaults):")
            try:
                energy = int(input("Energy level (1-5) [3]: ").strip() or "3")
                stress = int(input("Stress level (1-5) [3]: ").strip() or "3")
                sleep = float(input("Sleep hours [7.0]: ").strip() or "7.0")
                time_of_day = input("Time (morning/afternoon/evening/night) [morning]: ").strip() or "morning"
            except:
                energy, stress, sleep, time_of_day = 3, 3, 7.0, 'morning'

            result = predict_emotion(journal_text, energy, stress, sleep, time_of_day)
            display_result(result, journal_text, energy, stress, sleep, time_of_day)

        print("\n" + "-"*80)
        again = input("Try another? (y/n): ").strip().lower()
        if again != 'y' and again != 'yes':
            break

    print()


if __name__ == '__main__':
    main()
