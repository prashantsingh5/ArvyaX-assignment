"""
Interactive CLI Demo - Real-time Emotional State Prediction

This allows you to input your own journal reflection and get instant predictions.
Perfect for showcasing the system in interviews!
"""

import pandas as pd
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_model import EmotionalStatePredictor


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*80)
    print(" " * 20 + "🌿 ArvyaX Emotional State Prediction System")
    print(" " * 25 + "From Understanding → To Guiding")
    print("="*80)


def print_divider():
    """Print section divider"""
    print("-" * 80)


def generate_supportive_message(state, intensity, action, confidence):
    """Generate human-like supportive message"""

    messages = {
        'overwhelmed': {
            'high': "I can sense you're feeling overwhelmed right now. That's completely okay. Let's take things one step at a time.",
            'low': "You seem a bit overwhelmed. Let's slow things down and focus on what matters most."
        },
        'anxious': {
            'high': "It sounds like anxiety is running high. You're not alone in this. Let's work through it together.",
            'low': "I'm picking up some anxious energy. Let's channel that into something grounding."
        },
        'restless': {
            'high': "You seem quite restless. Your energy needs an outlet right now.",
            'low': "There's a restless quality to your reflection. Let's help you find some stillness."
        },
        'calm': {
            'high': "You're in a calm, centered space. This is a great time to channel this clarity.",
            'low': "You seem fairly calm. Let's build on this peaceful state."
        },
        'focused': {
            'high': "Your focus is strong! This is the perfect moment to dive into meaningful work.",
            'low': "You're feeling somewhat focused. Let's use this clarity wisely."
        },
        'neutral': {
            'high': "You're in a neutral space - neither up nor down. Let's find what resonates with you.",
            'low': "Things feel neutral right now. Let's see what you need most."
        },
        'mixed': {
            'high': "I hear complexity in your reflection - multiple feelings at once. That's natural.",
            'low': "You seem to be feeling a mix of things. Let's untangle that together."
        }
    }

    intensity_key = 'high' if intensity >= 4 else 'low'
    opening = messages.get(state, {}).get(intensity_key, "I hear you.")

    # Action suggestions
    action_messages = {
        'box_breathing': "Try a short breathing exercise to calm your nervous system.",
        'journaling': "Writing more about this might help clarify what you're feeling.",
        'grounding': "Let's ground yourself in the present moment with a grounding exercise.",
        'deep_work': "Your mind is ready for focused, meaningful work.",
        'yoga': "Some gentle movement could help release what you're holding.",
        'sound_therapy': "Calming sounds might help you transition into rest.",
        'light_planning': "A bit of planning might ease your mind without overwhelming you.",
        'rest': "Your body is telling you it needs rest. Listen to it.",
        'movement': "Your body needs movement to shift this energy.",
        'pause': "Take a moment to pause and check in with yourself.",
        'power_nap': "A short rest could recharge you right now."
    }

    action_msg = action_messages.get(action, "Let's find what helps you most.")

    # Confidence qualifier
    if confidence < 0.4:
        qualifier = "I'm not entirely certain, but "
    elif confidence < 0.6:
        qualifier = ""
    else:
        qualifier = ""

    return f"{opening} {qualifier}{action_msg}"


def get_user_input():
    """Collect input from user"""
    print("\n" + "="*80)
    print("SHARE YOUR REFLECTION")
    print("="*80)

    print("\nTell me about your session or how you're feeling:")
    journal_text = input("> ").strip()

    if not journal_text:
        journal_text = "I'm not sure how to describe it"

    print("\nLet me gather some context...")
    print_divider()

    # Context inputs with defaults
    print("\nWhat ambience were you in? (forest/ocean/rain/mountain/cafe) [forest]:")
    ambience = input("> ").strip() or "forest"

    print("\nHow long was your session? (minutes) [20]:")
    try:
        duration = int(input("> ").strip() or "20")
    except:
        duration = 20

    print("\nHow many hours did you sleep last night? [7.0]:")
    try:
        sleep = float(input("> ").strip() or "7.0")
    except:
        sleep = 7.0

    print("\nYour energy level right now? (1-5) [3]:")
    try:
        energy = int(input("> ").strip() or "3")
        energy = max(1, min(5, energy))
    except:
        energy = 3

    print("\nYour stress level right now? (1-5) [3]:")
    try:
        stress = int(input("> ").strip() or "3")
        stress = max(1, min(5, stress))
    except:
        stress = 3

    print("\nWhat time is it? (morning/afternoon/evening/night) [morning]:")
    time_of_day = input("> ").strip() or "morning"

    print("\nHow was your mood yesterday? (good/neutral/stressed/tired) [neutral]:")
    prev_mood = input("> ").strip() or "neutral"

    print("\nYour facial emotion hint? (calm/anxious/happy/tired/neutral) [neutral]:")
    emotion_hint = input("> ").strip() or "neutral"

    print("\nReflection quality? (clear/medium/vague) [medium]:")
    quality = input("> ").strip() or "medium"

    return {
        'journal_text': journal_text,
        'ambience_type': ambience,
        'duration_min': duration,
        'sleep_hours': sleep,
        'energy_level': energy,
        'stress_level': stress,
        'time_of_day': time_of_day,
        'previous_day_mood': prev_mood,
        'face_emotion_hint': emotion_hint,
        'reflection_quality': quality
    }


def make_prediction(predictor, user_input):
    """Make prediction on user input"""

    # Create dataframe
    input_df = pd.DataFrame([{
        'id': 99999,
        **user_input
    }])

    # Predict
    predictions, _ = predictor.predict_with_uncertainty(input_df)
    actions, timings = predictor.decide_action(predictions, input_df)

    return predictions, actions[0], timings[0]


def display_results(predictions, action, timing, user_input):
    """Display prediction results in a beautiful format"""

    state = predictions['predicted_state'].values[0]
    intensity = predictions['predicted_intensity'].values[0]
    confidence = predictions['confidence'].values[0]
    uncertain = bool(predictions['uncertain_flag'].values[0])

    print("\n\n" + "="*80)
    print(" " * 30 + "🎯 UNDERSTANDING YOU")
    print("="*80)

    print(f"\n💭 Emotional State: {state.upper()}")
    print(f"📊 Intensity: {intensity}/5", end="")
    print(" " + "█" * intensity + "░" * (5-intensity))
    print(f"🎲 Confidence: {confidence:.1%}", end="")

    if confidence >= 0.7:
        print(" (High)")
    elif confidence >= 0.5:
        print(" (Medium)")
    else:
        print(" (Low)")

    if uncertain:
        print("⚠️  Uncertainty Flag: YES (I'm not entirely sure)")

    print("\n" + "="*80)
    print(" " * 32 + "💡 GUIDANCE FOR YOU")
    print("="*80)

    print(f"\n🎯 Recommended Action: {action.upper().replace('_', ' ')}")
    print(f"⏰ When: {timing.upper().replace('_', ' ')}")

    # Generate supportive message
    message = generate_supportive_message(state, intensity, action, confidence)
    print(f"\n💬 Message:")
    print(f"   {message}")

    print("\n" + "="*80)

    # Context summary
    print("\n📋 Context Considered:")
    print(f"   Sleep: {user_input['sleep_hours']}h | Energy: {user_input['energy_level']}/5 | Stress: {user_input['stress_level']}/5")
    print(f"   Time: {user_input['time_of_day']} | Ambience: {user_input['ambience_type']}")

    print("\n" + "="*80)


def quick_mode(predictor):
    """Quick mode with preset examples"""
    print("\n" + "="*80)
    print("QUICK DEMO MODE - Select a preset example")
    print("="*80)

    examples = [
        {
            'name': "Stressed & Overwhelmed",
            'journal_text': "I feel so overwhelmed right now, everything is too much",
            'energy_level': 1,
            'stress_level': 5,
            'sleep_hours': 5.0,
            'time_of_day': 'evening'
        },
        {
            'name': "Calm & Focused Morning",
            'journal_text': "That forest session was peaceful. I feel clear and ready to work.",
            'energy_level': 4,
            'stress_level': 1,
            'sleep_hours': 8.0,
            'time_of_day': 'morning'
        },
        {
            'name': "Very Short Input",
            'journal_text': "ok",
            'energy_level': 3,
            'stress_level': 3,
            'sleep_hours': 6.5,
            'time_of_day': 'afternoon'
        },
        {
            'name': "Mixed Feelings",
            'journal_text': "I don't know... some parts felt good but I'm still anxious about work",
            'energy_level': 3,
            'stress_level': 4,
            'sleep_hours': 6.0,
            'time_of_day': 'evening'
        }
    ]

    print("\nChoose an example:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. {ex['name']}")

    choice = input("\nEnter number (1-4): ").strip()

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(examples):
            selected = examples[idx]

            # Add defaults
            user_input = {
                'journal_text': selected['journal_text'],
                'ambience_type': 'forest',
                'duration_min': 20,
                'sleep_hours': selected['sleep_hours'],
                'energy_level': selected['energy_level'],
                'stress_level': selected['stress_level'],
                'time_of_day': selected['time_of_day'],
                'previous_day_mood': 'neutral',
                'face_emotion_hint': 'neutral',
                'reflection_quality': 'clear'
            }

            print(f"\n✓ Selected: {selected['name']}")
            print(f'Text: "{selected["journal_text"]}"')

            return user_input
    except:
        pass

    print("Invalid choice. Using first example.")
    return None


def main():
    """Main interactive loop"""

    print_banner()

    print("\n📦 Loading trained models...")
    try:
        predictor = EmotionalStatePredictor()
        predictor.load_models('models')
        print("✓ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("\nPlease run 'python src/train_model.py' first to train the models.")
        return

    print("\n" + "="*80)
    print("DEMO MODE OPTIONS")
    print("="*80)
    print("\n1. Interactive Mode - Enter your own reflection")
    print("2. Quick Demo - Use preset examples")
    print("3. Exit")

    mode = input("\nChoose mode (1/2/3): ").strip()

    if mode == '3':
        print("\nGoodbye! 👋")
        return

    while True:
        print("\n")

        if mode == '2':
            user_input = quick_mode(predictor)
            if user_input is None:
                continue
        else:
            user_input = get_user_input()

        print("\n🔮 Processing your reflection...")

        try:
            predictions, action, timing = make_prediction(predictor, user_input)
            display_results(predictions, action, timing, user_input)

        except Exception as e:
            print(f"\n❌ Error making prediction: {e}")
            import traceback
            traceback.print_exc()

        # Ask to continue
        print("\n" + "="*80)
        again = input("\n🔄 Try another? (y/n): ").strip().lower()
        if again != 'y' and again != 'yes':
            break

    print("\n" + "="*80)
    print(" " * 25 + "Thank you for using ArvyaX! 🌿")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
