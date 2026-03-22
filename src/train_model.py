"""
ArvyaX ML Pipeline - Emotional State Understanding & Guidance System
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score, f1_score
import joblib
import re
import warnings
warnings.filterwarnings('ignore')

# Import NLTK for text processing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    # Download required NLTK data
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"NLTK import warning: {e}")


class EmotionalStatePredictor:
    """
    A comprehensive system for understanding emotional state and providing guidance.

    This system goes beyond prediction by:
    - Understanding emotional state from noisy, short text
    - Reasoning under uncertainty
    - Deciding meaningful next actions
    - Providing guidance with confidence awareness
    """

    def __init__(self):
        self.text_vectorizer = None
        self.state_encoder = None
        self.metadata_scaler = None

        # Models
        self.state_model = None  # For emotional_state prediction
        self.intensity_model = None  # For intensity prediction

        # Feature names
        self.text_feature_names = None
        self.metadata_feature_names = None

        # Encoding maps
        self.ambience_encoder = None
        self.time_encoder = None
        self.mood_encoder = None
        self.emotion_hint_encoder = None
        self.quality_encoder = None

    def preprocess_text(self, text):
        """Clean and preprocess journal text"""
        if pd.isna(text) or text == '':
            return "no_text"

        # Convert to lowercase
        text = str(text).lower()

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def extract_text_features(self, df, fit=False):
        """Extract TF-IDF features from journal text"""
        # Preprocess text
        texts = df['journal_text'].apply(self.preprocess_text)

        if fit:
            # Create TF-IDF vectorizer
            self.text_vectorizer = TfidfVectorizer(
                max_features=100,  # Keep model lightweight for edge deployment
                ngram_range=(1, 2),  # Unigrams and bigrams
                min_df=2,
                max_df=0.8,
                sublinear_tf=True
            )
            text_features = self.text_vectorizer.fit_transform(texts).toarray()
            self.text_feature_names = self.text_vectorizer.get_feature_names_out()
        else:
            text_features = self.text_vectorizer.transform(texts).toarray()

        return pd.DataFrame(
            text_features,
            columns=[f'text_{i}' for i in range(text_features.shape[1])],
            index=df.index
        )

    def extract_metadata_features(self, df, fit=False):
        """Extract and engineer features from metadata"""
        features = pd.DataFrame(index=df.index)

        # Numerical features - handle missing values
        features['duration_min'] = df['duration_min'].fillna(df['duration_min'].median())
        features['sleep_hours'] = df['sleep_hours'].fillna(df['sleep_hours'].median())
        features['energy_level'] = df['energy_level'].fillna(df['energy_level'].median())
        features['stress_level'] = df['stress_level'].fillna(df['stress_level'].median())

        # Categorical encoding
        if fit:
            self.ambience_encoder = LabelEncoder()
            self.time_encoder = LabelEncoder()
            self.mood_encoder = LabelEncoder()
            self.emotion_hint_encoder = LabelEncoder()
            self.quality_encoder = LabelEncoder()

            features['ambience_type'] = self.ambience_encoder.fit_transform(
                df['ambience_type'].fillna('unknown')
            )
            features['time_of_day'] = self.time_encoder.fit_transform(
                df['time_of_day'].fillna('unknown')
            )
            features['previous_day_mood'] = self.mood_encoder.fit_transform(
                df['previous_day_mood'].fillna('unknown')
            )
            features['face_emotion_hint'] = self.emotion_hint_encoder.fit_transform(
                df['face_emotion_hint'].fillna('unknown')
            )
            features['reflection_quality'] = self.quality_encoder.fit_transform(
                df['reflection_quality'].fillna('clear')
            )
        else:
            # Handle unseen labels gracefully
            features['ambience_type'] = df['ambience_type'].fillna('unknown').apply(
                lambda x: self.ambience_encoder.transform([x])[0] if x in self.ambience_encoder.classes_
                else self.ambience_encoder.transform(['unknown'])[0]
            )
            features['time_of_day'] = df['time_of_day'].fillna('unknown').apply(
                lambda x: self.time_encoder.transform([x])[0] if x in self.time_encoder.classes_
                else self.time_encoder.transform(['unknown'])[0]
            )
            features['previous_day_mood'] = df['previous_day_mood'].fillna('unknown').apply(
                lambda x: self.mood_encoder.transform([x])[0] if x in self.mood_encoder.classes_
                else self.mood_encoder.transform(['unknown'])[0]
            )
            features['face_emotion_hint'] = df['face_emotion_hint'].fillna('unknown').apply(
                lambda x: self.emotion_hint_encoder.transform([x])[0] if x in self.emotion_hint_encoder.classes_
                else self.emotion_hint_encoder.transform(['unknown'])[0]
            )
            features['reflection_quality'] = df['reflection_quality'].fillna('clear').apply(
                lambda x: self.quality_encoder.transform([x])[0] if x in self.quality_encoder.classes_
                else self.quality_encoder.transform([self.quality_encoder.classes_[0]])[0]
            )

        # Engineered features
        features['sleep_energy_ratio'] = features['sleep_hours'] / (features['energy_level'] + 1)
        features['stress_energy_gap'] = features['stress_level'] - features['energy_level']
        features['session_intensity'] = features['duration_min'] * features['energy_level']

        # Is user well-rested?
        features['well_rested'] = (features['sleep_hours'] >= 7).astype(int)

        # Is user exhausted?
        features['exhausted'] = ((features['energy_level'] <= 2) & (features['stress_level'] >= 4)).astype(int)

        # Text length features (indicates engagement)
        features['text_length'] = df['journal_text'].fillna('').apply(len)
        features['word_count'] = df['journal_text'].fillna('').apply(lambda x: len(str(x).split()))
        features['is_very_short'] = (features['word_count'] <= 3).astype(int)

        if fit:
            self.metadata_feature_names = features.columns.tolist()
            # Scale numerical features
            self.metadata_scaler = StandardScaler()
            features_scaled = self.metadata_scaler.fit_transform(features)
        else:
            features_scaled = self.metadata_scaler.transform(features)

        return pd.DataFrame(
            features_scaled,
            columns=self.metadata_feature_names,
            index=df.index
        )

    def train(self, train_df, use_text=True, use_metadata=True):
        """
        Train the emotional state and intensity models

        Args:
            train_df: Training dataframe
            use_text: Whether to use text features (for ablation study)
            use_metadata: Whether to use metadata features (for ablation study)
        """
        print("="*80)
        print("TRAINING EMOTIONAL STATE & INTENSITY MODELS")
        print("="*80)
        print(f"Using text features: {use_text}")
        print(f"Using metadata features: {use_metadata}")
        print()

        # Extract features
        feature_parts = []

        if use_text:
            text_features = self.extract_text_features(train_df, fit=True)
            text_features = text_features.reset_index(drop=True)
            feature_parts.append(text_features)
            print(f"Text features: {text_features.shape[1]}")

        if use_metadata:
            metadata_features = self.extract_metadata_features(train_df, fit=True)
            metadata_features = metadata_features.reset_index(drop=True)
            feature_parts.append(metadata_features)
            print(f"Metadata features: {metadata_features.shape[1]}")

        # Combine features
        X = pd.concat(feature_parts, axis=1, ignore_index=False)
        # CRITICAL: Ensure absolutely no numeric column names
        X.columns = [str(col).strip() for col in X.columns]
        # Verify no numeric-only columns
        problematic = [c for c in X.columns if c.isdigit()]
        if problematic:
            print(f"[WARNING] Found numeric column names: {problematic}")
            # Rename numeric columns to prevent XGBoost issues
            rename_map = {c: f'col_{c}' for c in problematic}
            X.rename(columns=rename_map, inplace=True)
        print(f"Total features: {X.shape[1]}")
        print()

        # Prepare labels
        self.state_encoder = LabelEncoder()
        y_state = self.state_encoder.fit_transform(train_df['emotional_state'])
        y_intensity = train_df['intensity'].values

        print(f"Emotional states: {self.state_encoder.classes_}")
        print(f"Training samples: {len(X)}")
        print()

        # Train emotional state model (classification)
        print("Training emotional state model...")
        self.state_model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss',
            validate_parameters=False  # Disable strict feature validation
        )
        self.state_model.fit(X, y_state)

        # Evaluate state model
        state_preds = self.state_model.predict(X)
        state_acc = accuracy_score(y_state, state_preds)
        print(f"State model training accuracy: {state_acc:.3f}")
        print()

        # Train intensity model (regression)
        print("Training intensity model...")
        self.intensity_model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            validate_parameters=False  # Disable strict feature validation
        )
        self.intensity_model.fit(X, y_intensity)

        # Evaluate intensity model
        intensity_preds = self.intensity_model.predict(X)
        intensity_rmse = np.sqrt(mean_squared_error(y_intensity, intensity_preds))
        print(f"Intensity model training RMSE: {intensity_rmse:.3f}")
        print()

        return X, y_state, y_intensity

    def predict_with_uncertainty(self, df, use_text=True, use_metadata=True):
        """
        Make predictions with uncertainty estimates

        Returns:
            DataFrame with predictions, confidence scores, and uncertainty flags
        """
        # Extract features (same as training)
        feature_parts = []

        if use_text:
            text_features = self.extract_text_features(df, fit=False)
            # Ensure text features have proper column names
            if not all(isinstance(c, str) for c in text_features.columns):
                text_features.columns = [str(c) for c in text_features.columns]
            feature_parts.append(text_features)

        if use_metadata:
            metadata_features = self.extract_metadata_features(df, fit=False)
            # Ensure metadata features have proper column names
            if not all(isinstance(c, str) for c in metadata_features.columns):
                metadata_features.columns = [str(c) for c in metadata_features.columns]
            feature_parts.append(metadata_features)

        X = pd.concat(feature_parts, axis=1)
        # CRITICAL: Ensure absolutely no numeric column names
        X.columns = [str(col).strip() for col in X.columns]
        # Verify no numeric-only columns  
        problematic = [c for c in X.columns if c.isdigit()]
        if problematic:
            print(f"[WARNING] Found numeric-only column names: {problematic}", flush=True)
            # Rename numeric columns to prevent XGBoost issues
            rename_map = {c: f'feat_{c}' for c in problematic}
            X.rename(columns=rename_map, inplace=True)

        # Predict emotional state
        state_probs = self.state_model.predict_proba(X)
        state_preds = self.state_model.predict(X)
        state_confidence = state_probs.max(axis=1)

        # Predict intensity
        intensity_preds = self.intensity_model.predict(X)
        # Clip to valid range [1, 5]
        intensity_preds = np.clip(intensity_preds, 1, 5)

        # Compute word count for uncertainty estimation
        word_counts = df['journal_text'].fillna('').apply(lambda x: len(str(x).split()))

        # Uncertainty flag: low confidence or ambiguous input
        uncertain_flag = (state_confidence < 0.6) | (word_counts <= 3)

        results = pd.DataFrame({
            'predicted_state': self.state_encoder.inverse_transform(state_preds),
            'predicted_intensity': intensity_preds.round().astype(int),
            'confidence': state_confidence,
            'uncertain_flag': uncertain_flag.astype(int),
            'state_probs': list(state_probs)
        }, index=df.index)

        # Add word count for decision engine
        results['word_count'] = word_counts

        return results, X

    def decide_action(self, predictions, metadata):
        """
        Decision Engine: Decide WHAT to do and WHEN to do it

        This is the core reasoning component that translates understanding into action.
        """
        actions = []
        timings = []

        for idx in predictions.index:
            state = predictions.loc[idx, 'predicted_state']
            intensity = predictions.loc[idx, 'predicted_intensity']
            confidence = predictions.loc[idx, 'confidence']

            # Get metadata
            stress = metadata.loc[idx, 'stress_level']
            energy = metadata.loc[idx, 'energy_level']
            sleep = metadata.loc[idx, 'sleep_hours']
            time_of_day = metadata.loc[idx, 'time_of_day']

            # Decision logic based on state, intensity, and context
            action, timing = self._decide_helper(
                state, intensity, confidence, stress, energy, sleep, time_of_day
            )

            actions.append(action)
            timings.append(timing)

        return actions, timings

    def _decide_helper(self, state, intensity, confidence, stress, energy, sleep, time_of_day):
        """
        Helper function to decide action and timing

        Logic:
        - High stress + low energy → immediate calming action
        - Positive state + high energy → productive action
        - Low confidence → gentler, exploratory action
        - Time of day matters for timing
        """

        # Map time strings to approximate hours
        time_map = {
            'morning': 8, 'afternoon': 14, 'evening': 18,
            'night': 22, 'late_night': 2, 'unknown': 12
        }
        current_hour = time_map.get(time_of_day, 12)

        # High stress situations - prioritize calming
        if stress >= 4:
            if energy <= 2:
                # Exhausted and stressed
                if current_hour >= 20:  # Evening/night
                    return 'rest', 'now'
                else:
                    return 'box_breathing', 'now'
            elif intensity >= 4:
                # High intensity stress
                return 'grounding', 'now'
            else:
                return 'box_breathing', 'within_15_min'

        # Low energy situations
        if energy <= 2:
            if sleep < 6:
                # Sleep deprived
                if current_hour >= 20:
                    return 'rest', 'tonight'
                else:
                    return 'power_nap', 'within_15_min'
            else:
                # Low energy despite sleep
                return 'movement', 'within_15_min'

        # Positive states with good energy
        if state in ['calm', 'energized', 'focused', 'content', 'peaceful']:
            if energy >= 4:
                if current_hour < 12:
                    return 'deep_work', 'now'
                elif current_hour < 18:
                    return 'deep_work', 'within_15_min'
                else:
                    return 'light_planning', 'later_today'
            else:
                return 'journaling', 'within_15_min'

        # Negative/challenging states
        if state in ['anxious', 'restless', 'overwhelmed', 'stressed']:
            if intensity >= 4:
                return 'grounding', 'now'
            elif confidence < 0.5:
                # Uncertain - gentle approach
                return 'journaling', 'within_15_min'
            else:
                return 'box_breathing', 'within_15_min'

        if state in ['sad', 'tired', 'disconnected', 'lonely']:
            if intensity >= 4:
                if current_hour >= 20:
                    return 'sound_therapy', 'tonight'
                else:
                    return 'movement', 'within_15_min'
            else:
                return 'journaling', 'within_15_min'

        # Neutral or unclear states
        if confidence < 0.5:
            # Low confidence - encourage reflection
            return 'journaling', 'within_15_min'

        # Morning time
        if current_hour < 10:
            if energy >= 3:
                return 'light_planning', 'now'
            else:
                return 'movement', 'within_15_min'

        # Evening/night
        if current_hour >= 20:
            if stress >= 3:
                return 'sound_therapy', 'tonight'
            else:
                return 'light_planning', 'tonight'

        # Default: gentle, adaptive action
        return 'pause', 'within_15_min'

    def save_models(self, output_dir='models'):
        """Save all models and encoders"""
        joblib.dump(self.state_model, f'{output_dir}/state_model.pkl')
        joblib.dump(self.intensity_model, f'{output_dir}/intensity_model.pkl')
        joblib.dump(self.text_vectorizer, f'{output_dir}/text_vectorizer.pkl')
        joblib.dump(self.metadata_scaler, f'{output_dir}/metadata_scaler.pkl')
        joblib.dump(self.state_encoder, f'{output_dir}/state_encoder.pkl')

        # Save categorical encoders
        encoders = {
            'ambience': self.ambience_encoder,
            'time': self.time_encoder,
            'mood': self.mood_encoder,
            'emotion_hint': self.emotion_hint_encoder,
            'quality': self.quality_encoder
        }
        joblib.dump(encoders, f'{output_dir}/categorical_encoders.pkl')
        
        # Save metadata feature names (CRITICAL for prediction)
        joblib.dump(self.metadata_feature_names, f'{output_dir}/metadata_feature_names.pkl')

        print(f"Models saved to {output_dir}/")

    def load_models(self, output_dir='models'):
        """Load all models and encoders"""
        self.state_model = joblib.load(f'{output_dir}/state_model.pkl')
        self.intensity_model = joblib.load(f'{output_dir}/intensity_model.pkl')
        
        # Disable XGBoost feature name validation to avoid mismatches during deployment
        # This allows flexible feature handling between training and inference
        self.state_model.get_booster().set_param({"validate_parameters": 0})
        self.intensity_model.get_booster().set_param({"validate_parameters": 0})
        
        self.text_vectorizer = joblib.load(f'{output_dir}/text_vectorizer.pkl')
        self.metadata_scaler = joblib.load(f'{output_dir}/metadata_scaler.pkl')
        self.state_encoder = joblib.load(f'{output_dir}/state_encoder.pkl')

        encoders = joblib.load(f'{output_dir}/categorical_encoders.pkl')
        self.ambience_encoder = encoders['ambience']
        self.time_encoder = encoders['time']
        self.mood_encoder = encoders['mood']
        self.emotion_hint_encoder = encoders['emotion_hint']
        self.quality_encoder = encoders['quality']
        
        # Load metadata feature names (CRITICAL for prediction)
        try:
            self.metadata_feature_names = joblib.load(f'{output_dir}/metadata_feature_names.pkl')
        except FileNotFoundError:
            print("[WARNING] metadata_feature_names.pkl not found - prediction may fail")

        print(f"Models loaded from {output_dir}/")


def main():
    """Main training and evaluation pipeline"""
    print("ArvyaX Emotional State Prediction System")
    print("="*80)
    print()

    # Load data
    print("Loading data...")
    train_df = pd.read_excel('Sample_arvyax_reflective_dataset.xlsx')
    test_df = pd.read_excel('arvyax_test_inputs_120.xlsx')

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print()

    # Add word count to test_df for decision engine
    test_df['word_count'] = test_df['journal_text'].fillna('').apply(lambda x: len(str(x).split()))

    # Initialize predictor
    predictor = EmotionalStatePredictor()

    # Train with text + metadata
    print("\n### Training with TEXT + METADATA ###")
    X_train, y_state, y_intensity = predictor.train(train_df, use_text=True, use_metadata=True)

    # Save models
    predictor.save_models('models')

    # Make predictions on test set
    print("\n" + "="*80)
    print("GENERATING TEST PREDICTIONS")
    print("="*80)
    predictions, X_test = predictor.predict_with_uncertainty(test_df, use_text=True, use_metadata=True)

    # Decision engine
    print("\nRunning decision engine...")
    actions, timings = predictor.decide_action(predictions, test_df)

    # Create final output
    output_df = pd.DataFrame({
        'id': test_df['id'],
        'predicted_state': predictions['predicted_state'],
        'predicted_intensity': predictions['predicted_intensity'],
        'confidence': predictions['confidence'],
        'uncertain_flag': predictions['uncertain_flag'],
        'what_to_do': actions,
        'when_to_do': timings
    })

    # Save predictions
    output_df.to_csv('outputs/predictions.csv', index=False)
    print(f"\nPredictions saved to outputs/predictions.csv")
    print(f"Total predictions: {len(output_df)}")
    print()

    # Show sample predictions
    print("Sample predictions:")
    print(output_df.head(10))

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
