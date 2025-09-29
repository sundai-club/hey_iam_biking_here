#!/usr/bin/env python3
"""
Improved Cambridge Bikes Model Training Script

This script trains a two-stage model:
1. Stage 1: Predict bicycle count
2. Stage 2: Predict accident rates per bike for each severity level

Final outputs:
1. Bike count (float)
2. Severe accidents count (float) 
3. Moderate accidents count (float)
4. Light accidents count (float)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def normalize_intersection_name(name):
    """Normalize intersection names to a common format"""
    if pd.isna(name):
        return None
    name = str(name).upper().strip()
    if '_AND_' in name:
        parts = name.split('_AND_')
        if len(parts) == 2:
            return f"{parts[0].strip()} & {parts[1].strip()}"
    elif ' & ' in name:
        return name
    return name

def normalize_weather_condition(weather):
    """Normalize weather conditions to standard categories"""
    if pd.isna(weather):
        return 'UNKNOWN'
    weather = str(weather).upper().strip()
    if any(word in weather for word in ['CLEAR', 'SUNNY']):
        return 'CLEAR'
    elif any(word in weather for word in ['CLOUDY', 'OVERCAST', 'BROKEN CLOUDS', 'SCATTERED CLOUDS', 'FEW CLOUDS', 'PARTIALLY CLOUDY', 'PARTLY CLOUDY', 'MOSTLY CLOUDY']):
        return 'CLOUDY'
    elif any(word in weather for word in ['RAIN', 'DRIZZLE']):
        return 'RAIN'
    elif any(word in weather for word in ['SNOW', 'SLEET', 'HAIL', 'FREEZING']):
        return 'SNOW'
    else:
        return 'UNKNOWN'

def get_accident_counts(row):
    """Get accident counts by severity for each record"""
    if row['accident_severity'] == 1.0:
        return 1, 0, 0  # severe, moderate, light
    elif row['accident_severity'] == 0.5:
        return 0, 1, 0
    elif row['accident_severity'] == 0.1:
        return 0, 0, 1
    else:
        return 0, 0, 0

def load_and_process_data():
    """Load and process the unified dataset"""
    print("Loading unified dataset...")
    
    df = pd.read_csv('data/processed/unified_final_dataset.csv')
    print(f"Loaded dataset: {df.shape}")
    
    # Focus on recent data and top intersections
    recent_data = df[df['datetime'] >= '2020-01-01'].copy()
    
    # Get top intersections by data volume
    intersection_counts = recent_data['normalized_intersection'].value_counts()
    top_intersections = intersection_counts.head(10).index.tolist()
    print(f"Top intersections for model: {top_intersections}")
    
    # Filter to top intersections
    model_data = recent_data[recent_data['normalized_intersection'].isin(top_intersections)].copy()
    
    # Create accident count columns
    accident_counts = model_data.apply(get_accident_counts, axis=1, result_type='expand')
    model_data['severe_accidents'] = accident_counts[0]
    model_data['moderate_accidents'] = accident_counts[1]
    model_data['light_accidents'] = accident_counts[2]
    
    print(f"Model data shape: {model_data.shape}")
    print(f"Records with bike counts > 0: {len(model_data[model_data['Count'] > 0])}")
    print(f"Records with severe accidents: {len(model_data[model_data['severe_accidents'] > 0])}")
    print(f"Records with moderate accidents: {len(model_data[model_data['moderate_accidents'] > 0])}")
    print(f"Records with light accidents: {len(model_data[model_data['light_accidents'] > 0])}")
    
    return model_data

def prepare_features(model_data):
    """Prepare features for model training"""
    print("Preparing features...")
    
    # Encode categorical variables
    intersection_encoder = LabelEncoder()
    model_data['intersection_encoded'] = intersection_encoder.fit_transform(model_data['normalized_intersection'])
    
    weather_encoder = LabelEncoder()
    model_data['weather_encoded'] = weather_encoder.fit_transform(model_data['normalized_weather'])
    
    # Create additional time features
    model_data['hour_sin'] = np.sin(2 * np.pi * model_data['hour'] / 24)
    model_data['hour_cos'] = np.cos(2 * np.pi * model_data['hour'] / 24)
    model_data['day_sin'] = np.sin(2 * np.pi * model_data['day_of_week'] / 7)
    model_data['day_cos'] = np.cos(2 * np.pi * model_data['day_of_week'] / 7)
    model_data['month_sin'] = np.sin(2 * np.pi * model_data['month'] / 12)
    model_data['month_cos'] = np.cos(2 * np.pi * model_data['month'] / 12)
    
    # Normalize temperature
    temp_scaler = StandardScaler()
    model_data['temperature_normalized'] = temp_scaler.fit_transform(model_data[['Temperature']])
    
    return model_data, intersection_encoder, weather_encoder, temp_scaler

def create_bike_count_model(input_shape):
    """Create model for predicting bicycle count"""
    print("Creating bike count model...")
    
    inputs = tf.keras.Input(shape=(input_shape,), name='features')
    
    x = tf.keras.layers.Dense(128, activation='relu', name='hidden1')(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu', name='hidden2')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu', name='hidden3')(x)
    
    # Output: bike count (non-negative)
    bike_count_output = tf.keras.layers.Dense(1, activation='relu', name='bike_count')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=bike_count_output, name='bike_count_model')
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_accident_rate_model(input_shape, output_name):
    """Create model for predicting accident rates per bike"""
    print(f"Creating {output_name} accident rate model...")
    
    inputs = tf.keras.Input(shape=(input_shape,), name='features')
    
    x = tf.keras.layers.Dense(64, activation='relu', name='hidden1')(inputs)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation='relu', name='hidden2')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(16, activation='relu', name='hidden3')(x)
    
    # Output: accident rate per bike (0-1 range, very small values expected)
    accident_rate_output = tf.keras.layers.Dense(1, activation='sigmoid', name=output_name)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=accident_rate_output, name=f'{output_name}_model')
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_bike_count_model(X_train, X_test, y_train, y_test, feature_scaler):
    """Train the bike count model"""
    print("=== Training Bike Count Model ===")
    
    # Normalize features
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Create and train model
    model = create_bike_count_model(X_train_scaled.shape[1])
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        batch_size=512,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)
    predictions = model.predict(X_test_scaled, verbose=0)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Bike Count Model Performance:")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  R²: {r2:.3f}")
    
    return model, feature_scaler

def train_accident_rate_models(X_train, X_test, y_severe_train, y_severe_test, 
                              y_moderate_train, y_moderate_test,
                              y_light_train, y_light_test, feature_scaler):
    """Train accident rate models"""
    print("=== Training Accident Rate Models ===")
    
    # Normalize features
    X_train_scaled = feature_scaler.transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    models = {}
    
    # Train severe accident rate model
    print("Training severe accident rate model...")
    severe_model = create_accident_rate_model(X_train_scaled.shape[1], 'severe_rate')
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7)
    ]
    
    severe_model.fit(
        X_train_scaled, y_severe_train,
        validation_data=(X_test_scaled, y_severe_test),
        epochs=200,
        batch_size=1024,
        callbacks=callbacks,
        verbose=1
    )
    
    # Train moderate accident rate model
    print("Training moderate accident rate model...")
    moderate_model = create_accident_rate_model(X_train_scaled.shape[1], 'moderate_rate')
    
    moderate_model.fit(
        X_train_scaled, y_moderate_train,
        validation_data=(X_test_scaled, y_moderate_test),
        epochs=200,
        batch_size=1024,
        callbacks=callbacks,
        verbose=1
    )
    
    # Train light accident rate model
    print("Training light accident rate model...")
    light_model = create_accident_rate_model(X_train_scaled.shape[1], 'light_rate')
    
    light_model.fit(
        X_train_scaled, y_light_train,
        validation_data=(X_test_scaled, y_light_test),
        epochs=200,
        batch_size=1024,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate models
    for name, model, y_true in [('severe', severe_model, y_severe_test),
                               ('moderate', moderate_model, y_moderate_test),
                               ('light', light_model, y_light_test)]:
        predictions = model.predict(X_test_scaled, verbose=0)
        mae = mean_absolute_error(y_true, predictions)
        print(f"{name.capitalize()} Accident Rate Model MAE: {mae:.6f}")
    
    models['severe'] = severe_model
    models['moderate'] = moderate_model
    models['light'] = light_model
    
    return models

def save_models_and_preprocessors(bike_count_model, accident_models, 
                                 intersection_encoder, weather_encoder, temp_scaler, feature_scaler):
    """Save all models and preprocessors"""
    print("Saving models and preprocessors...")
    
    os.makedirs('models_improved', exist_ok=True)
    
    # Save bike count model
    bike_count_model.save('models_improved/bike_count_model.h5')
    
    # Save accident rate models
    for name, model in accident_models.items():
        model.save(f'models_improved/{name}_accident_rate_model.h5')
    
    # Save preprocessors
    with open('models_improved/intersection_encoder.pkl', 'wb') as f:
        pickle.dump(intersection_encoder, f)
    
    with open('models_improved/weather_encoder.pkl', 'wb') as f:
        pickle.dump(weather_encoder, f)
    
    with open('models_improved/temp_scaler.pkl', 'wb') as f:
        pickle.dump(temp_scaler, f)
    
    with open('models_improved/feature_scaler.pkl', 'wb') as f:
        pickle.dump(feature_scaler, f)
    
    print("All models and preprocessors saved to models_improved/ directory")

def create_prediction_function():
    """Create a prediction function for the improved model"""
    prediction_code = '''
def predict_bicycle_traffic_and_accidents(intersection, time_15min, weather, temperature, day_of_week, month):
    """
    Predict bicycle count and accident counts for given conditions
    
    Args:
        intersection: Intersection name (string)
        time_15min: Time in 15-minute increments (0-95)
        weather: Weather condition ('CLEAR', 'CLOUDY', 'RAIN', 'SNOW')
        temperature: Temperature in Fahrenheit
        day_of_week: Day of week (0=Monday, 6=Sunday)
        month: Month (1-12)
    
    Returns:
        tuple: (bike_count, severe_accidents, moderate_accidents, light_accidents)
    """
    import tensorflow as tf
    import numpy as np
    import pickle
    
    # Load models
    bike_count_model = tf.keras.models.load_model('models_improved/bike_count_model.h5')
    severe_model = tf.keras.models.load_model('models_improved/severe_accident_rate_model.h5')
    moderate_model = tf.keras.models.load_model('models_improved/moderate_accident_rate_model.h5')
    light_model = tf.keras.models.load_model('models_improved/light_accident_rate_model.h5')
    
    # Load preprocessors
    with open('models_improved/intersection_encoder.pkl', 'rb') as f:
        intersection_encoder = pickle.load(f)
    with open('models_improved/weather_encoder.pkl', 'rb') as f:
        weather_encoder = pickle.load(f)
    with open('models_improved/temp_scaler.pkl', 'rb') as f:
        temp_scaler = pickle.load(f)
    with open('models_improved/feature_scaler.pkl', 'rb') as f:
        feature_scaler = pickle.load(f)
    
    # Encode inputs
    intersection_encoded = intersection_encoder.transform([intersection])[0]
    weather_encoded = weather_encoder.transform([weather])[0]
    
    # Create time features
    hour = time_15min // 4
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Normalize temperature
    temp_normalized = temp_scaler.transform([[temperature]])[0][0]
    
    # Create feature vector
    features = np.array([[
        time_15min, intersection_encoded, weather_encoded,
        hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos,
        temp_normalized
    ]])
    
    # Scale features
    features_scaled = feature_scaler.transform(features)
    
    # Make predictions
    bike_count = bike_count_model.predict(features_scaled, verbose=0)[0][0]
    
    severe_rate = severe_model.predict(features_scaled, verbose=0)[0][0]
    moderate_rate = moderate_model.predict(features_scaled, verbose=0)[0][0]
    light_rate = light_model.predict(features_scaled, verbose=0)[0][0]
    
    # Calculate accident counts
    severe_accidents = bike_count * severe_rate
    moderate_accidents = bike_count * moderate_rate
    light_accidents = bike_count * light_rate
    
    return bike_count, severe_accidents, moderate_accidents, light_accidents
'''
    
    with open('models_improved/prediction_function.py', 'w') as f:
        f.write(prediction_code)
    
    print("Saved prediction function to models_improved/prediction_function.py")

def main():
    """Main function to run the improved training pipeline"""
    print("=== Improved Cambridge Bikes Model Training Pipeline ===")
    
    # Load and process data
    model_data = load_and_process_data()
    
    # Prepare features
    model_data, intersection_encoder, weather_encoder, temp_scaler = prepare_features(model_data)
    
    # Prepare features and targets
    feature_columns = [
        'time_15min', 'intersection_encoded', 'weather_encoded', 
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'temperature_normalized'
    ]
    
    X = model_data[feature_columns].values
    y_bike_count = model_data['Count'].values
    y_severe = model_data['severe_accidents'].values
    y_moderate = model_data['moderate_accidents'].values
    y_light = model_data['light_accidents'].values
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_bike_train, y_bike_test, y_severe_train, y_severe_test, y_moderate_train, y_moderate_test, y_light_train, y_light_test = train_test_split(
        X, y_bike_count, y_severe, y_moderate, y_light, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train bike count model
    bike_count_model, feature_scaler = train_bike_count_model(
        X_train, X_test, y_bike_train, y_bike_test, StandardScaler()
    )
    
    # Train accident rate models
    accident_models = train_accident_rate_models(
        X_train, X_test, y_severe_train, y_severe_test,
        y_moderate_train, y_moderate_test, y_light_train, y_light_test, feature_scaler
    )
    
    # Save everything
    save_models_and_preprocessors(
        bike_count_model, accident_models, intersection_encoder, weather_encoder, temp_scaler, feature_scaler
    )
    
    # Create prediction function
    create_prediction_function()
    
    print("\n=== Improved Pipeline Completed Successfully! ===")
    print("Models saved to models_improved/ directory")
    print("You can now predict:")
    print("1. Bike count (float)")
    print("2. Severe accidents count (float)")
    print("3. Moderate accidents count (float)")
    print("4. Light accidents count (float)")

if __name__ == "__main__":
    main()
