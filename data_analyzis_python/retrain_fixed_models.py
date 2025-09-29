#!/usr/bin/env python3
"""
Retrain models with proper input shapes for TensorFlow.js compatibility
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
    """Create model for predicting bicycle count with proper input shape"""
    print("Creating bike count model...")
    
    # Define input with explicit shape
    inputs = tf.keras.Input(shape=(input_shape,), name='input_features')
    
    x = tf.keras.layers.Dense(128, activation='relu', name='hidden1')(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu', name='hidden2')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu', name='hidden3')(x)
    
    # Output: bike count (non-negative)
    bike_count_output = tf.keras.layers.Dense(1, activation='relu', name='bike_count')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=bike_count_output, name='bike_count_model')
    
    # Use simple string for loss to avoid serialization issues
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_accident_rate_model(input_shape, output_name):
    """Create model for predicting accident rates per bike with proper input shape"""
    print(f"Creating {output_name} accident rate model...")
    
    # Define input with explicit shape
    inputs = tf.keras.Input(shape=(input_shape,), name='input_features')
    
    x = tf.keras.layers.Dense(64, activation='relu', name='hidden1')(inputs)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation='relu', name='hidden2')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(16, activation='relu', name='hidden3')(x)
    
    # Output: accident rate per bike (0-1 range, very small values expected)
    accident_rate_output = tf.keras.layers.Dense(1, activation='sigmoid', name=output_name)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=accident_rate_output, name=f'{output_name}_model')
    
    # Use simple string for loss to avoid serialization issues
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
    
    # Create and train model with explicit input shape
    model = create_bike_count_model(X_train_scaled.shape[1])
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=10,  # Only 10 epochs as requested
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
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]
    
    severe_model.fit(
        X_train_scaled, y_severe_train,
        validation_data=(X_test_scaled, y_severe_test),
        epochs=10,  # Only 10 epochs as requested
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
        epochs=10,  # Only 10 epochs as requested
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
        epochs=10,  # Only 10 epochs as requested
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

def save_models_tfjs_compatible(bike_count_model, accident_models, 
                               intersection_encoder, weather_encoder, temp_scaler, feature_scaler):
    """Save models in TensorFlow.js compatible format with proper input shapes"""
    print("Saving models in TensorFlow.js compatible format...")
    
    os.makedirs('models_tfjs_fixed', exist_ok=True)
    
    # Save bike count model with explicit input shape
    bike_count_model.save('models_tfjs_fixed/bike_count_model.keras')
    
    # Save accident rate models with explicit input shapes
    for name, model in accident_models.items():
        model.save(f'models_tfjs_fixed/{name}_accident_rate_model.keras')
    
    # Save preprocessors
    with open('models_tfjs_fixed/intersection_encoder.pkl', 'wb') as f:
        pickle.dump(intersection_encoder, f)
    
    with open('models_tfjs_fixed/weather_encoder.pkl', 'wb') as f:
        pickle.dump(weather_encoder, f)
    
    with open('models_tfjs_fixed/temp_scaler.pkl', 'wb') as f:
        pickle.dump(temp_scaler, f)
    
    with open('models_tfjs_fixed/feature_scaler.pkl', 'wb') as f:
        pickle.dump(feature_scaler, f)
    
    print("All models and preprocessors saved to models_tfjs_fixed/ directory")

def convert_to_tfjs():
    """Convert the .keras models to TensorFlow.js format"""
    print("Converting models to TensorFlow.js format...")
    
    import tensorflowjs as tfjs
    
    models = {
        "bike_count_model.keras": "bike_count",
        "severe_accident_rate_model.keras": "severe_accident_rate", 
        "moderate_accident_rate_model.keras": "moderate_accident_rate",
        "light_accident_rate_model.keras": "light_accident_rate"
    }
    
    for model_file, model_name in models.items():
        input_path = f"models_tfjs_fixed/{model_file}"
        output_path = f"models_tfjs_fixed/{model_name}_tfjs"
        
        if os.path.exists(input_path):
            print(f"Converting {model_file} to {model_name}_tfjs/")
            try:
                # Load the Keras model
                model = tf.keras.models.load_model(input_path)
                # Convert to TensorFlow.js format
                tfjs.converters.save_keras_model(model, output_path)
                print(f"✅ Successfully converted {model_file}")
            except Exception as e:
                print(f"❌ Error converting {model_file}: {e}")
        else:
            print(f"⚠️  Model file not found: {input_path}")

def create_intersection_coordinates_mapping():
    """Create mapping of intersection names to coordinates"""
    print("Creating intersection coordinates mapping...")
    
    # Load the crash data to get coordinates
    crashes_df = pd.read_csv('data/processed/bicycle_crashes_cleaned.csv')
    
    # Get top intersections from the model data
    df = pd.read_csv('data/processed/unified_final_dataset.csv')
    recent_data = df[df['datetime'] >= '2020-01-01'].copy()
    intersection_counts = recent_data['normalized_intersection'].value_counts()
    top_intersections = intersection_counts.head(10).index.tolist()
    
    # Create mapping
    intersection_coords = {}
    
    for intersection in top_intersections:
        # Find matching records in crash data - check if column exists
        if 'normalized_intersection' in crashes_df.columns:
            matching_records = crashes_df[crashes_df['normalized_intersection'] == intersection]
        else:
            # Fallback to original intersection column
            matching_records = crashes_df[crashes_df['Intersection'] == intersection]
        
        if len(matching_records) > 0:
            # Use median coordinates for the intersection
            lat = matching_records['Latitude'].median()
            lon = matching_records['Longitude'].median()
            intersection_coords[intersection] = {
                'latitude': float(lat),
                'longitude': float(lon),
                'count': len(matching_records)
            }
        else:
            # Fallback coordinates for Cambridge, MA center
            intersection_coords[intersection] = {
                'latitude': 42.3736,
                'longitude': -71.1097,
                'count': 0
            }
    
    # Save the mapping
    with open('models_tfjs_fixed/intersection_coordinates.pkl', 'wb') as f:
        pickle.dump(intersection_coords, f)
    
    # Also save as JSON for easy web access
    import json
    with open('models_tfjs_fixed/intersection_coordinates.json', 'w') as f:
        json.dump(intersection_coords, f, indent=2)
    
    print("Intersection coordinates mapping saved!")
    print("Intersection coordinates:")
    for intersection, coords in intersection_coords.items():
        print(f"  {intersection}: ({coords['latitude']:.4f}, {coords['longitude']:.4f}) - {coords['count']} accidents")
    
    return intersection_coords

def main():
    """Main function to run the retraining pipeline"""
    print("=== Retraining Models with Proper Input Shapes for TensorFlow.js ===")
    
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
    
    # Save everything in TF.js compatible format
    save_models_tfjs_compatible(
        bike_count_model, accident_models, intersection_encoder, weather_encoder, temp_scaler, feature_scaler
    )
    
    # Convert to TensorFlow.js format
    convert_to_tfjs()
    
    # Create intersection coordinates mapping
    intersection_coords = create_intersection_coordinates_mapping()
    
    print("\n=== Retraining Completed Successfully! ===")
    print("Models saved to models_tfjs_fixed/ directory")
    print("TensorFlow.js models ready for web deployment!")
    print(f"Input shape: {X_train.shape[1]} features")

if __name__ == "__main__":
    main()
