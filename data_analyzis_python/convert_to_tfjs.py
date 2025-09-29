#!/usr/bin/env python3
"""
Convert TensorFlow models to TensorFlow.js format for web deployment
"""

import os
import tensorflow as tf
import tensorflowjs as tfjs

def convert_models():
    """Convert all models to TensorFlow.js format"""
    print("=== Converting Models to TensorFlow.js Format ===")
    
    # Create output directory
    output_dir = "models_tfjs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Model mappings
    models = {
        "bike_count_model.h5": "bike_count",
        "severe_accident_rate_model.h5": "severe_accident_rate", 
        "moderate_accident_rate_model.h5": "moderate_accident_rate",
        "light_accident_rate_model.h5": "light_accident_rate"
    }
    
    for model_file, model_name in models.items():
        input_path = f"models_improved/{model_file}"
        output_path = f"{output_dir}/{model_name}"
        
        if os.path.exists(input_path):
            print(f"Converting {model_file} to {model_name}/")
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
    
    print(f"\n=== Conversion Complete ===")
    print(f"TensorFlow.js models saved to: {output_dir}/")
    
    # List the created files
    print("\nCreated files:")
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"  {file_path} ({file_size:,} bytes)")

if __name__ == "__main__":
    convert_models()
