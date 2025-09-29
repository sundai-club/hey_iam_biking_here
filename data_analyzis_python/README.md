# Cambridge Bikes Prediction Model

A machine learning system that predicts bicycle traffic and accident counts at intersections in Cambridge, MA using a two-stage neural network approach.

## 🎯 Project Overview

This project addresses the challenge of predicting both bicycle traffic volume and accident counts at Cambridge intersections. The system uses a novel two-stage approach to handle the sparse nature of accident data while providing meaningful predictions for traffic planning and safety assessment.

### Model Outputs
The system predicts **4 separate outputs**:
1. **Bike count** (float) - Number of bicycles passing the intersection
2. **Severe accidents count** (float) - Number of severe accidents
3. **Moderate accidents count** (float) - Number of moderate accidents  
4. **Light accidents count** (float) - Number of light accidents

## 📊 Input Data Sources

The model was trained on three unified datasets from Cambridge, MA:

### 1. Bicycle Crash Data (`data/processed/bicycle_crashes_cleaned.csv`)
- **Source**: Cambridge Police Department crash reports
- **Records**: 552 bicycle accidents (2002-2025)
- **Key Fields**: Date/time, intersection location, weather conditions, injury severity
- **Accident Severity Mapping**:
  - Light accidents (0.1): No apparent injury
  - Moderate accidents (0.5): Minor/suspected minor injury
  - Severe accidents (1.0): Fatal/serious/suspected serious injury

### 2. City Bike Count Data (`data/city_bike_count.csv`)
- **Source**: Cambridge citywide bicycle counts
- **Records**: 238,799 manual counts (2002-2019)
- **Collection**: Weekdays (Tuesday-Thursday) in September, annual counts
- **Key Fields**: Date/time, intersection, weather, temperature, bike count
- **Bias**: Heavily weighted toward September data

### 3. Eco-Totem Automatic Counter (`data/eco_totem.csv`)
- **Source**: Automatic in-ground loop detectors on Broadway
- **Records**: 306,854 15-minute interval counts (2019-2024)
- **Location**: Broadway near Kendall Square
- **Key Fields**: DateTime, total bike count
- **Advantage**: Captures seasonal and day/night fluctuations

## 🏗️ Model Architecture

### Two-Stage Approach

**Stage 1: Bike Count Model**
- **Purpose**: Predict bicycle traffic volume
- **Architecture**: 128 → 64 → 32 neurons with ReLU activation
- **Dropout**: 0.3 between layers
- **Output**: Non-negative bike count (ReLU activation)
- **Loss**: MSE, Optimizer: Adam

**Stage 2: Accident Rate Models (3 separate models)**
- **Purpose**: Predict accident rates per bike for each severity level
- **Architecture**: 64 → 32 → 16 neurons with ReLU activation
- **Dropout**: 0.4 between layers (higher for sparse data)
- **Output**: Accident rate per bike (Sigmoid activation, 0-1 range)
- **Final Prediction**: `bike_count × accident_rate = accident_count`

### Input Features (10 total)
1. `time_15min` - Time in 15-minute increments (0-95)
2. `intersection_encoded` - Encoded intersection name
3. `weather_encoded` - Encoded weather condition
4. `hour_sin/cos` - Cyclical hour encoding
5. `day_sin/cos` - Cyclical day of week encoding
6. `month_sin/cos` - Cyclical month encoding
7. `temperature_normalized` - Standardized temperature

## 📈 Model Performance

### Training Results
- **Total Training Records**: 311,504 (filtered to 2020+ and top 10 intersections)
- **Training/Test Split**: 80/20
- **Training Epochs**: 100 (bike count), 200 (accident models)

### Performance Metrics

**Bike Count Model:**
- **MAE**: ~2.7 bikes
- **MSE**: Low (good fit)
- **R²**: High (explains most variance in bike traffic)

**Accident Rate Models:**
- **Severe Accidents**: MAE ~0.000007 (very small rates as expected)
- **Moderate Accidents**: MAE ~0.011 (small but measurable rates)
- **Light Accidents**: MAE ~0.005 (small but measurable rates)

### Data Insights
- **Total Accidents**: 552 out of 311,504 records (0.18% accident rate)
- **Accident Distribution**: 279 light, 145 moderate, 128 severe
- **Peak Accident Hours**: 9:00 AM (55), 5:00 PM (50), 11:00 AM (45)
- **Top Accident Intersections**: Massachusetts Ave (137), Cambridge St (35), Broadway (29)

## 🚀 Usage Instructions

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Activate environment (if using conda)
conda activate cambridge_bikes
```

### Model Files
The trained models are stored in `models_improved/`:
```
models_improved/
├── bike_count_model.h5                    # Stage 1: Bike count prediction
├── severe_accident_rate_model.h5          # Stage 2: Severe accident rates
├── moderate_accident_rate_model.h5        # Stage 2: Moderate accident rates
├── light_accident_rate_model.h5           # Stage 2: Light accident rates
├── intersection_encoder.pkl               # Intersection name encoder
├── weather_encoder.pkl                    # Weather condition encoder
├── temp_scaler.pkl                        # Temperature normalizer
├── feature_scaler.pkl                     # Feature normalizer
└── prediction_function.py                 # Complete prediction function
```

### Python Usage
```python
from models_improved.prediction_function import predict_bicycle_traffic_and_accidents

# Example prediction
bike_count, severe_accidents, moderate_accidents, light_accidents = predict_bicycle_traffic_and_accidents(
    intersection="BROADWAY",
    time_15min=32,  # 8:00 AM (rush hour)
    weather="CLEAR",
    temperature=70,
    day_of_week=1,  # Tuesday
    month=6  # June
)

print(f"Predicted bike count: {bike_count:.1f}")
print(f"Severe accidents: {severe_accidents:.4f}")
print(f"Moderate accidents: {moderate_accidents:.4f}")
print(f"Light accidents: {light_accidents:.4f}")
```

### TensorFlow.js Conversion
For browser deployment:

```bash
# Convert models to TensorFlow.js format
tensorflowjs_converter --input_format keras models_improved/bike_count_model.h5 models_js/bike_count/
tensorflowjs_converter --input_format keras models_improved/severe_accident_rate_model.h5 models_js/severe/
tensorflowjs_converter --input_format keras models_improved/moderate_accident_rate_model.h5 models_js/moderate/
tensorflowjs_converter --input_format keras models_improved/light_accident_rate_model.h5 models_js/light/
```

Browser usage:
```javascript
// Load models
const bikeModel = await tf.loadLayersModel('models_js/bike_count/model.json');
const severeModel = await tf.loadLayersModel('models_js/severe/model.json');
const moderateModel = await tf.loadLayersModel('models_js/moderate/model.json');
const lightModel = await tf.loadLayersModel('models_js/light/model.json');

// Make predictions (with same preprocessing pipeline)
const bikeCount = bikeModel.predict(features);
const severeRate = severeModel.predict(features);
const moderateRate = moderateModel.predict(features);
const lightRate = lightModel.predict(features);

// Calculate final counts
const severeAccidents = bikeCount.mul(severeRate);
const moderateAccidents = bikeCount.mul(moderateRate);
const lightAccidents = bikeCount.mul(lightRate);
```

## 🔧 Data Processing Pipeline

### Data Unification
1. **Intersection Name Normalization**: Standardized formats (e.g., "STREET1_AND_STREET2" → "STREET1 & STREET2")
2. **Weather Standardization**: Mapped to 5 categories (CLEAR, CLOUDY, RAIN, SNOW, UNKNOWN)
3. **Time Feature Engineering**: Created cyclical sine/cosine features for hour, day, month
4. **Missing Data Handling**: Filled Eco-Totem weather data from closest city count records

### Feature Engineering
- **Cyclical Encoding**: Hour, day of week, and month as sine/cosine pairs
- **Time Intervals**: 15-minute time slots (0-95)
- **Categorical Encoding**: Label encoding for intersections and weather
- **Normalization**: Standard scaling for temperature and features

## 📁 Project Structure

```
cambridge_bikes/
├── data/                                    # Raw and processed datasets
│   ├── processed/
│   │   ├── bicycle_crashes_cleaned.csv     # Cleaned crash data
│   │   └── unified_final_dataset.csv       # Final unified dataset
│   ├── city_bike_count.csv                 # City manual counts
│   └── eco_totem.csv                       # Automatic counter data
├── models_improved/                         # Trained models and preprocessors
├── train_improved_model.py                 # Model training script
├── test_improved_model.py                  # Model testing script
├── improved_model_summary.py               # Model analysis and summary
├── save_final_dataset.py                   # Data processing utility
├── README.md                               # This file
└── requirements.txt                        # Python dependencies
```

## 🎯 Key Innovations

1. **Two-Stage Architecture**: Separates traffic prediction from accident prediction
2. **Rate-Based Approach**: Predicts accident rates per bike rather than absolute counts
3. **Sparse Data Handling**: Uses higher dropout and specialized training for rare events
4. **Multi-Output Design**: Provides granular accident severity breakdown
5. **Traffic-Aware Predictions**: Accounts for the relationship between bike volume and accident risk

## 📊 Business Applications

- **Traffic Planning**: Predict bike volumes for infrastructure planning
- **Safety Assessment**: Identify high-risk intersections and time periods
- **Resource Allocation**: Optimize police patrols and safety measures
- **Policy Evaluation**: Assess impact of bike infrastructure changes
- **Real-time Monitoring**: Deploy in web applications for live predictions

## 🔬 Technical Details

- **Framework**: TensorFlow/Keras
- **Language**: Python 3.11
- **Dependencies**: pandas, numpy, scikit-learn, tensorflow
- **Model Format**: HDF5 (.h5) for Python, JSON/Binary for TensorFlow.js
- **Preprocessing**: Label encoding, standard scaling, cyclical encoding

## 📝 License

This project is part of Cambridge's open data initiative and follows the city's data usage policies.

## 🤝 Contributing

This model was developed as part of Cambridge's bicycle safety and traffic analysis efforts. For improvements or questions, please refer to the Cambridge Open Data Portal.

---

**Note**: This model is designed for research and planning purposes. Always consult with traffic engineering professionals for critical safety decisions.
