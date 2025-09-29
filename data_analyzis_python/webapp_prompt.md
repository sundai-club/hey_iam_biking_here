# Cambridge Bicycle Traffic & Safety Prediction System - Web App Development Prompt

## Project Overview

You are building a web application that predicts bicycle traffic and accident rates for intersections in Cambridge, MA using a sophisticated two-stage machine learning system. The system uses 4 separate TensorFlow.js models to predict:

1. **Bicycle Count**: How many cyclists will pass through an intersection
2. **Severe Accidents**: Number of severe accidents (hospitalization/fatality)
3. **Moderate Accidents**: Number of moderate accidents (injury requiring medical attention)
4. **Light Accidents**: Number of light accidents (minor injury/property damage)

## Model Architecture

### Two-Stage Approach
The system uses a two-stage prediction approach to handle sparse accident data:

**Stage 1: Bike Count Prediction**
- Predicts the total number of cyclists passing through an intersection
- Uses ReLU activation to ensure non-negative predictions
- Input: 10 features (time, intersection, weather, temperature, cyclical features)
- Output: Single float value (bike count)

**Stage 2: Accident Rate Prediction**
- Three separate models predict accident rates per cyclist for each severity level
- Uses Sigmoid activation to keep rates between 0 and 1
- Final accident count = bike_count × accident_rate
- Input: Same 10 features as bike count model
- Output: Single float value (accident rate per bike)

## Input Features (10 total)

The models expect exactly 10 input features in this order:

1. **time_15min** (float): Time of day in 15-minute increments (0-95, where 0=00:00, 1=00:15, etc.)
2. **intersection_encoded** (int): Encoded intersection name (0-9 for top 10 intersections)
3. **weather_encoded** (int): Encoded weather condition (0-4: CLEAR, CLOUDY, RAIN, SNOW, UNKNOWN)
4. **hour_sin** (float): Cyclical encoding of hour (sin(2π × hour / 24))
5. **hour_cos** (float): Cyclical encoding of hour (cos(2π × hour / 24))
6. **day_sin** (float): Cyclical encoding of day of week (sin(2π × day / 7))
7. **day_cos** (float): Cyclical encoding of day of week (cos(2π × day / 7))
8. **month_sin** (float): Cyclical encoding of month (sin(2π × month / 12))
9. **month_cos** (float): Cyclical encoding of month (cos(2π × month / 12))
10. **temperature_normalized** (float): Standardized temperature (mean=0, std=1)

## Model Files Structure

```
models_tfjs_fixed/
├── bike_count_tfjs/           # TensorFlow.js bike count model
│   ├── model.json
│   └── group1-shard1of1.bin
├── severe_accident_rate_tfjs/ # TensorFlow.js severe accident rate model
│   ├── model.json
│   └── group1-shard1of1.bin
├── moderate_accident_rate_tfjs/ # TensorFlow.js moderate accident rate model
│   ├── model.json
│   └── group1-shard1of1.bin
├── light_accident_rate_tfjs/  # TensorFlow.js light accident rate model
│   ├── model.json
│   └── group1-shard1of1.bin
├── intersection_encoder.pkl   # Intersection name encoder
├── weather_encoder.pkl        # Weather condition encoder
├── temp_scaler.pkl           # Temperature normalizer
├── feature_scaler.pkl        # Feature normalizer
├── intersection_coordinates.json # Intersection coordinates mapping
└── intersection_coordinates.pkl  # Same data in pickle format
```

## Intersection Mapping

The system supports 10 top intersections in Cambridge, MA:

```json
{
  "BROADWAY": {"latitude": 42.3736, "longitude": -71.1097, "count": 0},
  "MASSACHUSETTS AVENUE": {"latitude": 42.3736, "longitude": -71.1097, "count": 0},
  "CHILTON ST & HURON AVE": {"latitude": 42.3736, "longitude": -71.1097, "count": 0},
  "CONCORD AVE & GARDEN ST": {"latitude": 42.3736, "longitude": -71.1097, "count": 0},
  "BROOKLINE ST & GRANITE ST": {"latitude": 42.3736, "longitude": -71.1097, "count": 0},
  "BROADWAY & HAMPSHIRE ST": {"latitude": 42.3736, "longitude": -71.1097, "count": 0},
  "BRATTLE ST & SPARKS ST": {"latitude": 42.3736, "longitude": -71.1097, "count": 0},
  "MASSACHUSETTS AVE & SOMERVILLE AVE": {"latitude": 42.3736, "longitude": -71.1097, "count": 0},
  "PUTNAM AVE & RIVER ST": {"latitude": 42.3736, "longitude": -71.1097, "count": 0},
  "MASSACHUSETTS AVE & SIDNEY ST": {"latitude": 42.3736, "longitude": -71.1097, "count": 0}
}
```

## Weather Conditions

Supported weather conditions:
- **CLEAR**: Clear/sunny weather
- **CLOUDY**: Cloudy/overcast conditions
- **RAIN**: Rain/drizzle
- **SNOW**: Snow/sleet/hail
- **UNKNOWN**: Other or unknown conditions

## Model Performance

Based on training with 10 epochs:

- **Bike Count Model**: MAE ≈ 3.9 bikes
- **Severe Accident Rate Model**: MAE ≈ 0.00017 (very low rates expected)
- **Moderate Accident Rate Model**: MAE ≈ 0.013 (low rates expected)
- **Light Accident Rate Model**: MAE ≈ 0.006 (low rates expected)

## JavaScript Implementation Example

```javascript
// Load TensorFlow.js
import * as tf from '@tensorflow/tfjs';

// Load models
const bikeCountModel = await tf.loadLayersModel('/models_tfjs_fixed/bike_count_tfjs/model.json');
const severeRateModel = await tf.loadLayersModel('/models_tfjs_fixed/severe_accident_rate_tfjs/model.json');
const moderateRateModel = await tf.loadLayersModel('/models_tfjs_fixed/moderate_accident_rate_tfjs/model.json');
const lightRateModel = await tf.loadLayersModel('/models_tfjs_fixed/light_accident_rate_tfjs/model.json');

// Load preprocessors (you'll need to implement pickle loading or convert to JSON)
// For now, implement the preprocessing logic directly:

function preprocessInput(intersection, timeStr, weather, temperature, dayOfWeek, month) {
    // Convert time string (e.g., "08:30") to 15-minute increments
    const [hours, minutes] = timeStr.split(':').map(Number);
    const time15min = Math.floor((hours * 60 + minutes) / 15);
    
    // Encode intersection (0-9)
    const intersectionMap = {
        'BROADWAY': 0, 'MASSACHUSETTS AVENUE': 1, 'CHILTON ST & HURON AVE': 2,
        'CONCORD AVE & GARDEN ST': 3, 'BROOKLINE ST & GRANITE ST': 4,
        'BROADWAY & HAMPSHIRE ST': 5, 'BRATTLE ST & SPARKS ST': 6,
        'MASSACHUSETTS AVE & SOMERVILLE AVE': 7, 'PUTNAM AVE & RIVER ST': 8,
        'MASSACHUSETTS AVE & SIDNEY ST': 9
    };
    const intersectionEncoded = intersectionMap[intersection] || 0;
    
    // Encode weather (0-4)
    const weatherMap = {
        'CLEAR': 0, 'CLOUDY': 1, 'RAIN': 2, 'SNOW': 3, 'UNKNOWN': 4
    };
    const weatherEncoded = weatherMap[weather] || 4;
    
    // Cyclical encoding
    const hour = hours;
    const hourSin = Math.sin(2 * Math.PI * hour / 24);
    const hourCos = Math.cos(2 * Math.PI * hour / 24);
    
    const dayMap = { 'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                    'Friday': 4, 'Saturday': 5, 'Sunday': 6 };
    const dayOfWeekNum = dayMap[dayOfWeek] || 0;
    const daySin = Math.sin(2 * Math.PI * dayOfWeekNum / 7);
    const dayCos = Math.cos(2 * Math.PI * dayOfWeekNum / 7);
    
    const monthSin = Math.sin(2 * Math.PI * month / 12);
    const monthCos = Math.cos(2 * Math.PI * month / 12);
    
    // Normalize temperature (you'll need the actual scaler parameters)
    // For now, use a simple normalization: (temp - 50) / 20
    const temperatureNormalized = (temperature - 50) / 20;
    
    return [
        time15min, intersectionEncoded, weatherEncoded,
        hourSin, hourCos, daySin, dayCos, monthSin, monthCos,
        temperatureNormalized
    ];
}

async function predictBicycleTrafficAndAccidents(intersection, timeStr, weather, temperature, dayOfWeek, month) {
    // Preprocess input
    const features = preprocessInput(intersection, timeStr, weather, temperature, dayOfWeek, month);
    const inputTensor = tf.tensor2d([features], [1, 10]);
    
    // Stage 1: Predict bike count
    const bikeCountPred = await bikeCountModel.predict(inputTensor);
    const bikeCount = await bikeCountPred.data();
    bikeCountPred.dispose();
    
    // Stage 2: Predict accident rates
    const severeRatePred = await severeRateModel.predict(inputTensor);
    const moderateRatePred = await moderateRateModel.predict(inputTensor);
    const lightRatePred = await lightRateModel.predict(inputTensor);
    
    const severeRate = await severeRatePred.data();
    const moderateRate = await moderateRatePred.data();
    const lightRate = await lightRatePred.data();
    
    // Clean up tensors
    severeRatePred.dispose();
    moderateRatePred.dispose();
    lightRatePred.dispose();
    inputTensor.dispose();
    
    // Calculate final accident counts
    const bikeCountValue = bikeCount[0];
    const severeAccidents = bikeCountValue * severeRate[0];
    const moderateAccidents = bikeCountValue * moderateRate[0];
    const lightAccidents = bikeCountValue * lightRate[0];
    
    return {
        bikeCount: bikeCountValue,
        severeAccidents: severeAccidents,
        moderateAccidents: moderateAccidents,
        lightAccidents: lightAccidents
    };
}

// Example usage
const prediction = await predictBicycleTrafficAndAccidents(
    'BROADWAY', '08:30', 'CLEAR', 65, 'Tuesday', 6
);
console.log(prediction);
```

## Web App Features to Implement

### 1. Interactive Map
- Display Cambridge, MA with intersection markers
- Use the provided coordinates for intersection locations
- Color-code intersections based on predicted risk levels
- Allow users to click on intersections for detailed predictions

### 2. Prediction Interface
- Dropdown for intersection selection
- Time picker (15-minute increments)
- Weather condition selector
- Temperature input
- Day of week selector
- Month selector
- Real-time prediction updates

### 3. Visualization Components
- Bar charts showing predicted bike counts
- Pie charts for accident severity distribution
- Time-series graphs for different time periods
- Heat maps for risk visualization

### 4. Data Display
- Show predicted bike count with confidence intervals
- Display accident predictions with severity breakdown
- Include model performance metrics
- Show historical data comparison

## Technical Requirements

### Dependencies
```json
{
  "@tensorflow/tfjs": "^4.0.0",
  "@tensorflow/tfjs-node": "^4.0.0" // for server-side if needed
}
```

### Browser Compatibility
- Modern browsers with WebGL support
- ES6+ JavaScript support
- TensorFlow.js compatibility

### Performance Considerations
- Models are relatively small (~1MB each)
- Predictions should be fast (<100ms)
- Consider caching predictions for common inputs
- Implement loading states for model initialization

## Data Sources

The models were trained on:
- **Bicycle crash data**: 552 accident records from Cambridge, MA
- **City bike count data**: Daily counts from 2002-2019
- **Eco-Totem data**: 15-minute interval counts from Broadway counter
- **Weather data**: Integrated from multiple sources

## Model Limitations

1. **Limited Intersections**: Only supports top 10 intersections by data volume
2. **Weather Simplification**: Uses 5 broad weather categories
3. **Sparse Accident Data**: Very few accidents in training data
4. **Geographic Scope**: Cambridge, MA only
5. **Time Range**: Trained on data up to 2020

## Future Enhancements

1. **Real-time Weather Integration**: Connect to weather APIs
2. **More Intersections**: Expand to cover more Cambridge intersections
3. **Historical Trends**: Show prediction trends over time
4. **User Feedback**: Allow users to report actual vs predicted values
5. **Mobile Optimization**: Ensure mobile-friendly interface
6. **Accessibility**: Implement WCAG compliance

## Getting Started

1. Set up a web server to serve the model files
2. Implement the preprocessing logic
3. Load the TensorFlow.js models
4. Create the user interface
5. Test with various input combinations
6. Deploy and monitor performance

This system provides a foundation for bicycle safety prediction in Cambridge, MA, with room for expansion and improvement based on user feedback and additional data sources.