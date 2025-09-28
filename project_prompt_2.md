Then I want a tab where I can inspect the model predictions: heatmap overlay over the map. The user should be able to control the weather/ time / day of the weak and month.
Please also add ways to run interpolation animations -- how does the heatmap changes through the day for each of the outputs: number of bikes, probability of a collision, probability of a serious/minor/light collision.

Finally, in the last tab I want some way to visually inspect the data vs the predictions and see if it is accurate.

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
- Uses a neural network with ReLU activation
- Output: Non-negative float representing bike count

**Stage 2: Accident Rate Prediction**
- Three separate models predict accident rates per cyclist
- Each model uses Sigmoid activation (outputs 0-1 range)
- Final accident count = Bike Count × Accident Rate

### Model Files Structure
```
models_tfjs_ready/
├── bike_count_tfjs/           # Stage 1: Bike count prediction
│   ├── model.json
│   └── group1-shard1of1.bin
├── severe_accident_rate_tfjs/ # Stage 2: Severe accident rate
│   ├── model.json
│   └── group1-shard1of1.bin
├── moderate_accident_rate_tfjs/ # Stage 2: Moderate accident rate
│   ├── model.json
│   └── group1-shard1of1.bin
├── light_accident_rate_tfjs/  # Stage 2: Light accident rate
│   ├── model.json
│   └── group1-shard1of1.bin
└── [preprocessing files]
    ├── intersection_encoder.pkl
    ├── weather_encoder.pkl
    ├── temp_scaler.pkl
    └── feature_scaler.pkl
```

## Input Parameters

The system requires these 6 input parameters:

### 1. Intersection Name (String)
**Valid intersections** (top 10 by data volume):
- "MASSACHUSETTS AVE & PROSPECT ST"
- "MASSACHUSETTS AVE & VASSAR ST"
- "MASSACHUSETTS AVE & AMHERST ST"
- "MASSACHUSETTS AVE & SIDNEY ST"
- "MASSACHUSETTS AVE & PEARL ST"
- "MASSACHUSETTS AVE & INMAN ST"
- "MASSACHUSETTS AVE & BEACON ST"
- "MASSACHUSETTS AVE & MEMORIAL DR"
- "MASSACHUSETTS AVE & BROADWAY"
- "MASSACHUSETTS AVE & CAMBRIDGE ST"

### 2. Time (String)
- Format: "HH:MM" (24-hour format)
- Example: "08:30", "14:15", "18:45"
- Represents time in 15-minute increments

### 3. Weather Condition (String)
**Valid weather conditions**:
- "CLEAR" (sunny, clear skies)
- "CLOUDY" (overcast, partly cloudy, broken clouds)
- "RAIN" (rain, drizzle)
- "SNOW" (snow, sleet, hail, freezing conditions)
- "UNKNOWN" (fallback for unclear conditions)

### 4. Temperature (Number)
- Temperature in Fahrenheit
- Range: Typically 0-100°F
- Example: 72.5

### 5. Day of Week (String)
**Valid days**:
- "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"

### 6. Month (Number)
- Month number (1-12)
- Example: 6 (June), 12 (December)

## Output Format

The system returns 4 predictions as floats:

```javascript
{
  bikeCount: 45.2,           // Predicted cyclists
  severeAccidents: 0.001,    // Predicted severe accidents
  moderateAccidents: 0.023,  // Predicted moderate accidents
  lightAccidents: 0.156      // Predicted light accidents
}
```

## Technical Implementation

### TensorFlow.js Model Loading
```javascript
// Load all 4 models
const bikeCountModel = await tf.loadLayersModel('/models_tfjs_ready/bike_count_tfjs/model.json');
const severeModel = await tf.loadLayersModel('/models_tfjs_ready/severe_accident_rate_tfjs/model.json');
const moderateModel = await tf.loadLayersModel('/models_tfjs_ready/moderate_accident_rate_tfjs/model.json');
const lightModel = await tf.loadLayersModel('/models_tfjs_ready/light_accident_rate_tfjs/model.json');
```

### Feature Preprocessing
The input features need to be preprocessed before prediction:

1. **Time Features**: Convert time to 15-minute increments (0-95)
2. **Cyclical Encoding**: Create sin/cos features for hour, day, month
3. **Categorical Encoding**: Encode intersection and weather using LabelEncoders
4. **Temperature Scaling**: Normalize temperature using StandardScaler
5. **Feature Scaling**: Apply StandardScaler to all features

### Prediction Pipeline
```javascript
async function predictBicycleTrafficAndAccidents(
  intersectionName, timeStr, weatherCondition, 
  temperatureFahrenheit, dayOfWeekStr, monthNum
) {
  // 1. Preprocess inputs
  const features = preprocessFeatures(/* inputs */);
  
  // 2. Stage 1: Predict bike count
  const bikeCount = await bikeCountModel.predict(features).data();
  
  // 3. Stage 2: Predict accident rates
  const severeRate = await severeModel.predict(features).data();
  const moderateRate = await moderateModel.predict(features).data();
  const lightRate = await lightModel.predict(features).data();
  
  // 4. Calculate final accident counts
  const severeAccidents = bikeCount * severeRate;
  const moderateAccidents = bikeCount * moderateRate;
  const lightAccidents = bikeCount * lightRate;
  
  return {
    bikeCount,
    severeAccidents,
    moderateAccidents,
    lightAccidents
  };
}
```

## Data Context

### Training Data Sources
The models were trained on unified data from:
1. **Bicycle Crashes**: 552 accident records (2010-2024)
2. **City Bike Counts**: Government bicycle counts (2002-2019)
3. **Eco-Totem Counter**: 15-minute interval counts from Broadway/Kendall Square

### Model Performance
- **Bike Count Model**: MAE ~2.5 cyclists
- **Severe Accident Rate**: MAE ~0.000004 (very rare events)
- **Moderate Accident Rate**: MAE ~0.015
- **Light Accident Rate**: MAE ~0.010

### Data Limitations
- Models trained on top 10 intersections by data volume
- Recent data focus (2020+)
- Accident predictions are rates per cyclist (very small numbers)
- Predictions represent expected values, not guarantees

## Web App Features to Implement

### Core Functionality
1. **Input Form**: Clean UI for the 6 input parameters
2. **Real-time Prediction**: Show predictions as user changes inputs
3. **Results Display**: Clear visualization of all 4 outputs
4. **Model Loading**: Proper async loading of TensorFlow.js models

### Recommended UI Elements
1. **Intersection Dropdown**: Pre-populated with valid intersections
2. **Time Picker**: 15-minute increment time selector
3. **Weather Selector**: Radio buttons or dropdown for weather
4. **Temperature Slider**: Range input for temperature
5. **Day/Month Selectors**: Standard date pickers
6. **Results Cards**: Separate cards for each prediction type

### Visualization Suggestions
1. **Bike Count**: Large number with trend indicators
2. **Accident Predictions**: Bar chart or gauge showing relative risk
3. **Risk Assessment**: Color-coded risk levels (low/medium/high)
4. **Comparison Mode**: Compare predictions across different scenarios

### Error Handling
1. **Invalid Inputs**: Validate intersection names, time format, etc.
2. **Model Loading**: Handle failed model loads gracefully
3. **Prediction Errors**: Show user-friendly error messages
4. **Fallback Values**: Provide reasonable defaults for missing data

## Technical Requirements

### Dependencies
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
```

### Browser Compatibility
- Modern browsers with WebGL support
- TensorFlow.js requires ES6+ support
- Recommended: Chrome, Firefox, Safari, Edge (latest versions)

### Performance Considerations
1. **Model Loading**: Load models once on app initialization
2. **Prediction Caching**: Cache results for identical inputs
3. **Memory Management**: Dispose of tensors after predictions
4. **Loading States**: Show loading indicators during predictions

## Example Usage

```javascript
// Example prediction call
const result = await predictBicycleTrafficAndAccidents(
  "MASSACHUSETTS AVE & PROSPECT ST",  // intersection
  "08:30",                            // time
  "CLEAR",                            // weather
  72.5,                               // temperature
  "Tuesday",                          // day of week
  6                                   // month (June)
);

console.log(result);
// {
//   bikeCount: 45.2,
//   severeAccidents: 0.001,
//   moderateAccidents: 0.023,
//   lightAccidents: 0.156
// }
```

## Additional Notes

### Model Interpretability
- Higher bike counts generally correlate with higher accident counts
- Rush hours (7-9 AM, 5-7 PM) typically show higher bike traffic
- Weather conditions significantly affect both bike counts and accident rates
- Weekends show different patterns than weekdays

### Future Enhancements
- Add more intersections as data becomes available
- Implement confidence intervals for predictions
- Add historical comparison features
- Include seasonal trend analysis

This system provides a sophisticated approach to bicycle safety prediction by normalizing accident rates by exposure (bike traffic), making it more accurate than raw accident count predictions.
