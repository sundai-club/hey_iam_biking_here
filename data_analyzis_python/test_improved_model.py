#!/usr/bin/env python3
"""
Test the improved Cambridge Bikes Model with 4 outputs
"""

import sys
import os
sys.path.append('models_improved')

from prediction_function import predict_bicycle_traffic_and_accidents

def test_predictions():
    """Test the improved model with various scenarios"""
    print("=== Testing Improved Cambridge Bikes Model ===")
    print("Model outputs: 1) Bike count, 2) Severe accidents, 3) Moderate accidents, 4) Light accidents")
    print()
    
    # Test scenarios
    test_cases = [
        {
            "name": "Rush Hour - Broadway - Clear Weather",
            "intersection": "BROADWAY",
            "time_15min": 32,  # 8:00 AM (rush hour)
            "weather": "CLEAR",
            "temperature": 70,
            "day_of_week": 1,  # Tuesday
            "month": 6  # June
        },
        {
            "name": "Evening Rush - Massachusetts Ave - Rainy",
            "intersection": "MASSACHUSETTS AVENUE",
            "time_15min": 68,  # 5:00 PM (evening rush)
            "weather": "RAIN",
            "temperature": 45,
            "day_of_week": 2,  # Wednesday
            "month": 11  # November
        },
        {
            "name": "Weekend Afternoon - Cambridge St - Cloudy",
            "intersection": "CAMBRIDGE STREET",
            "time_15min": 56,  # 2:00 PM
            "weather": "CLOUDY",
            "temperature": 60,
            "day_of_week": 6,  # Saturday
            "month": 9  # September
        },
        {
            "name": "Late Night - Hampshire St - Clear",
            "intersection": "HAMPSHIRE STREET",
            "time_15min": 88,  # 10:00 PM
            "weather": "CLEAR",
            "temperature": 55,
            "day_of_week": 4,  # Thursday
            "month": 7  # July
        },
        {
            "name": "Winter Morning - Broadway - Snow",
            "intersection": "BROADWAY",
            "time_15min": 28,  # 7:00 AM
            "weather": "SNOW",
            "temperature": 25,
            "day_of_week": 0,  # Monday
            "month": 1  # January
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['name']}")
        print(f"  Input: {test_case['intersection']}, {test_case['time_15min']//4}:{(test_case['time_15min']%4)*15:02d}, {test_case['weather']}, {test_case['temperature']}°F, {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][test_case['day_of_week']]}, Month {test_case['month']}")
        
        try:
            bike_count, severe_accidents, moderate_accidents, light_accidents = predict_bicycle_traffic_and_accidents(
                intersection=test_case['intersection'],
                time_15min=test_case['time_15min'],
                weather=test_case['weather'],
                temperature=test_case['temperature'],
                day_of_week=test_case['day_of_week'],
                month=test_case['month']
            )
            
            print(f"  Output:")
            print(f"    🚴 Bike count: {bike_count:.1f} bikes")
            print(f"    🚨 Severe accidents: {severe_accidents:.4f} accidents")
            print(f"    ⚠️  Moderate accidents: {moderate_accidents:.4f} accidents")
            print(f"    🟡 Light accidents: {light_accidents:.4f} accidents")
            
            # Calculate total accident risk
            total_accidents = severe_accidents + moderate_accidents + light_accidents
            if bike_count > 0:
                accident_rate = (total_accidents / bike_count) * 100
                print(f"    📊 Total accident risk: {total_accidents:.4f} accidents ({accident_rate:.3f}% of bike traffic)")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()

def analyze_model_behavior():
    """Analyze how the model behaves across different conditions"""
    print("=== Model Behavior Analysis ===")
    
    # Test different times of day
    print("1. Time of Day Analysis (Broadway, Clear, 70°F, Tuesday, June):")
    times = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88]  # Every 2 hours
    for time_15min in times:
        hour = time_15min // 4
        minute = (time_15min % 4) * 15
        bike_count, severe, moderate, light = predict_bicycle_traffic_and_accidents(
            "BROADWAY", time_15min, "CLEAR", 70, 1, 6
        )
        total_accidents = severe + moderate + light
        print(f"  {hour:2d}:{minute:02d} - {bike_count:5.1f} bikes, {total_accidents:.4f} accidents")
    
    print()
    
    # Test different weather conditions
    print("2. Weather Impact Analysis (Broadway, 8:00 AM, 70°F, Tuesday, June):")
    weathers = ["CLEAR", "CLOUDY", "RAIN", "SNOW"]
    for weather in weathers:
        bike_count, severe, moderate, light = predict_bicycle_traffic_and_accidents(
            "BROADWAY", 32, weather, 70, 1, 6
        )
        total_accidents = severe + moderate + light
        print(f"  {weather:6s} - {bike_count:5.1f} bikes, {total_accidents:.4f} accidents")
    
    print()
    
    # Test different intersections
    print("3. Intersection Comparison (8:00 AM, Clear, 70°F, Tuesday, June):")
    intersections = ["BROADWAY", "MASSACHUSETTS AVENUE", "CAMBRIDGE STREET", "HAMPSHIRE STREET"]
    for intersection in intersections:
        bike_count, severe, moderate, light = predict_bicycle_traffic_and_accidents(
            intersection, 32, "CLEAR", 70, 1, 6
        )
        total_accidents = severe + moderate + light
        print(f"  {intersection:20s} - {bike_count:5.1f} bikes, {total_accidents:.4f} accidents")

if __name__ == "__main__":
    test_predictions()
    print("\n" + "="*60 + "\n")
    analyze_model_behavior()
