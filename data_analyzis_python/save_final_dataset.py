#!/usr/bin/env python3
"""
Save the final unified dataset for inspection
"""

import pandas as pd
import numpy as np
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

def get_accident_severity(row):
    """Convert injury data to severity scores (0-1 range)"""
    p1_injury = str(row['P1 Injury']).upper() if pd.notna(row['P1 Injury']) else ''
    p2_injury = str(row['P2 Injury']).upper() if pd.notna(row['P2 Injury']) else ''
    
    if any(severity in p1_injury or severity in p2_injury for severity in ['FATAL', 'SUSPECTED SERIOUS', 'SERIOUS']):
        return 1.0  # Severe
    elif any(severity in p1_injury or severity in p2_injury for severity in ['SUSPECTED MINOR', 'MINOR']):
        return 0.5  # Moderate
    else:
        return 0.1  # Light (no apparent injury but still an accident)

def main():
    print("=== Creating Final Unified Dataset ===")
    
    # Load datasets
    print("Loading datasets...")
    crashes_df = pd.read_csv('data/processed/bicycle_crashes_cleaned.csv')
    city_count_df = pd.read_csv('data/city_bike_count.csv')
    eco_totem_df = pd.read_csv('data/eco_totem.csv')
    
    print(f"Crashes data: {crashes_df.shape}")
    print(f"City count data: {city_count_df.shape}")
    print(f"Eco-totem data: {eco_totem_df.shape}")
    
    # Process crashes data
    print("Processing crashes data...")
    crashes_df['normalized_intersection'] = crashes_df['Intersection_ID'].apply(normalize_intersection_name)
    crashes_df['normalized_weather'] = crashes_df['Weather Condition 1'].apply(normalize_weather_condition)
    crashes_df['datetime'] = pd.to_datetime(crashes_df['Date Time'])
    crashes_df['time_15min'] = (crashes_df['datetime'].dt.hour * 4 + crashes_df['datetime'].dt.minute // 15) % 96
    crashes_df['day_of_week'] = crashes_df['datetime'].dt.dayofweek
    crashes_df['month'] = crashes_df['datetime'].dt.month
    crashes_df['accident_severity'] = crashes_df.apply(get_accident_severity, axis=1)
    
    # Process city count data
    print("Processing city count data...")
    city_count_df['normalized_intersection'] = city_count_df['Count Location'].apply(normalize_intersection_name)
    city_count_df['normalized_weather'] = city_count_df['Weather'].apply(normalize_weather_condition)
    city_count_df['datetime'] = pd.to_datetime(city_count_df['Date'] + ' ' + city_count_df['Time'])
    city_count_df['time_15min'] = (city_count_df['datetime'].dt.hour * 4 + city_count_df['datetime'].dt.minute // 15) % 96
    city_count_df['day_of_week'] = city_count_df['datetime'].dt.dayofweek
    city_count_df['month'] = city_count_df['datetime'].dt.month
    
    # Aggregate city count data
    city_count_agg = city_count_df.groupby([
        'normalized_intersection', 'datetime', 'time_15min', 'day_of_week', 
        'month', 'normalized_weather', 'Temperature'
    ]).agg({'Count': 'sum'}).reset_index()
    
    # Process eco-totem data
    print("Processing eco-totem data...")
    eco_totem_df['datetime'] = pd.to_datetime(eco_totem_df['DateTime'])
    eco_totem_df['time_15min'] = (eco_totem_df['datetime'].dt.hour * 4 + eco_totem_df['datetime'].dt.minute // 15) % 96
    eco_totem_df['day_of_week'] = eco_totem_df['datetime'].dt.dayofweek
    eco_totem_df['month'] = eco_totem_df['datetime'].dt.month
    eco_totem_df['normalized_intersection'] = 'BROADWAY'
    eco_totem_df['normalized_weather'] = 'CLEAR'  # Default
    eco_totem_df['Temperature'] = 20.0  # Default
    
    # Create unified dataset
    print("Creating unified dataset...")
    
    # Prepare datasets for merging
    city_count_unified = city_count_agg[['normalized_intersection', 'datetime', 'time_15min', 
                                        'day_of_week', 'month', 'normalized_weather', 'Temperature', 'Count']].copy()
    city_count_unified['accident_severity'] = 0.0
    city_count_unified['data_source'] = 'city_count'
    
    eco_totem_unified = eco_totem_df[['normalized_intersection', 'datetime', 'time_15min', 
                                     'day_of_week', 'month', 'normalized_weather', 'Temperature', 'Total']].copy()
    eco_totem_unified = eco_totem_unified.rename(columns={'Total': 'Count'})
    eco_totem_unified['accident_severity'] = 0.0
    eco_totem_unified['data_source'] = 'eco_totem'
    
    crashes_unified = crashes_df[['normalized_intersection', 'datetime', 'time_15min', 
                                 'day_of_week', 'month', 'normalized_weather', 'accident_severity']].copy()
    crashes_unified['Count'] = 0
    crashes_unified['Temperature'] = 20.0
    crashes_unified['data_source'] = 'crashes'
    
    # Combine all datasets
    unified_df = pd.concat([city_count_unified, eco_totem_unified, crashes_unified], ignore_index=True)
    
    # Add additional features for model training
    print("Adding model features...")
    unified_df['hour'] = unified_df['datetime'].dt.hour
    unified_df['minute'] = unified_df['datetime'].dt.minute
    unified_df['year'] = unified_df['datetime'].dt.year
    
    # Create cyclical time features
    unified_df['hour_sin'] = np.sin(2 * np.pi * unified_df['hour'] / 24)
    unified_df['hour_cos'] = np.cos(2 * np.pi * unified_df['hour'] / 24)
    unified_df['day_sin'] = np.sin(2 * np.pi * unified_df['day_of_week'] / 7)
    unified_df['day_cos'] = np.cos(2 * np.pi * unified_df['day_of_week'] / 7)
    unified_df['month_sin'] = np.sin(2 * np.pi * unified_df['month'] / 12)
    unified_df['month_cos'] = np.cos(2 * np.pi * unified_df['month'] / 12)
    
    # Add season
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    unified_df['season'] = unified_df['month'].apply(get_season)
    
    # Add weekend flag
    unified_df['is_weekend'] = unified_df['day_of_week'].isin([5, 6])
    
    # Add rush hour flag
    unified_df['is_rush_hour'] = (
        ((unified_df['hour'] >= 7) & (unified_df['hour'] <= 9)) |  # Morning rush
        ((unified_df['hour'] >= 17) & (unified_df['hour'] <= 19))  # Evening rush
    )
    
    # Sort by datetime
    unified_df = unified_df.sort_values('datetime').reset_index(drop=True)
    
    print(f"Final unified dataset shape: {unified_df.shape}")
    print(f"Date range: {unified_df['datetime'].min()} to {unified_df['datetime'].max()}")
    print(f"Unique intersections: {unified_df['normalized_intersection'].nunique()}")
    print(f"Data sources: {unified_df['data_source'].value_counts().to_dict()}")
    print(f"Records with bike counts > 0: {len(unified_df[unified_df['Count'] > 0])}")
    print(f"Records with accidents > 0: {len(unified_df[unified_df['accident_severity'] > 0])}")
    
    # Save the final dataset
    output_file = 'data/processed/unified_final_dataset.csv'
    unified_df.to_csv(output_file, index=False)
    print(f"\nFinal unified dataset saved to: {output_file}")
    
    # Show sample of the data
    print("\n=== Sample of Final Dataset ===")
    print(unified_df.head(10))
    
    print("\n=== Column Information ===")
    print("Columns in final dataset:")
    for i, col in enumerate(unified_df.columns, 1):
        print(f"{i:2d}. {col}")
    
    print("\n=== Data Summary ===")
    print(f"Total records: {len(unified_df):,}")
    print(f"Date range: {unified_df['datetime'].min()} to {unified_df['datetime'].max()}")
    print(f"Unique intersections: {unified_df['normalized_intersection'].nunique()}")
    print(f"Weather distribution: {unified_df['normalized_weather'].value_counts().to_dict()}")
    print(f"Season distribution: {unified_df['season'].value_counts().to_dict()}")
    print(f"Data source distribution: {unified_df['data_source'].value_counts().to_dict()}")
    
    print("\n=== Dataset Creation Completed Successfully! ===")

if __name__ == "__main__":
    main()
