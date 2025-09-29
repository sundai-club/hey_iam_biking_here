import Papa from 'papaparse';
import fs from 'fs';
import path from 'path';
import { BicycleCrash } from '../types/crashes';

let crashData: BicycleCrash[] | null = null;

export async function loadCrashData(): Promise<BicycleCrash[]> {
  if (crashData) {
    return crashData;
  }

  try {
    const csvPath = path.join(process.cwd(), 'public', 'data', 'bicycle_crashes_cleaned.csv');
    const csvContent = await fs.promises.readFile(csvPath, 'utf-8');
    
    const result = Papa.parse<any>(csvContent, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: false
    });

    // Transform the data to match the BicycleCrash interface
    crashData = result.data.map((row: any): BicycleCrash => ({
      'Date Time': String(row['Date Time'] || ''),
      'Day of Week': String(row['Day of Week'] || ''),
      'Street Name Cleaned': String(row['Street Name Cleaned'] || ''),
      'Cross Street Cleaned': String(row['Cross Street Cleaned'] || ''),
      'Intersection_ID': String(row['Intersection_ID'] || ''),
      'Latitude': parseFloat(row['Latitude']) || 0,
      'Longitude': parseFloat(row['Longitude']) || 0,
      'Manner of Collision': String(row['Manner of Collision'] || ''),
      'First Harmful Event': String(row['First Harmful Event'] || ''),
      'Weather Condition 1': String(row['Weather Condition 1'] || ''),
      'Weather Condition 2': String(row['Weather Condition 2'] || ''),
      'Ambient Light': String(row['Ambient Light'] || ''),
      'Road Surface Condition': String(row['Road Surface Condition'] || ''),
      'Traffic Control Device Type': String(row['Traffic Control Device Type'] || ''),
      'Traffic Control Device Functionality': String(row['Traffic Control Device Functionality'] || ''),
      'Roadway Junction Type': String(row['Roadway Junction Type'] || ''),
      'Trafficway Description': String(row['Trafficway Description'] || ''),
      'Speed Limit': String(row['Speed Limit'] || ''),
      'Object 1': String(row['Object 1'] || ''),
      'Object 2': String(row['Object 2'] || ''),
      'P1 Injury': String(row['P1 Injury'] || ''),
      'P2 Injury': String(row['P2 Injury'] || ''),
      'P1 Age': parseFloat(row['P1 Age']) || 0,
      'P2 Age': parseFloat(row['P2 Age']) || 0,
      'P1 Sex': String(row['P1 Sex'] || ''),
      'P2 Sex': String(row['P2 Sex'] || ''),
      'May involve cyclist': parseFloat(row['May involve cyclist']) || 0,
      'May Involve Pedestrian': parseFloat(row['May Involve Pedestrian']) || 0,
      'P1 Non Motorist Desc': String(row['P1 Non Motorist Desc'] || ''),
      'P2 Non Motorist Desc': String(row['P2 Non Motorist Desc'] || ''),
      'Year': parseInt(row['Year']) || 0,
      'Month': parseInt(row['Month']) || 0,
      'Day': parseInt(row['Day']) || 0,
      'Hour': parseInt(row['Hour']) || 0,
      'Day_of_Week_Num': parseInt(row['Day_of_Week_Num']) || 0,
      'Is_Weekend': row['Is_Weekend'] === 'True',
      'Is_Rush_Hour': row['Is_Rush_Hour'] === 'True',
      'Season': String(row['Season'] || '')
    }));

    return crashData;
  } catch (error) {
    console.error('Error loading crash data:', error);
    console.error('Error details:', error);
    return [];
  }
}