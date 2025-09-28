import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';

export interface BicycleCrash {
  'Date Time': string;
  'Day of Week': string;
  'Street Name Cleaned': string;
  'Cross Street Cleaned': string;
  'Intersection_ID': string;
  'Latitude': number;
  'Longitude': number;
  'Manner of Collision': string;
  'First Harmful Event': string;
  'Weather Condition 1': string;
  'Weather Condition 2': string;
  'Ambient Light': string;
  'Road Surface Condition': string;
  'Traffic Control Device Type': string;
  'Traffic Control Device Functionality': string;
  'Roadway Junction Type': string;
  'Trafficway Description': string;
  'Speed Limit': string;
  'Object 1': string;
  'Object 2': string;
  'P1 Injury': string;
  'P2 Injury': string;
  'P1 Age': number;
  'P2 Age': number;
  'P1 Sex': string;
  'P2 Sex': string;
  'May involve cyclist': number;
  'May Involve Pedestrian': number;
  'P1 Non Motorist Desc': string;
  'P2 Non Motorist Desc': string;
  'Year': number;
  'Month': number;
  'Day': number;
  'Hour': number;
  'Day_of_Week_Num': number;
  'Is_Weekend': boolean;
  'Is_Rush_Hour': boolean;
  'Season': string;
}

let crashData: BicycleCrash[] | null = null;

function loadCrashData(): BicycleCrash[] {
  if (crashData) {
    return crashData;
  }

  try {
    const csvPath = path.join(process.cwd(), 'public', 'data', 'bicycle_crashes_cleaned.csv');
    const csvContent = fs.readFileSync(csvPath, 'utf-8');
    
    const result = Papa.parse<BicycleCrash>(csvContent, {
      header: true,
      skipEmptyLines: true,
      transform: (value, field) => {
        // Transform numeric fields
        if (['Latitude', 'Longitude', 'P1 Age', 'P2 Age', 'Year', 'Month', 'Day', 'Hour', 'Day_of_Week_Num', 'May involve cyclist', 'May Involve Pedestrian'].includes(field)) {
          const num = parseFloat(value);
          return isNaN(num) ? value : num;
        }
        // Transform boolean fields
        if (['Is_Weekend', 'Is_Rush_Hour'].includes(field)) {
          return value === 'True';
        }
        return value;
      }
    });

    crashData = result.data;
    return crashData;
  } catch (error) {
    console.error('Error loading crash data:', error);
    return [];
  }
}

export async function GET() {
  try {
    const data = loadCrashData();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching crash data:', error);
    return NextResponse.json({ error: 'Failed to fetch crash data' }, { status: 500 });
  }
}
