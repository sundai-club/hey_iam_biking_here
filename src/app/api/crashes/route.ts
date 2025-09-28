import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';
import { BicycleCrash } from '../../../types/crashes';

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
