import Papa from 'papaparse';
import { BicycleCrash } from '../types/crashes';

let crashData: BicycleCrash[] | null = null;

export async function loadCrashData(): Promise<BicycleCrash[]> {
  if (crashData) {
    return crashData;
  }

  try {
    // Use fetch to read the CSV file from the public directory
    // This works in Vercel's serverless environment
    const baseUrl = process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3000';
    const csvUrl = `${baseUrl}/data/bicycle_crashes_cleaned.csv`;
    
    const response = await fetch(csvUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch CSV: ${response.status} ${response.statusText}`);
    }
    
    const csvContent = await response.text();
    
    const result = Papa.parse<BicycleCrash>(csvContent, {
      header: true,
      skipEmptyLines: true,
      transform: (value, field) => {
        // Transform numeric fields
        if (['Latitude', 'Longitude', 'P1 Age', 'P2 Age', 'Year', 'Month', 'Day', 'Hour', 'Day_of_Week_Num', 'May involve cyclist', 'May Involve Pedestrian'].includes(String(field))) {
          const num = parseFloat(value);
          return isNaN(num) ? value : num;
        }
        // Transform boolean fields
        if (['Is_Weekend', 'Is_Rush_Hour'].includes(String(field))) {
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
