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

export interface CrashStats {
  yearly: { [year: string]: number };
  monthly: { [month: string]: number };
  daily: { [day: string]: number };
  hourly: { [hour: string]: number };
  seasonal: { [season: string]: number };
  weekend: { weekday: number; weekend: number };
  summary: {
    total: number;
    peakYear: string;
    peakMonth: string;
    peakDay: string;
    peakHour: string;
    peakSeason: string;
    weekendPercentage: number;
  };
}
