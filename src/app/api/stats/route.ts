import { NextResponse } from 'next/server';
import { BicycleCrash, CrashStats } from '../../../types/crashes';

async function getCrashData(): Promise<BicycleCrash[]> {
  const response = await fetch(`${process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3000'}/api/crashes`);
  if (!response.ok) {
    throw new Error('Failed to fetch crash data');
  }
  return response.json();
}

export async function GET() {
  try {
    const data = await getCrashData();
    
    // Calculate yearly stats
    const yearly: { [year: string]: number } = {};
    data.forEach(crash => {
      const year = crash.Year.toString();
      yearly[year] = (yearly[year] || 0) + 1;
    });

    // Calculate monthly stats
    const monthly: { [month: string]: number } = {};
    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    data.forEach(crash => {
      const month = monthNames[crash.Month - 1];
      monthly[month] = (monthly[month] || 0) + 1;
    });

    // Calculate daily stats
    const daily: { [day: string]: number } = {};
    data.forEach(crash => {
      const day = crash['Day of Week'];
      daily[day] = (daily[day] || 0) + 1;
    });

    // Calculate hourly stats
    const hourly: { [hour: string]: number } = {};
    data.forEach(crash => {
      const hour = crash.Hour.toString();
      hourly[hour] = (hourly[hour] || 0) + 1;
    });

    // Calculate seasonal stats
    const seasonal: { [season: string]: number } = {};
    data.forEach(crash => {
      const season = crash.Season;
      seasonal[season] = (seasonal[season] || 0) + 1;
    });

    // Calculate weekend stats
    const weekend = { weekday: 0, weekend: 0 };
    data.forEach(crash => {
      if (crash.Is_Weekend) {
        weekend.weekend++;
      } else {
        weekend.weekday++;
      }
    });

    // Find peaks
    const peakYear = Object.keys(yearly).reduce((a, b) => yearly[a] > yearly[b] ? a : b);
    const peakMonth = Object.keys(monthly).reduce((a, b) => monthly[a] > monthly[b] ? a : b);
    const peakDay = Object.keys(daily).reduce((a, b) => daily[a] > daily[b] ? a : b);
    const peakHour = Object.keys(hourly).reduce((a, b) => hourly[a] > hourly[b] ? a : b);
    const peakSeason = Object.keys(seasonal).reduce((a, b) => seasonal[a] > seasonal[b] ? a : b);

    const stats: CrashStats = {
      yearly,
      monthly,
      daily,
      hourly,
      seasonal,
      weekend,
      summary: {
        total: data.length,
        peakYear,
        peakMonth,
        peakDay,
        peakHour,
        peakSeason,
        weekendPercentage: Math.round((weekend.weekend / data.length) * 100)
      }
    };

    return NextResponse.json(stats);
  } catch (error) {
    console.error('Error calculating stats:', error);
    return NextResponse.json({ error: 'Failed to calculate stats' }, { status: 500 });
  }
}
