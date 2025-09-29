import { NextResponse } from 'next/server';
import { loadCrashData } from '../../../lib/crashData';

export async function GET() {
  try {
    const data = await loadCrashData();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching crash data:', error);
    return NextResponse.json({ error: 'Failed to fetch crash data' }, { status: 500 });
  }
}
