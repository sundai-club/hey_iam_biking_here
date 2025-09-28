'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { BicycleCrash } from './api/crashes/route';
import { CrashStats } from './api/stats/route';
import StatsCharts from '../components/StatsCharts';

// Dynamically import the map component to avoid SSR issues
const CrashMap = dynamic(() => import('../components/CrashMap'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-96 bg-gray-100 rounded-lg flex items-center justify-center">
      <p className="text-gray-500">Loading map...</p>
    </div>
  )
});

export default function Home() {
  const [crashes, setCrashes] = useState<BicycleCrash[]>([]);
  const [stats, setStats] = useState<CrashStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [yearFilter, setYearFilter] = useState<number | undefined>(undefined);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [crashesResponse, statsResponse] = await Promise.all([
          fetch('/api/crashes'),
          fetch('/api/stats')
        ]);

        if (!crashesResponse.ok || !statsResponse.ok) {
          throw new Error('Failed to fetch data');
        }

        const [crashesData, statsData] = await Promise.all([
          crashesResponse.json(),
          statsResponse.json()
        ]);

        setCrashes(crashesData);
        setStats(statsData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const availableYears = Array.from(new Set(crashes.map(crash => crash.Year))).sort((a, b) => b - a);

  if (loading) {
  return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading bicycle crash data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-600 mb-4">Error: {error}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Cambridge Bicycle Crash Analysis
              </h1>
              <p className="mt-2 text-gray-600">
                Interactive visualization of bicycle accident data in Cambridge, MA
              </p>
            </div>
            <div className="mt-4 md:mt-0 flex flex-col sm:flex-row gap-4">
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium text-gray-700">Year Filter:</label>
                <select
                  value={yearFilter || ''}
                  onChange={(e) => setYearFilter(e.target.value ? parseInt(e.target.value) : undefined)}
                  className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">All Years</option>
                  {availableYears.map(year => (
                    <option key={year} value={year}>{year}</option>
                  ))}
                </select>
              </div>
              <button
                onClick={() => setShowHeatmap(!showHeatmap)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  showHeatmap
                    ? 'bg-red-600 text-white hover:bg-red-700'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {showHeatmap ? 'Show Markers' : 'Show Heatmap'}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Map Section */}
        <section className="mb-12">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Interactive Crash Map
            </h2>
            <p className="text-gray-600 mb-6">
              Explore bicycle crashes across Cambridge. Click on markers for detailed information.
              {showHeatmap && ' Heatmap shows crash density across the city.'}
            </p>
            <CrashMap 
              crashes={crashes} 
              showHeatmap={showHeatmap}
              yearFilter={yearFilter}
            />
            <div className="mt-4 flex flex-wrap gap-4 text-sm text-gray-600">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-600 rounded-full"></div>
                <span>2023+ crashes</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                <span>2021-2022 crashes</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <span>2019-2020 crashes</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-green-600 rounded-full"></div>
                <span>Pre-2019 crashes</span>
              </div>
            </div>
          </div>
        </section>

        {/* Statistics Section */}
        {stats && (
          <section>
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-2xl font-semibold text-gray-800 mb-4">
                Temporal Analysis & Statistics
              </h2>
              <p className="text-gray-600 mb-6">
                Comprehensive analysis of crash patterns by time, season, and day of week.
              </p>
              <StatsCharts stats={stats} />
        </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-600">
            <p className="mb-2">
              Data source: Cambridge Police Department traffic incident reports
            </p>
            <p className="text-sm">
              This visualization shows {crashes.length} bicycle crashes in Cambridge, MA
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
