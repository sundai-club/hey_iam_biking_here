'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { CrashStats } from '../types/crashes';

interface StatsChartsProps {
  stats: CrashStats;
}

const COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];

export default function StatsCharts({ stats }: StatsChartsProps) {
  // Prepare data for charts
  const yearlyData = Object.entries(stats.yearly)
    .map(([year, count]) => ({ year, crashes: count }))
    .sort((a, b) => parseInt(a.year) - parseInt(b.year));

  const monthlyData = [
    { month: 'Jan', crashes: stats.monthly.Jan || 0 },
    { month: 'Feb', crashes: stats.monthly.Feb || 0 },
    { month: 'Mar', crashes: stats.monthly.Mar || 0 },
    { month: 'Apr', crashes: stats.monthly.Apr || 0 },
    { month: 'May', crashes: stats.monthly.May || 0 },
    { month: 'Jun', crashes: stats.monthly.Jun || 0 },
    { month: 'Jul', crashes: stats.monthly.Jul || 0 },
    { month: 'Aug', crashes: stats.monthly.Aug || 0 },
    { month: 'Sep', crashes: stats.monthly.Sep || 0 },
    { month: 'Oct', crashes: stats.monthly.Oct || 0 },
    { month: 'Nov', crashes: stats.monthly.Nov || 0 },
    { month: 'Dec', crashes: stats.monthly.Dec || 0 },
  ];

  const dailyData = [
    { day: 'Monday', crashes: stats.daily.Monday || 0 },
    { day: 'Tuesday', crashes: stats.daily.Tuesday || 0 },
    { day: 'Wednesday', crashes: stats.daily.Wednesday || 0 },
    { day: 'Thursday', crashes: stats.daily.Thursday || 0 },
    { day: 'Friday', crashes: stats.daily.Friday || 0 },
    { day: 'Saturday', crashes: stats.daily.Saturday || 0 },
    { day: 'Sunday', crashes: stats.daily.Sunday || 0 },
  ];

  const hourlyData = Array.from({ length: 24 }, (_, i) => ({
    hour: i,
    crashes: stats.hourly[i.toString()] || 0
  }));

  const seasonalData = [
    { season: 'Spring', crashes: stats.seasonal.Spring || 0 },
    { season: 'Summer', crashes: stats.seasonal.Summer || 0 },
    { season: 'Fall', crashes: stats.seasonal.Fall || 0 },
    { season: 'Winter', crashes: stats.seasonal.Winter || 0 },
  ];

  const weekendData = [
    { name: 'Weekday', value: stats.weekend.weekday, color: '#3b82f6' },
    { name: 'Weekend', value: stats.weekend.weekend, color: '#ef4444' },
  ];

  return (
    <div className="space-y-8">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="bg-white p-6 rounded-lg shadow-md border">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Total Crashes</h3>
          <p className="text-3xl font-bold text-blue-600">{stats.summary.total}</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md border">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Peak Year</h3>
          <p className="text-3xl font-bold text-red-600">{stats.summary.peakYear}</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md border">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Weekend Crashes</h3>
          <p className="text-3xl font-bold text-orange-600">{stats.summary.weekendPercentage}%</p>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Yearly Crashes */}
        <div className="bg-white p-6 rounded-lg shadow-md border">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Crashes by Year</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={yearlyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="crashes" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Monthly Crashes */}
        <div className="bg-white p-6 rounded-lg shadow-md border">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Crashes by Month</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={monthlyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="crashes" fill="#10b981" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Daily Crashes */}
        <div className="bg-white p-6 rounded-lg shadow-md border">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Crashes by Day of Week</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={dailyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" angle={-45} textAnchor="end" height={80} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="crashes" fill="#f59e0b" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Hourly Crashes */}
        <div className="bg-white p-6 rounded-lg shadow-md border">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Crashes by Hour of Day</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={hourlyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="crashes" fill="#8b5cf6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Seasonal Crashes */}
        <div className="bg-white p-6 rounded-lg shadow-md border">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Crashes by Season</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={seasonalData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="season" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="crashes" fill="#ec4899" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Weekend vs Weekday */}
        <div className="bg-white p-6 rounded-lg shadow-md border">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Weekend vs Weekday</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={weekendData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {weekendData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Key Insights */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-lg border">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Key Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
          <div className="bg-white p-4 rounded-lg">
            <p className="font-medium text-gray-700">Peak Month</p>
            <p className="text-lg font-bold text-blue-600">{stats.summary.peakMonth}</p>
          </div>
          <div className="bg-white p-4 rounded-lg">
            <p className="font-medium text-gray-700">Peak Day</p>
            <p className="text-lg font-bold text-green-600">{stats.summary.peakDay}</p>
          </div>
          <div className="bg-white p-4 rounded-lg">
            <p className="font-medium text-gray-700">Peak Hour</p>
            <p className="text-lg font-bold text-purple-600">{stats.summary.peakHour}:00</p>
          </div>
          <div className="bg-white p-4 rounded-lg">
            <p className="font-medium text-gray-700">Peak Season</p>
            <p className="text-lg font-bold text-pink-600">{stats.summary.peakSeason}</p>
          </div>
          <div className="bg-white p-4 rounded-lg">
            <p className="font-medium text-gray-700">Weekday Crashes</p>
            <p className="text-lg font-bold text-blue-600">{stats.weekend.weekday}</p>
          </div>
          <div className="bg-white p-4 rounded-lg">
            <p className="font-medium text-gray-700">Weekend Crashes</p>
            <p className="text-lg font-bold text-red-600">{stats.weekend.weekend}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
