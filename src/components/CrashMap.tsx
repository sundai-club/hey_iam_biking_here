'use client';

import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import { LatLngTuple } from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { BicycleCrash } from '../app/api/crashes/route';

// Fix for default markers in react-leaflet
import L from 'leaflet';
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface CrashMapProps {
  crashes: BicycleCrash[];
  showHeatmap?: boolean;
  yearFilter?: number;
}

function HeatmapLayer({ crashes }: { crashes: BicycleCrash[] }) {
  const map = useMap();
  
  useEffect(() => {
    if (crashes.length === 0) return;

    // Create heatmap data
    const heatmapData = crashes.map(crash => [crash.Latitude, crash.Longitude, 1]);
    
    // Simple heatmap implementation using circle markers with opacity
    const heatmapLayer = L.layerGroup();
    
    crashes.forEach(crash => {
      const intensity = 0.3; // Base intensity
      const circle = L.circleMarker([crash.Latitude, crash.Longitude], {
        radius: 8,
        fillColor: '#ff4444',
        color: '#ff4444',
        weight: 0,
        opacity: intensity,
        fillOpacity: intensity
      });
      heatmapLayer.addLayer(circle);
    });
    
    heatmapLayer.addTo(map);
    
    return () => {
      map.removeLayer(heatmapLayer);
    };
  }, [crashes, map]);

  return null;
}

export default function CrashMap({ crashes, showHeatmap = false, yearFilter }: CrashMapProps) {
  const [filteredCrashes, setFilteredCrashes] = useState<BicycleCrash[]>(crashes);

  useEffect(() => {
    if (yearFilter) {
      setFilteredCrashes(crashes.filter(crash => crash.Year === yearFilter));
    } else {
      setFilteredCrashes(crashes);
    }
  }, [crashes, yearFilter]);

  if (filteredCrashes.length === 0) {
    return (
      <div className="w-full h-96 bg-gray-100 rounded-lg flex items-center justify-center">
        <p className="text-gray-500">No crash data available</p>
      </div>
    );
  }

  // Calculate center point
  const centerLat = filteredCrashes.reduce((sum, crash) => sum + crash.Latitude, 0) / filteredCrashes.length;
  const centerLon = filteredCrashes.reduce((sum, crash) => sum + crash.Longitude, 0) / filteredCrashes.length;
  const center: LatLngTuple = [centerLat, centerLon];

  const getMarkerColor = (year: number) => {
    if (year >= 2023) return '#dc2626'; // red
    if (year >= 2021) return '#ea580c'; // orange
    if (year >= 2019) return '#eab308'; // yellow
    return '#16a34a'; // green
  };

  return (
    <div className="w-full h-96 rounded-lg overflow-hidden border border-gray-200">
      <MapContainer
        center={center}
        zoom={13}
        style={{ height: '100%', width: '100%' }}
        className="z-0"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        {showHeatmap && <HeatmapLayer crashes={filteredCrashes} />}
        
        {!showHeatmap && filteredCrashes.map((crash, index) => (
          <CircleMarker
            key={index}
            center={[crash.Latitude, crash.Longitude]}
            radius={5}
            pathOptions={{
              color: 'black',
              fillColor: getMarkerColor(crash.Year),
              fillOpacity: 0.7,
              weight: 1
            }}
          >
            <Popup maxWidth={300}>
              <div className="text-sm">
                <h3 className="font-bold text-red-600 mb-2">Bicycle Crash</h3>
                <p><strong>Date:</strong> {new Date(crash['Date Time']).toLocaleDateString()} {new Date(crash['Date Time']).toLocaleTimeString()}</p>
                <p><strong>Street:</strong> {crash['Street Name Cleaned']}</p>
                {crash['Cross Street Cleaned'] && (
                  <p><strong>Cross Street:</strong> {crash['Cross Street Cleaned']}</p>
                )}
                <p><strong>Intersection:</strong> {crash['Intersection_ID']}</p>
                <p><strong>Collision:</strong> {crash['Manner of Collision']}</p>
                <p><strong>Weather:</strong> {crash['Weather Condition 1']}</p>
                <p><strong>Injury:</strong> {crash['P1 Injury']}</p>
              </div>
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>
    </div>
  );
}
