'use client';

import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, LayersControl } from 'react-leaflet';
import { LatLngTuple } from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { BicycleCrash } from '../types/crashes';
import HeatmapLayer from './HeatmapLayer';

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

  const getMarkerColor = (injury: string) => {
    const injuryLower = injury.toLowerCase();
    if (injuryLower.includes('fatal') || injuryLower.includes('killed')) {
      return '#dc2626'; // red - fatal
    }
    if (injuryLower.includes('major') || injuryLower.includes('incapacitating')) {
      return '#ea580c'; // orange - major injury
    }
    if (injuryLower.includes('minor') || injuryLower.includes('suspected')) {
      return '#eab308'; // yellow - minor injury
    }
    if (injuryLower.includes('no apparent') || injuryLower.includes('none')) {
      return '#16a34a'; // green - no injury
    }
    return '#6b7280'; // gray - unknown
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
        
        <LayersControl position="topright">
          {/* Always show heatmap as base layer */}
          <LayersControl.Overlay name="Crash Heatmap" checked={true}>
            <HeatmapLayer 
              crashes={filteredCrashes} 
              intensity={0.6}
              radius={25}
              blur={15}
            />
          </LayersControl.Overlay>
          
          {/* Show individual markers as overlay */}
          <LayersControl.Overlay name="Individual Crashes" checked={true}>
            <div>
              {filteredCrashes.map((crash, index) => (
                <CircleMarker
                  key={index}
                  center={[crash.Latitude, crash.Longitude]}
                  radius={4}
                  pathOptions={{
                    color: 'black',
                    fillColor: getMarkerColor(crash['P1 Injury']),
                    fillOpacity: 0.8,
                    weight: 1.5
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
            </div>
          </LayersControl.Overlay>
        </LayersControl>
      </MapContainer>
    </div>
  );
}
