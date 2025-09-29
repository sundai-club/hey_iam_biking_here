'use client';

import { useEffect } from 'react';
import { useMap } from 'react-leaflet';
import { BicycleCrash } from '../types/crashes';
import L from 'leaflet';

// Import leaflet.heat
import 'leaflet.heat';

let heatLayer: L.Layer | null = null;

interface HeatmapLayerProps {
  crashes: BicycleCrash[];
  intensity?: number;
  radius?: number;
  blur?: number;
  maxZoom?: number;
}

export default function HeatmapLayer({ 
  crashes, 
  intensity = 0.6, 
  radius = 25, 
  blur = 15, 
  maxZoom = 18 
}: HeatmapLayerProps) {
  const map = useMap();

  useEffect(() => {
    if (crashes.length === 0) return;

    // Remove existing heat layer if it exists
    if (heatLayer) {
      map.removeLayer(heatLayer);
    }

    try {
      // Prepare heatmap data
      const heatData = crashes.map(crash => [
        crash.Latitude, 
        crash.Longitude, 
        intensity
      ]);

      // Create heatmap layer using the imported plugin
      heatLayer = ((L as unknown as Record<string, unknown>).heatLayer as (data: number[][], options: Record<string, unknown>) => L.Layer)(heatData, {
        radius: radius,
        blur: blur,
        maxZoom: maxZoom,
        gradient: {
          0.0: 'blue',
          0.2: 'cyan', 
          0.4: 'lime',
          0.6: 'yellow',
          0.8: 'orange',
          1.0: 'red'
        }
      });

      // Add heatmap to map
      heatLayer.addTo(map);

    } catch (error) {
      console.error('Error creating heatmap:', error);
      
      // Fallback: create a simple heatmap using circle markers
      const fallbackLayer = L.layerGroup();
      
      crashes.forEach(crash => {
        const circle = L.circleMarker([crash.Latitude, crash.Longitude], {
          radius: 12,
          fillColor: '#ff4444',
          color: '#ff4444',
          weight: 0,
          opacity: 0.3,
          fillOpacity: 0.3
        });
        fallbackLayer.addLayer(circle as unknown as L.Layer);
      });
      
      fallbackLayer.addTo(map);
      heatLayer = fallbackLayer;
    }

    return () => {
      if (heatLayer) {
        map.removeLayer(heatLayer);
        heatLayer = null;
      }
    };
  }, [crashes, map, intensity, radius, blur, maxZoom]);

  return null;
}
