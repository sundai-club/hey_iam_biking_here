Please build a website that visualizes bicyle accident rates in Cambridge MA.

For this you have public/data/bicycle_crashes_cleaned.csv that are police reports of traffic incidents involving bikes in Cambridge MA.

I want a website that displays stats of the actual data: overlay over a map and seasonal/time stats. This is a great example how it works in python:

# map overlay with a heatmap and points:
# Create interactive map of bicycle crashes
print("CREATING INTERACTIVE MAP:")
print("="*30)

# Calculate center point for the map
center_lat = df['Latitude'].mean()
center_lon = df['Longitude'].mean()

print(f"Map center: ({center_lat:.4f}, {center_lon:.4f})")

# Create base map
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=13,
    tiles='OpenStreetMap'
)

# Add bicycle crash markers with color coding by year
for idx, row in df.iterrows():
    # Color code by year
    year = row['Date Time'].year
    if year >= 2023:
        color = 'red'
    elif year >= 2021:
        color = 'orange'
    elif year >= 2019:
        color = 'yellow'
    else:
        color = 'green'
    
    # Create popup text
    popup_text = f"""
    <b>Bicycle Crash</b><br>
    Date: {row['Date Time'].strftime('%Y-%m-%d %H:%M')}<br>
    Street: {row['Street Name Cleaned']}<br>
    Cross Street: {row['Cross Street Cleaned']}<br>
    Intersection: {row['Intersection_ID']}<br>
    Collision: {row['Manner of Collision']}<br>
    Weather: {row['Weather Condition 1']}<br>
    """
    
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        popup=folium.Popup(popup_text, max_width=300),
        color='black',
        fillColor=color,
        fillOpacity=0.7,
        weight=1
    ).add_to(m)

# Add heatmap layer
heat_data = [[row['Latitude'], row['Longitude']] for idx, row in df.iterrows()]
plugins.HeatMap(heat_data, name='Bicycle Crash Heatmap').add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Save and display map
m.save('bicycle_crashes_interactive_map.html')
print("Interactive map saved as 'bicycle_crashes_interactive_map.html'")
print(f"Map shows {len(df)} bicycle crashes with coordinates")

# Display the map
display(m)


# Seasonal stats:

# Create comprehensive temporal visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Bicycle Crash Temporal Patterns in Cambridge, MA', fontsize=16, fontweight='bold')

# 1. Crashes by Year
yearly_crashes = df['Year'].value_counts().sort_index()
axes[0, 0].bar(yearly_crashes.index, yearly_crashes.values, color='skyblue', alpha=0.7)
axes[0, 0].set_title('Bicycle Crashes by Year')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Number of Crashes')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# 2. Crashes by Month
monthly_crashes = df['Month'].value_counts().sort_index()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
axes[0, 1].bar(month_names, monthly_crashes.values, color='lightcoral', alpha=0.7)
axes[0, 1].set_title('Bicycle Crashes by Month')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Number of Crashes')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# 3. Crashes by Day of Week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_crashes = df['Day of Week'].value_counts()
daily_crashes = daily_crashes.reindex(day_order)
axes[0, 2].bar(daily_crashes.index, daily_crashes.values, color='lightgreen', alpha=0.7)
axes[0, 2].set_title('Bicycle Crashes by Day of Week')
axes[0, 2].set_xlabel('Day of Week')
axes[0, 2].set_ylabel('Number of Crashes')
axes[0, 2].tick_params(axis='x', rotation=45)
axes[0, 2].grid(True, alpha=0.3)

# 4. Crashes by Hour of Day
hourly_crashes = df['Hour'].value_counts().sort_index()
axes[1, 0].bar(hourly_crashes.index, hourly_crashes.values, color='gold', alpha=0.7)
axes[1, 0].set_title('Bicycle Crashes by Hour of Day')
axes[1, 0].set_xlabel('Hour')
axes[1, 0].set_ylabel('Number of Crashes')
axes[1, 0].grid(True, alpha=0.3)

# 5. Crashes by Season
seasonal_crashes = df['Season'].value_counts()
season_order = ['Spring', 'Summer', 'Fall', 'Winter']
seasonal_crashes = seasonal_crashes.reindex(season_order)
axes[1, 1].bar(seasonal_crashes.index, seasonal_crashes.values, color='mediumpurple', alpha=0.7)
axes[1, 1].set_title('Bicycle Crashes by Season')
axes[1, 1].set_xlabel('Season')
axes[1, 1].set_ylabel('Number of Crashes')
axes[1, 1].grid(True, alpha=0.3)

# 6. Weekend vs Weekday
weekend_crashes = df['Is_Weekend'].value_counts()
weekend_labels = ['Weekday', 'Weekend']
axes[1, 2].bar(weekend_labels, weekend_crashes.values, color='orange', alpha=0.7)
axes[1, 2].set_title('Bicycle Crashes: Weekend vs Weekday')
axes[1, 2].set_xlabel('Day Type')
axes[1, 2].set_ylabel('Number of Crashes')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print("TEMPORAL PATTERN SUMMARY:")
print("="*30)
print(f"Peak year: {yearly_crashes.idxmax()} ({yearly_crashes.max()} crashes)")
print(f"Peak month: {month_names[monthly_crashes.idxmax()-1]} ({monthly_crashes.max()} crashes)")
print(f"Peak day: {daily_crashes.idxmax()} ({daily_crashes.max()} crashes)")
print(f"Peak hour: {hourly_crashes.idxmax()}:00 ({hourly_crashes.max()} crashes)")
print(f"Peak season: {seasonal_crashes.idxmax()} ({seasonal_crashes.max()} crashes)")
print(f"Weekend crashes: {weekend_crashes.get(True, 0)} ({weekend_crashes.get(True, 0)/len(df)*100:.1f}%)")
print(f"Weekday crashes: {weekend_crashes.get(False, 0)} ({weekend_crashes.get(False, 0)/len(df)*100:.1f}%)")


Please make them look nice and interactive.