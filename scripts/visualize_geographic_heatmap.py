"""
Geographic Khipu Heatmap

Creates interactive geographic visualization of khipu distribution and patterns:
- Provenance locations with summation rates
- Heatmap overlay for pattern intensity
- Interactive popups with statistics
- Export to HTML

Usage: python scripts/visualize_geographic_heatmap.py
"""

import pandas as pd
import folium
from folium.plugins import HeatMap

# Provenance location data (approximate coordinates from archaeological sites)
# Source: Archaeological records and literature on Inka khipus
PROVENANCE_LOCATIONS = {
    'Chachapoyas': (-6.2308, -77.8691),
    'Ica': (-14.0678, -75.7286),
    'Pachacamac': (-12.2667, -76.9167),
    'Puruchuco': (-12.0167, -76.9833),
    'Cajamarquilla': (-11.9833, -76.9167),
    'Chuquibamba': (-15.8333, -72.6500),
    'Chachapoyas region': (-6.2308, -77.8691),
    'Ica Valley': (-14.0678, -75.7286),
    'Nazca': (-14.8333, -74.9333),
    'Cusco': (-13.5319, -71.9675),
    'Cuzco': (-13.5319, -71.9675),
    'Lima': (-12.0464, -77.0428),
    'Arequipa': (-16.4090, -71.5375),
    'Puno': (-15.8402, -70.0219),
    'Ayacucho': (-13.1639, -74.2233),
    'Huancavelica': (-12.7867, -74.9760),
    'Junin': (-11.1589, -75.9928),
    'Ancash': (-9.5287, -77.5281),
    'La Libertad': (-8.1116, -79.0292),
    'Cajamarca': (-7.1627, -78.5127),
    'Lambayeque': (-6.7014, -79.9061),
    'Piura': (-5.1945, -80.6328),
    'Tumbes': (-3.5667, -80.4511)
}

def load_khipu_data():
    """Load and aggregate khipu data by provenance."""
    import sqlite3
    
    summation = pd.read_csv("data/processed/phase3/summation_test_results.csv")
    clusters = pd.read_csv("data/processed/phase4/cluster_assignments_kmeans.csv")  # Already has structural features
    
    # Get provenance from database
    conn = sqlite3.connect("data/khipu.db")
    provenance = pd.read_sql_query("SELECT KHIPU_ID, PROVENANCE FROM khipu_main", conn)
    conn.close()
    
    # Normalize KHIPU_ID to khipu_id for merging
    provenance['khipu_id'] = provenance['KHIPU_ID']
    
    # Merge data (clusters already has structural features, no need for separate features file)
    data = provenance.merge(summation, on='khipu_id', how='inner')
    data = data.merge(clusters, on='khipu_id', how='inner')
    
    # Clean provenance names
    data['PROVENANCE'] = data['PROVENANCE'].fillna('Unknown')
    data = data[data['PROVENANCE'] != 'Unknown']
    
    print(f"Available columns: {list(data.columns)}")
    print(f"Data shape: {data.shape}")
    
    return data

def aggregate_by_provenance(data):
    """Aggregate statistics by provenance."""
    agg_data = data.groupby('PROVENANCE').agg({
        'khipu_id': 'count',
        'has_pendant_summation': 'mean',
        'num_nodes': 'mean',
        'depth': 'mean',
        'avg_branching': 'mean',
        'has_numeric': 'mean',
        'pendant_match_rate': 'mean'
    }).reset_index()
    
    agg_data.columns = [
        'Provenance',
        'Count',
        'Summation Rate',
        'Avg Size',
        'Avg Depth',
        'Avg Branching',
        'Numeric Coverage',
        'Match Rate'
    ]
    
    return agg_data

def create_geographic_heatmap(output_file='outputs/visualizations/geographic_heatmap.html'):
    """Create interactive geographic heatmap of khipu patterns."""
    print("Loading khipu data...")
    data = load_khipu_data()
    
    print("Aggregating by provenance...")
    agg_data = aggregate_by_provenance(data)
    
    # Match with coordinates
    agg_data['Lat'] = agg_data['Provenance'].apply(
        lambda x: PROVENANCE_LOCATIONS.get(x, (None, None))[0]
    )
    agg_data['Lon'] = agg_data['Provenance'].apply(
        lambda x: PROVENANCE_LOCATIONS.get(x, (None, None))[1]
    )
    
    # Filter to locations with coordinates
    agg_data = agg_data.dropna(subset=['Lat', 'Lon'])
    
    print(f"Found {len(agg_data)} provenances with coordinates")
    
    # Create base map centered on Peru
    m = folium.Map(
        location=[-10.0, -75.0],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Add heatmap layer for summation rate
    heat_data = [
        [row['Lat'], row['Lon'], row['Summation Rate']]
        for _, row in agg_data.iterrows()
    ]
    
    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_opacity=0.8,
        radius=30,
        blur=25,
        gradient={
            0.0: 'blue',
            0.5: 'yellow',
            1.0: 'red'
        }
    ).add_to(m)
    
    # Add markers for each provenance
    for _, row in agg_data.iterrows():
        # Scale marker size by count
        radius = min(5 + row['Count'] * 0.5, 25)
        
        # Color by summation rate
        if row['Summation Rate'] > 0.4:
            color = 'red'
        elif row['Summation Rate'] > 0.25:
            color = 'orange'
        else:
            color = 'blue'
        
        # Create popup content
        popup_html = f"""
        <div style="font-family: Arial; min-width: 250px;">
            <h4 style="margin: 0 0 10px 0; color: #1f77b4;">{row['Provenance']}</h4>
            <hr style="margin: 5px 0;">
            <table style="width: 100%; font-size: 12px;">
                <tr><td><b>Khipus:</b></td><td>{row['Count']}</td></tr>
                <tr><td><b>Summation Rate:</b></td><td>{row['Summation Rate']*100:.1f}%</td></tr>
                <tr><td><b>Avg Size:</b></td><td>{row['Avg Size']:.0f} nodes</td></tr>
                <tr><td><b>Avg Depth:</b></td><td>{row['Avg Depth']:.1f} levels</td></tr>
                <tr><td><b>Avg Branching:</b></td><td>{row['Avg Branching']:.2f}</td></tr>
                <tr><td><b>Numeric Coverage:</b></td><td>{row['Numeric Coverage']*100:.1f}%</td></tr>
                <tr><td><b>Match Rate:</b></td><td>{row['Match Rate']:.3f}</td></tr>
            </table>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=300),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; 
                width: 220px; height: auto;
                background-color: white;
                border: 2px solid grey;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                z-index: 9999;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <h4 style="margin: 0 0 10px 0;">Legend</h4>
        <p style="margin: 5px 0;"><b>Heatmap:</b> Summation Rate</p>
        <div style="display: flex; justify-content: space-between; margin: 5px 0;">
            <span>Low</span>
            <div style="width: 100px; height: 10px; 
                       background: linear-gradient(to right, blue, yellow, red);"></div>
            <span>High</span>
        </div>
        <hr style="margin: 10px 0;">
        <p style="margin: 5px 0;"><b>Markers:</b></p>
        <p style="margin: 5px 0;">🔴 Red: >40% summation</p>
        <p style="margin: 5px 0;">🟠 Orange: 25-40% summation</p>
        <p style="margin: 5px 0;">🔵 Blue: <25% summation</p>
        <p style="margin: 5px 0;"><i>Size = khipu count</i></p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = """
    <div style="position: fixed; 
                top: 10px; left: 50%;
                transform: translateX(-50%);
                background-color: white;
                border: 2px solid #1f77b4;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 18px;
                font-weight: bold;
                z-index: 9999;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        🧶 Geographic Distribution of Inka Khipus
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save map
    m.save(output_file)
    print(f"Saved interactive map to {output_file}")
    
    # Export statistics
    stats_file = output_file.replace('.html', '_statistics.csv')
    agg_data.to_csv(stats_file, index=False)
    print(f"Saved statistics to {stats_file}")

def create_cluster_geographic_map(output_file='outputs/visualizations/cluster_geographic_map.html'):
    """Create map showing cluster distribution across provenances."""
    print("Loading data for cluster map...")
    data = load_khipu_data()
    
    # Count clusters by provenance
    cluster_prov = data.groupby(['PROVENANCE', 'cluster']).size().reset_index(name='count')
    
    # Get dominant cluster per provenance
    dominant = cluster_prov.loc[cluster_prov.groupby('PROVENANCE')['count'].idxmax()]
    
    # Match with coordinates
    dominant['Lat'] = dominant['PROVENANCE'].apply(
        lambda x: PROVENANCE_LOCATIONS.get(x, (None, None))[0]
    )
    dominant['Lon'] = dominant['PROVENANCE'].apply(
        lambda x: PROVENANCE_LOCATIONS.get(x, (None, None))[1]
    )
    
    dominant = dominant.dropna(subset=['Lat', 'Lon'])
    
    # Create map
    m = folium.Map(
        location=[-10.0, -75.0],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # Color palette for clusters
    cluster_colors = {
        0: '#1f77b4',  # blue
        1: '#ff7f0e',  # orange
        2: '#2ca02c',  # green
        3: '#d62728',  # red
        4: '#9467bd',  # purple
        5: '#8c564b',  # brown
        6: '#e377c2'   # pink
    }
    
    # Add markers
    for _, row in dominant.iterrows():
        color = cluster_colors.get(row['cluster'], 'gray')
        
        popup_html = f"""
        <div style="font-family: Arial;">
            <h4 style="color: {color};">{row['PROVENANCE']}</h4>
            <p><b>Dominant Cluster:</b> {row['cluster']}</p>
            <p><b>Count:</b> {row['count']} khipus</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=10 + row['count'] * 0.3,
            popup=folium.Popup(popup_html, max_width=200),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=3
        ).add_to(m)
    
    # Add title
    title_html = """
    <div style="position: fixed; 
                top: 10px; left: 50%;
                transform: translateX(-50%);
                background-color: white;
                border: 2px solid #1f77b4;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 18px;
                font-weight: bold;
                z-index: 9999;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        🧶 Dominant Khipu Archetype by Provenance
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    m.save(output_file)
    print(f"Saved cluster map to {output_file}")

def main():
    print("Creating geographic visualizations...")
    print("=" * 60)
    
    # Create summation heatmap
    create_geographic_heatmap()
    print()
    
    # Create cluster distribution map
    create_cluster_geographic_map()
    print()
    
    print("=" * 60)
    print("Visualizations complete!")
    print("\nOpen the HTML files in a web browser to interact with the maps.")

if __name__ == "__main__":
    main()
