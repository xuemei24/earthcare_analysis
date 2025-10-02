import pandas as pd

# Download and read the site list (you might need to clean it a bit)
url = "aeronet_locations_v3.csv"
site_data = pd.read_csv(url, delimiter=",", skiprows=1)
print(site_data.keys())

# Clean column names
site_data.columns = [col.strip() for col in site_data.columns]

# Convert lat/lon columns to numeric
site_data["Latitude(decimal_degrees)"] = pd.to_numeric(site_data["Latitude(decimal_degrees)"], errors='coerce')
site_data["Longitude(decimal_degrees)"] = pd.to_numeric(site_data["Longitude(decimal_degrees)"], errors='coerce')

# Filter: example for sites in a region
lat_min, lat_max = 10, 50
lon_min, lon_max = -20, 40

subset = site_data[
    (site_data["Latitude(decimal_degrees)"] >= lat_min) &
    (site_data["Latitude(decimal_degrees)"] <= lat_max) &
    (site_data["Longitude(decimal_degrees)"] >= lon_min) &
    (site_data["Longitude(decimal_degrees)"] <= lon_max)
]

print(subset[["Site_Name", "Elevation(meters)", "Latitude(decimal_degrees)", "Longitude(decimal_degrees)"]])

