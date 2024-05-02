import numpy as np
import pandas as pd

# Function to calculate the distance between two points using the Haversine formula
def haversine_manual(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

# Function to check if the current location is more than 1 kilometer away from all points on the path
def is_out_of_path_manual(current_location, path_coordinates):
    for _, row in path_coordinates.iterrows():
        distance = haversine_manual(current_location[1], current_location[0], row['Lon'], row['Lat'])
        if distance <= 1:  # If the distance is less than or equal to 1 kilometer
            return False  # The current location is not out of the path
    return True  # If the loop completes, the current location is out of the path

# Example usage:
# Load your coordinates from a CSV file
file_path = 'c:\\Tymcz\\coordinates.csv'  # Change this to the path of your CSV file
coordinates_df = pd.read_csv(file_path)

# Define the current vehicle coordinates (example coordinates here)
current_coords = (53.63266, 20.84044)

# Check if the current coordinates are out of 1 kilometer from the planned path
out_of_path_manual = is_out_of_path_manual(current_coords, coordinates_df)
print("Is the vehicle out of path? ", out_of_path_manual)
