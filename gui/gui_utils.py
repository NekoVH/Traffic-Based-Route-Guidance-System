from math import atan2, radians, sin, cos, sqrt # For calculating edge length/distance 

# Function to calculate length of road segments
def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula to calculate distance
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return round(distance, 4) # The result is in kilometers

def reconstruct_path(node: str, prev: list):
    """Given a node, backtrack through the prev list to see the path taken"""
    path = []
    while node in prev:
        path.append(node)
        node = prev[node]
    
    return list(reversed(path))
