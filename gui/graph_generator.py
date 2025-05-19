import pandas as pd

from dataclasses import dataclass

@dataclass
class Graph:
    nodes: dict
    edges: dict
    origin: str
    destinations: list
    adj_list: dict

def print_list(list, name):
    print(name + ":")
    
    for item in list:
        print(item)

    print()

def print_dict(dict, name):
    print(name + ":")

    for key, value in dict.items():
        print(f"{key}: {value}")

    print()

def scale_coordinates(coords, width=800, height=600, padding=20):
    # Extract all latitudes and longitudes
    lats = [lat for lat, lon in coords.values()]
    lons = [lon for lat, lon in coords.values()]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # Optionally add padding to avoid drawing on the very edge
    def scale(lat, lon):
        # Invert latitude for screen coordinates (y increases downward)
        x = padding + (lon - min_lon) / (max_lon - min_lon) * (width - 2 * padding)
        y = padding + (max_lat - lat) / (max_lat - min_lat) * (height - 2 * padding)
        return int(x), int(y)

    # Return a dict of (site, (x, y)) tuples
    return {site: scale(lat, lon) for site, (lat, lon) in coords.items()}

def generate_graph(scats_data, scats_sites):
    nodes = generate_nodes(scats_data)
    #print_list(nodes, "Nodes")       

    coordinates = generate_coordinates(nodes, scats_data)
    #print_list(coordinates, "Nodes with Coordinates")

    scaled_coordinates = scale_coordinates(coordinates)
    #print_dict(scaled_coordinates, "Nodes with Coordinates")

    edges = generate_edges(nodes, scats_sites)
    #print_dict(edges, "Edges")

    adj_list = generate_adj_list(nodes, edges)
    #print_dict(adj_list, "Adjacency List")

    #Graph is generated with placeholder origin and destination
    graph = Graph(scaled_coordinates, edges, "970", ["4335"], adj_list)

    return graph

def generate_nodes(scats_data):
    #Reads the Scats Data October 2006 sheet for the list of nodes to use.
    df = pd.read_excel(scats_data, sheet_name="Data", header=1)
    df = df.drop_duplicates(subset='SCATS Number', keep='first')

    #Stores the list of nodes as ints
    nodes = df['SCATS Number'].astype(int).to_list()

    return nodes

def generate_edges(nodes, scats_sites):
    #Reads the SCATS Site Listing sheet for the list of location descriptions
    df = pd.read_excel(scats_sites, sheet_name="SCATS Site Numbers", header=0, skiprows=9)
    df = df.drop_duplicates(subset='Site Number', keep='first')

    #Converts the column into ints for comparision with nodes
    df['Site Number'] = df['Site Number'].astype(int)

    #Stores the location descriptions and SCATS number for all sites in nodes
    matching_sites = []
    edges = {}

    for _, row in df.iterrows():
        if row['Site Number'] in nodes:
            
            #Stores Location Description as multiple strings
            location = row['Location Description']
            if isinstance(location, str) and '/' in location:
                locations = [s.strip() for s in location.split('/')]
            else:
                locations = [str(location).strip()]

            #Check for connections with previously stored SCAT sites
            for site_number, site_locations in matching_sites:
                
                #If any location string is shared...
                if set(locations) & set(site_locations):
                    
                    #Sorts the edge connection from smaller value to larger
                    edge = (str(min(site_number, row['Site Number'])), str(max(site_number, row['Site Number'])))
                    edges[edge] = 0

            matching_sites.append((row['Site Number'], locations))

    return edges

def generate_coordinates(nodes, scats_data):
    #Reads the Scats Data October 2006 sheet for the list of nodes to use.
    df = pd.read_excel(scats_data, sheet_name="Data", header=1)
    df = df.drop_duplicates(subset='NB_LATITUDE', keep='first')

    #Sets the columns to the expected variable type
    df['SCATS Number'] = df['SCATS Number'].astype(int)
    df['NB_LATITUDE'] = df['NB_LATITUDE'].astype(float)
    df['NB_LONGITUDE'] = df['NB_LONGITUDE'].astype(float)

    current_site = None
    current_coords = []
    coords = {}

    for _, row in df.iterrows():
        if row['SCATS Number'] in nodes:

            #Stores as current site if not already
            if row['SCATS Number'] != current_site:
                
                #If there is a current_site, appends the average coord to coord[] before moving to the next SCAT site
                if current_site != None:
                    lats, lons = zip(*current_coords)
                    avg_lat = sum(lats) / len(lats)
                    avg_lon = sum(lons) / len(lons)
                    coords[str(current_site)] = (avg_lat, avg_lon)                
                    current_coords = []

                current_site = row['SCATS Number']
            
            #Stores the current coords to average out later (as long as they aren't 0)
            if row['NB_LATITUDE'] != 0 and row['NB_LONGITUDE'] != 0:
                current_coords.append((row['NB_LATITUDE'], row['NB_LONGITUDE']))

    return coords

def generate_adj_list(nodes, edges):
    adj_list = {str(node): {} for node in nodes}
    for (site1, site2), weight in edges.items():
        site1 = str(site1)
        site2 = str(site2)
        adj_list[site1][site2] = weight
        adj_list[site2][site1] = weight
    return adj_list


    
