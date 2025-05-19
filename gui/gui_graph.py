from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Dict, Tuple
from gui_utils import haversine
import pandas as pd


# Each SCAT has multiple ways, each with a location, latitude, longitude, and direction
@dataclass
class SCATWay:
    location: str
    lat: float
    long: float
    direction: str  # Extracted from location name (N, E, S, W)

    @staticmethod
    def extract_direction(location: str) -> str:
        upper_location = location.upper()
        """Extract direction (N, E, S, W) from location name."""
        if ' N OF ' in upper_location:
            return 'NORTH'
        elif ' E OF ' in upper_location:
            return 'EAST'
        elif ' S OF ' in upper_location:
            return 'SOUTH'
        elif ' W OF ' in upper_location:
            return 'WEST'
        elif ' SE OF ' in upper_location:
            return 'SOUTH_EAST'
        elif ' SW OF ' in upper_location:
            return 'SOUTH_WEST'
        elif ' NE OF ' in upper_location:
            return 'NORTH_EAST'
        elif ' NW OF ' in upper_location:
            return 'NORTH_WEST'
        else:
            raise ValueError(f"Could not determine direction from location: {location}")

@dataclass
class SCAT:
    number: str
    ways: Dict[str, SCATWay]  # Dictionary mapping direction to SCATWay

@dataclass
class SCATConnection:
    from_scat: str
    from_direction: str
    to_scat: str
    to_direction: str
    distance: float

@dataclass
class Graph:
    nodes: Dict[str, SCAT]  # Dictionary mapping SCAT number to SCAT object
    edges: list[SCATConnection]  # List of connections between SCAT ways
    adj_lists: Dict[str, tuple[str, SCATConnection]] # The adjacency list of each SCAT node

def load_scats_from_excel(filename: str) -> Dict[str, SCAT]:
    """Load SCAT data from Excel file and create SCAT objects with their ways."""
    df = pd.read_excel(filename, sheet_name="Data", header=1)
    scats = {}
    
    # Group by SCAT number since each SCAT has multiple rows (one for each way)
    for scat_number, group in df.groupby('SCATS Number'):
        scat_number = str(scat_number)  # Convert to string to handle leading zeros
        ways = {}
        
        # Process each way of this SCAT
        for _, row in group.iterrows():
            location = row['Location']
            direction = SCATWay.extract_direction(location)
            
            way = SCATWay(
                location=location,
                lat=row['NB_LATITUDE'],
                long=row['NB_LONGITUDE'],
                direction=direction
            )
            ways[direction] = way
        
        scats[scat_number] = SCAT(scat_number, ways)
    
    return scats

def add_connection(graph: Graph, from_scat: str, from_direction: str, 
                  to_scat: str, to_direction: str) -> None:
    """Add a connection between two SCAT ways and calculate the distance."""
    if from_scat not in graph.nodes or to_scat not in graph.nodes:
        raise ValueError(f"SCAT {from_scat} or {to_scat} not found in graph")
    
    if from_direction not in graph.nodes[from_scat].ways:
        raise ValueError(f"Direction {from_direction} not found in SCAT {from_scat}")
    
    if to_direction not in graph.nodes[to_scat].ways:
        raise ValueError(f"Direction {to_direction} not found in SCAT {to_scat}")
    
    from_way = graph.nodes[from_scat].ways[from_direction]
    to_way = graph.nodes[to_scat].ways[to_direction]
    
    distance = haversine(from_way.lat, from_way.long, to_way.lat, to_way.long)
    
    connection = SCATConnection(
        from_scat=from_scat,
        from_direction=from_direction,
        to_scat=to_scat,
        to_direction=to_direction,
        distance=distance
    )
    
    graph.connections.append(connection)

def build_graph(filename: str) -> Graph:
    scats = load_scats_from_excel(filename)
    edges = []
    adj_lists = {}
    print(scats)
    graph = Graph(scats, edges, adj_lists)

    # SCATS: 0970, 2000, 2200, 2820, 2825, 2827, 2846, 3001, 3002, 3120, 3122, 3126, 3127, 3180, 3662, 3682, 3685, 3804, 3812, 4030, 4032, 4034, 4040, 
    # 4043, 4051, 4057, 4063, 4262, 4263, 4264, 4266, 4270, 4272, 4273, 4321, 4324, 4335, 4812, 4821 

    # These connections need to be added manually based on IBM data_Bor.pdf, SCATSSiteListingSpreadSheet_VicRoads.xls
    # IBM data_Bor.png also have bene provided to give more informations of what SCATs we are using (with a big red circle)

    # SCATS: 0970
     # add_connection(graph, "0970", "EAST", )


if __name__ == "__main__":
    build_graph("../datasets/Scats Data October 2006.xls")