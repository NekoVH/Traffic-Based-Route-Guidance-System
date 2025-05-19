from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Dict, Tuple
from gui_utils import haversine
import pandas as pd
from enum import Enum

class Direction(Enum):
    NORTH = "NORTH"
    EAST = "EAST"
    SOUTH = "SOUTH"
    WEST = "WEST"
    NORTH_EAST = "NORTH_EAST"
    NORTH_WEST = "NORTH_WEST"
    SOUTH_EAST = "SOUTH_EAST"
    SOUTH_WEST = "SOUTH_WEST"
    UNKNOWN = "UNKNOWN"

# Each SCAT has multiple ways, each with a location, latitude, longitude, and direction
@dataclass
class SCATWay:
    location: str
    lat: float
    long: float
    direction: Direction

    @staticmethod
    def extract_direction(location: str) -> Direction:
        upper_location = location.upper()
        """Extract direction (N, E, S, W) from location name."""
        if ' N OF ' in upper_location:
            return Direction.NORTH
        elif ' E OF ' in upper_location:
            return Direction.EAST
        elif ' S OF ' in upper_location:
            return Direction.SOUTH
        elif ' W OF ' in upper_location:
            return Direction.WEST
        elif ' SE OF ' in upper_location:
            return Direction.SOUTH_EAST
        elif ' SW OF ' in upper_location:
            return Direction.SOUTH_WEST
        elif ' NE OF ' in upper_location:
            return Direction.NORTH_EAST
        elif ' NW OF ' in upper_location:
            return Direction.NORTH_WEST
        else:
            return Direction.UNKNOWN

@dataclass
class SCAT:
    number: str
    ways: Dict[Direction, SCATWay]

@dataclass
class SCATConnection:
    from_scat: str
    from_direction: Direction
    to_scat: str
    to_direction: Direction
    distance: float

@dataclass
class Graph:
    nodes: Dict[str, SCAT]  # Dictionary mapping SCAT number to SCAT object
    edges: list[SCATConnection]  # List of connections between SCAT ways
    adj_lists: Dict[str, list[tuple[str, SCATConnection]]] # The adjacency list of each SCAT node

def load_scats_from_excel(filename: str) -> Dict[str, SCAT]:
    """Load SCAT data from Excel file and create SCAT objects with their ways."""
    df = pd.read_excel(filename, sheet_name="Data", header=1, dtype={'SCATS Number': str})
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

def add_connection(graph: Graph, from_scat: str, from_direction: Direction, 
                  to_scat: str, to_direction: Direction) -> None:
    """Add a connection between two SCAT ways and calculate the distance."""
    if from_scat not in graph.nodes or to_scat not in graph.nodes:
        raise ValueError(f"SCAT {from_scat} or {to_scat} not found in graph")
    
    if from_direction not in graph.nodes[from_scat].ways:
        raise ValueError(f"Direction {from_direction} not found in SCAT {from_scat}")
    
    from_way = graph.nodes[from_scat].ways[from_direction]

    if to_direction != Direction.UNKNOWN:
        to_way = graph.nodes[to_scat].ways[to_direction]
    else:
        # If the direction is UNKNOWN, we need to find the average lat/long of all the ways of the SCAT
        lat_sum = 0
        long_sum = 0
        for way in graph.nodes[to_scat].ways.values():
            lat_sum += way.lat
            long_sum += way.long

        to_way = SCATWay(
            location=f"Average of {to_scat}",
            lat=lat_sum / len(graph.nodes[to_scat].ways),
            long=long_sum / len(graph.nodes[to_scat].ways),
            direction=to_direction
        )
    
    distance = haversine(from_way.lat, from_way.long, to_way.lat, to_way.long)
    
    connection = SCATConnection(
        from_scat=from_scat,
        from_direction=from_direction,
        to_scat=to_scat,
        to_direction=to_direction,
        distance=distance
    )
    
    graph.edges.append(connection)
    graph.adj_lists.setdefault(from_scat, []).append((to_scat, connection))

def build_graph(filename: str) -> Graph:
    scats = load_scats_from_excel(filename)
    edges = []
    adj_lists = {}
    graph = Graph(scats, edges, adj_lists)

    # SCATS: 0970, 2000, 2200, 2820, 2825, 2827, 2846, 3001, 3002, 3120, 3122, 3126, 3127, 3180, 3662, 3682, 3685, 3804, 3812, 4030, 4032, 4034, 4035,
    # 4040, 4043, 4051, 4057, 4063, 4262, 4263, 4264, 4266, 4270, 4272, 4273, 4321, 4324, 4335, 4812, 4821 

    # These connections need to be added manually based on IBM data_Bor.pdf, SCATSSiteListingSpreadSheet_VicRoads.xls
    # IBM data_Bor.png also have been provided to give more informations of what SCATs we are using (with a big red circle, and blue lines for edges)

    # These are one-way connection because the graph is not fully undirected, so you may see there are multiple connections between the same SCATs
    # SCAT: 0970
    add_connection(graph, "0970", Direction.WEST, "2846", Direction.EAST)
    add_connection(graph, "0970", Direction.NORTH, "3685", Direction.SOUTH)

    # SCAT: 2000
    add_connection(graph, "2000", Direction.SOUTH, "3685", Direction.NORTH)
    add_connection(graph, "2000", Direction.WEST, "4043", Direction.EAST)
    add_connection(graph, "2000", Direction.NORTH, "3682", Direction.SOUTH)

    # SCAT: 2200
    add_connection(graph, "2200", Direction.SOUTH, "3126", Direction.UNKNOWN)
    add_connection(graph, "2200", Direction.WEST, "4063", Direction.EAST)

    # SCAT: 2820
    add_connection(graph, "2820", Direction.SOUTH_EAST, "4321", Direction.WEST)
    add_connection(graph, "2820", Direction.SOUTH, "3662", Direction.NORTH)

    # SCAT: 2825
    add_connection(graph, "2825", Direction.SOUTH, "4030", Direction.UNKNOWN)

    # SCAT: 2827
    add_connection(graph, "2827", Direction.WEST, "2825", Direction.UNKNOWN)
    add_connection(graph, "2827", Direction.SOUTH, "4051", Direction.UNKNOWN)

    # SCAT: 2846
    add_connection(graph, "2846", Direction.EAST, "0970", Direction.WEST)

    # SCAT: 3001
    add_connection(graph, "3001", Direction.WEST, "4821", Direction.EAST)
    add_connection(graph, "3001", Direction.EAST, "3002", Direction.WEST)
    add_connection(graph, "3001", Direction.NORTH_EAST, "3662", Direction.SOUTH_WEST)
    add_connection(graph, "3001", Direction.SOUTH_WEST, "4262", Direction.UNKNOWN)

    # SCAT: 3002
    add_connection(graph, "3002", Direction.SOUTH, "4263", Direction.NORTH)
    add_connection(graph, "3002", Direction.WEST, "3001", Direction.EAST)
    add_connection(graph, "3002", Direction.NORTH, "3662", Direction.UNKNOWN)
    add_connection(graph, "3002", Direction.EAST, "4035", Direction.WEST)
    
    # SCAT: 3120
    add_connection(graph, "3120", Direction.SOUTH, "4040", Direction.NORTH)
    add_connection(graph, "3120", Direction.NORTH, "4035", Direction.SOUTH)
    add_connection(graph, "3120", Direction.EAST, "3122", Direction.WEST)

    # SCAT: 3122
    add_connection(graph, "3122", Direction.SOUTH, "3804", Direction.NORTH)
    add_connection(graph, "3122", Direction.WEST, "3120", Direction.EAST)
    add_connection(graph, "3122", Direction.EAST, "3127", Direction.WEST)

    # SCAT: 3126
    add_connection(graph, "3126", Direction.SOUTH, "3682", Direction.NORTH)
    add_connection(graph, "3126", Direction.WEST, "3127", Direction.EAST)

    # SCAT: 3127    
    add_connection(graph, "3127", Direction.WEST, "3122", Direction.EAST)
    add_connection(graph, "3127", Direction.NORTH, "4063", Direction.SOUTH)
    add_connection(graph, "3127", Direction.EAST, "3126", Direction.WEST)

    # SCAT: 3180
    add_connection(graph, "3180", Direction.WEST, "4051", Direction.EAST)
    add_connection(graph, "3180", Direction.SOUTH, "4057", Direction.NORTH)

    # SCAT: 3662
    add_connection(graph, "3662", Direction.NORTH, "2820", Direction.SOUTH)
    add_connection(graph, "3662", Direction.NORTH_EAST, "4335", Direction.UNKNOWN)
    add_connection(graph, "3662", Direction.SOUTH_WEST, "3001", Direction.NORTH_EAST)

    # SCAT: 3682
    add_connection(graph, "3682", Direction.SOUTH, "2000", Direction.NORTH)
    add_connection(graph, "3682", Direction.WEST, "3804", Direction.EAST)
    add_connection(graph, "3682", Direction.NORTH, "3126", Direction.SOUTH)

    # SCAT: 3685
    add_connection(graph, "3685", Direction.SOUTH, "0970", Direction.NORTH)
    add_connection(graph, "3685", Direction.NORTH, "2000", Direction.SOUTH)

    # SCAT: 3804
    add_connection(graph, "3804", Direction.SOUTH, "3812", Direction.NORTH_EAST)
    add_connection(graph, "3804", Direction.WEST, "4040", Direction.EAST)
    add_connection(graph, "3804", Direction.NORTH, "3122", Direction.SOUTH)
    add_connection(graph, "3804", Direction.EAST, "3682", Direction.WEST)

    # SCAT: 3812
    add_connection(graph, "3812", Direction.NORTH_WEST, "4040", Direction.SOUTH_EAST)
    add_connection(graph, "3812", Direction.NORTH_EAST, "3804", Direction.SOUTH)

    # SCAT: 4030
    add_connection(graph, "4030", Direction.SOUTH, "4032", Direction.NORTH)
    add_connection(graph, "4030", Direction.SOUTH_WEST, "4321", Direction.NORTH_EAST)

    # SCAT: 4032
    add_connection(graph, "4032", Direction.SOUTH, "4034", Direction.NORTH)
    add_connection(graph, "4032", Direction.WEST, "4321", Direction.EAST)
    add_connection(graph, "4032", Direction.NORTH, "4030", Direction.SOUTH)
    add_connection(graph, "4032", Direction.EAST, "4057", Direction.WEST)
    
    # SCAT: 4034
    add_connection(graph, "4034", Direction.SOUTH, "4035", Direction.NORTH)
    add_connection(graph, "4034", Direction.WEST, "4324", Direction.EAST)
    add_connection(graph, "4034", Direction.NORTH, "4032", Direction.SOUTH)
    add_connection(graph, "4034", Direction.EAST, "4063", Direction.WEST)

    # SCAT: 4035
    add_connection(graph, "4035", Direction.SOUTH, "3120", Direction.NORTH)
    add_connection(graph, "4035", Direction.WEST, "3002", Direction.EAST)
    add_connection(graph, "4035", Direction.NORTH, "4034", Direction.SOUTH)

    # SCAT: 4040
    add_connection(graph, "4040", Direction.SOUTH, "4043", Direction.NORTH)
    add_connection(graph, "4040", Direction.WEST, "4272", Direction.EAST)
    add_connection(graph, "4040", Direction.NORTH, "3120", Direction.SOUTH)
    add_connection(graph, "4040", Direction.EAST, "3804", Direction.WEST)
    add_connection(graph, "4040", Direction.NORTH_WEST, "4266", Direction.WEST)
    add_connection(graph, "4040", Direction.SOUTH_EAST, "3812", Direction.WEST)

    # SCAT: 4043
    add_connection(graph, "4043", Direction.WEST, "4273", Direction.EAST)
    add_connection(graph, "4043", Direction.NORTH, "4040", Direction.SOUTH)
    add_connection(graph, "4043", Direction.EAST, "2000", Direction.WEST)

    # SCAT: 4051
    add_connection(graph, "4051", Direction.EAST, "3180", Direction.WEST)
    
    # SCAT: 4057
    add_connection(graph, "4057", Direction.SOUTH, "4063", Direction.NORTH)
    add_connection(graph, "4057", Direction.WEST, "4032", Direction.EAST)
    add_connection(graph, "4057", Direction.NORTH, "3180", Direction.SOUTH)

    # SCAT: 4063
    add_connection(graph, "4063", Direction.SOUTH, "3127", Direction.NORTH)
    add_connection(graph, "4063", Direction.WEST, "4034", Direction.EAST)
    add_connection(graph, "4063", Direction.NORTH, "4057", Direction.SOUTH)
    add_connection(graph, "4063", Direction.EAST, "2200", Direction.WEST)

    # SCAT: 4262
    add_connection(graph, "4262", Direction.SOUTH_WEST, "4812", Direction.NORTH_EAST)

    # SCAT: 4263
    add_connection(graph, "4263", Direction.WEST, "4262", Direction.UNKNOWN)
    add_connection(graph, "4263", Direction.NORTH, "3002", Direction.SOUTH)
    add_connection(graph, "4263", Direction.EAST, "4264", Direction.WEST)

    # SCAT: 4264
    add_connection(graph, "4264", Direction.SOUTH, "4270", Direction.NORTH)
    add_connection(graph, "4264", Direction.WEST, "4263", Direction.EAST)
    add_connection(graph, "4264", Direction.NORTH, "4324", Direction.SOUTH)
    add_connection(graph, "4264", Direction.EAST, "4266", Direction.WEST)

    # SCAT: 4266
    add_connection(graph, "4266", Direction.WEST, "4264", Direction.EAST)
    add_connection(graph, "4266", Direction.EAST, "4040", Direction.NORTH_WEST)

    # SCAT: 4270
    add_connection(graph, "4270", Direction.WEST, "4812", Direction.UNKNOWN)
    add_connection(graph, "4270", Direction.NORTH, "4264", Direction.SOUTH)
    add_connection(graph, "4270", Direction.EAST, "4272", Direction.WEST)

    # SCAT: 4272
    add_connection(graph, "4272", Direction.SOUTH, "4273", Direction.NORTH)
    add_connection(graph, "4272", Direction.WEST, "4270", Direction.EAST)
    add_connection(graph, "4272", Direction.EAST, "4040", Direction.WEST)

    # SCAT: 4273
    add_connection(graph, "4273", Direction.EAST, "4043", Direction.WEST)
    add_connection(graph, "4273", Direction.NORTH, "4272", Direction.SOUTH)

    # SCAT: 4321
    add_connection(graph, "4321", Direction.WEST, "2820", Direction.SOUTH_EAST)
    add_connection(graph, "4321", Direction.EAST, "4032", Direction.WEST)
    add_connection(graph, "4321", Direction.NORTH_EAST, "4030", Direction.SOUTH_WEST)
    add_connection(graph, "4321", Direction.SOUTH_WEST, "4335", Direction.NORTH_EAST)

    # SCAT: 4324
    add_connection(graph, "4324", Direction.SOUTH, "4264", Direction.NORTH)
    add_connection(graph, "4324", Direction.WEST, "3662", Direction.UNKNOWN)
    add_connection(graph, "4324", Direction.EAST, "4034", Direction.WEST)

    # SCAT: 4335
    add_connection(graph, "4335", Direction.NORTH_EAST, "4321", Direction.SOUTH_WEST)

    # SCAT: 4812
    add_connection(graph, "4812", Direction.NORTH_EAST, "4262", Direction.SOUTH_WEST)

    # SCAT: 4821
    add_connection(graph, "4821", Direction.EAST, "3001", Direction.WEST)

    return graph

if __name__ == "__main__":
    graph = build_graph("../datasets/Scats Data October 2006.xls")
    print(graph)