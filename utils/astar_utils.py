import math
from dataclasses import dataclass
from collections import defaultdict

#Imported from 2A
@dataclass
class Graph:
    nodes: dict
    edges: dict
    origin: str
    destinations: list
    adj_list: dict

def pythagoras(graph: Graph, goals: list[str]):
    result = {}

    goal_coords = [graph.nodes[goal] for goal in goals]

    for node, (start_x, start_y) in graph.nodes.items():
        # Set node heuristic to closest goal
        result[node] = min(
            math.sqrt((goal_x - start_x) ** 2 + (goal_y - start_y) ** 2) for (goal_x, goal_y) in goal_coords
        )

    return result

def reconstruct_path(node: str, prev: list):
    """Given a node, backtrack through the prev list to see the path taken"""
    path = []
    while node in prev:
        path.append(node)
        node = prev[node]
    
    return list(reversed(path))

"""
Below is code from Assignment 2A that is currently only used for testing the program with dummy data.
It may not be useful for the final version of the program.
"""

#Takes the dummy data graph and converts to coordinates (scaled by 10x by default)
def path_coordinates(path: list, graph: Graph, scale: int = 1):
    return [(x * scale, y * scale) for node in path for x, y in [graph.nodes[node]]]

#Parse graph information from text file (USED FOR TESTING)
def parse_graph(filename):
    nodes = {}
    edges = {}
    origin = None
    destinations = []
    adj_list = defaultdict(dict)
    graph = None
    with open(filename, 'r') as file:
        current_section = None

        for line in file:
            line = line.strip()

            if not line:
                continue

            if line.startswith('Nodes:'):
                current_section = 'nodes'
                continue
            elif line.startswith('Edges:'):
                current_section = 'edges'
                continue
            elif line.startswith('Origin:'):
                current_section = 'origin'
                continue
            elif line.startswith('Destinations:'):
                current_section = 'destinations'
                continue

            try:
                match current_section:
                    case 'nodes':
                        node_id, coords = line.split(': ')
                        coords = tuple(map(int, coords.strip('()').split(',')))
                        nodes[node_id] = coords
                        adj_list.setdefault(node_id, {})

                    case 'edges':
                        edge, weight = line.split(': ')
                        src, dest = edge.strip('()').split(',')
                        weight = int(weight)

                        # Add edges
                        edges[(src, dest)] = weight

                        # Add adjacency to the list, set empty array default is empty
                        adj_list.setdefault(src, {}).setdefault(dest, weight)

                    case 'origin':
                        origin = line
                    case 'destinations':
                        destinations = [dest.strip() for dest in line.split(';')]
                    case _:
                        raise ValueError("Unknown section")
            except:
                raise IOError("Unable to read input files, please check the input format again")


        graph = Graph(nodes, edges, origin, destinations, adj_list)

    return graph
