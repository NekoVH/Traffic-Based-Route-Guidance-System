import math
from dataclasses import dataclass

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
