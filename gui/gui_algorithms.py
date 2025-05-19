from gui_graph import Graph, build_graph
from gui_utils import reconstruct_path
import heapq
from copy import deepcopy
from typing import List, Tuple, Set

def dijkstra(graph: Graph, source: str, destination: str, blocked_edges: Set[Tuple[str, str]] = None) -> Tuple[List[str], float]:
    '''
    Modified Dijkstra's algorithm that can avoid certain edges.
    blocked_edges is a set of (from_scat, to_scat) tuples to avoid.
    These blocked_edges are used to find the k-shortest paths by Yen's algorithm.
    '''
    if blocked_edges is None:
        blocked_edges = set()
        
    dist = {node: float('inf') for node in graph.nodes}
    prev = {node: None for node in graph.nodes}
    dist[source] = 0
    visited = set()
    pq = [(0, source)]

    while pq:
        current_dist, current_node = heapq.heappop(pq)
        if current_node in visited:
            continue
        visited.add(current_node)
        if current_node == destination:
            return reconstruct_path(current_node, prev), current_dist
        
        for neighbor, connection in graph.adj_lists.get(current_node, []):
            if neighbor in visited:
                continue
            if (current_node, neighbor) in blocked_edges:
                continue
            alt = current_dist + connection.distance
            if alt < dist[neighbor]:
                dist[neighbor] = alt
                prev[neighbor] = current_node
                heapq.heappush(pq, (alt, neighbor))
    
    return [], float('inf')

def yen_k_shortest_paths(graph: Graph, source: str, destination: str, k: int) -> List[Tuple[List[str], float]]:
    '''
    Yen's algorithm to find k-shortest paths between source and destination SCATs.
    Returns a list of tuples, each containing (path, distance).
    '''
    # Find the shortest path
    shortest_path, shortest_dist = dijkstra(graph, source, destination)
    
    if not shortest_path or shortest_path[-1] != destination:
        return []
    
    # Initialize the list of k-shortest paths
    k_paths = [(shortest_path, shortest_dist)]
    candidates = []
    
    for k_idx in range(1, k):
        # Get the previous k-1 shortest path
        prev_path, prev_dist = k_paths[-1]
        
        # For each node in the previous path (except the destination)
        for i in range(len(prev_path) - 1):
            # Create a set of edges to block
            blocked_edges = set()
            
            # Block all edges from the root path up to the current node
            for j in range(i):
                blocked_edges.add((prev_path[j], prev_path[j + 1]))
            
            # Block the edge we're currently considering
            blocked_edges.add((prev_path[i], prev_path[i + 1]))
            
            # Find the shortest path avoiding the blocked edges
            spur_path, spur_dist = dijkstra(graph, prev_path[i], destination, blocked_edges)
            
            if spur_path and spur_path[-1] == destination:
                # Combine the root path and spur path
                root_path = prev_path[:i]
                total_path = root_path + spur_path
                
                # Check for loops in the path
                if len(total_path) != len(set(total_path)):
                    continue
                
                # Calculate total distance
                total_dist = 0
                valid_path = True
                for j in range(len(total_path) - 1):
                    found_connection = False
                    for neighbor, connection in graph.adj_lists[total_path[j]]:
                        if neighbor == total_path[j + 1]:
                            total_dist += connection.distance
                            found_connection = True
                            break
                    if not found_connection:
                        valid_path = False
                        break
                
                if not valid_path:
                    continue
                
                # Add to candidates if it's not already in k_paths
                candidate = (total_path, total_dist)
                if candidate not in k_paths and candidate not in [c[1] for c in candidates]:
                    heapq.heappush(candidates, (total_dist, total_path))
        
        # If no more candidates, we're done
        if not candidates:
            break
            
        # Get the next shortest path from candidates
        next_dist, next_path = heapq.heappop(candidates)
        k_paths.append((next_path, next_dist))
    
    return k_paths

if __name__ == "__main__":
    graph = build_graph("../datasets/Scats Data October 2006.xls")
    
    # Find 5 shortest paths between SCAT 3685 and 2200
    k_paths = yen_k_shortest_paths(graph, "3685", "2200", 5)
    
    print("Top 5 shortest paths:")
    for i, (path, dist) in enumerate(k_paths, 1):
        print(f"\nPath {i} (Distance: {dist:.2f} km):")
        print(" -> ".join(path))
