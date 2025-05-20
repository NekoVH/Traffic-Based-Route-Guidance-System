from gui_graph import Graph, build_graph
from gui_algorithms import dijkstra, yen_k_shortest_paths
from typing import List, Dict
import json

def load_test_cases() -> List[Dict]:
    with open("algorithm_tests/tests.json", "r") as f:
        return json.load(f)

def run_tests(graph: Graph, test_suite: List[Dict]) -> Dict[str, int]:
    """
    Run the test suite and return statistics about passed/failed tests.
    """
    results = {
        "total": len(test_suite),
        "dijkstra_passed": 0,
        "yen_passed": 0,
        "failed_tests": []
    }
    
    for test_case in test_suite:
        source = test_case["source"]
        destination = test_case["destination"]
        k = test_case["k"]
        
        # Test Dijkstra's algorithm
        dijkstra_path, dijkstra_dist = dijkstra(graph, source, destination)
        dijkstra_expected = test_case["expected_dijkstra"]
        
        dijkstra_passed = (
            dijkstra_path == dijkstra_expected["path"] and
            abs(dijkstra_dist - dijkstra_expected["distance"]) < 0.1  # Allow 0.1km tolerance
        )
        
        # Test Yen's algorithm
        yen_paths = yen_k_shortest_paths(graph, source, destination, k)
        yen_expected = test_case["expected_yen"]
        
        yen_passed = len(yen_paths) == len(yen_expected)
        if yen_passed:
            for (path, dist), expected in zip(yen_paths, yen_expected):
                if path != expected["path"] or abs(dist - expected["distance"]) >= 0.1:
                    yen_passed = False
                    break
        
        if dijkstra_passed:
            results["dijkstra_passed"] += 1
        if yen_passed:
            results["yen_passed"] += 1
            
        if not (dijkstra_passed and yen_passed):
            results["failed_tests"].append({
                "name": test_case["name"],
                "source": source,
                "destination": destination,
                "dijkstra_passed": dijkstra_passed,
                "yen_passed": yen_passed,
                "dijkstra_expected": dijkstra_expected,
                "dijkstra_actual": {"path": dijkstra_path, "distance": dijkstra_dist},
                "yen_expected": yen_expected,
                "yen_actual": [{"path": path, "distance": dist} for path, dist in yen_paths]
            })
    
    return results

def print_test_results(results: Dict[str, int]):
    """
    Print the test results in a formatted way.
    """
    print("\nTest Results:")
    print(f"Total tests: {results['total']}")
    print(f"Dijkstra's algorithm passed: {results['dijkstra_passed']}/{results['total']}")
    print(f"Yen's algorithm passed: {results['yen_passed']}/{results['total']}")
    
    if results["failed_tests"]:
        print("\nFailed Tests:")
        for failed in results["failed_tests"]:
            print(f"\nTest case: {failed['name']}")
            print(f"Source: {failed['source']}")
            print(f"Destination: {failed['destination']}")
            print(f"Dijkstra passed: {failed['dijkstra_passed']}")
            print(f"Yen passed: {failed['yen_passed']}")
            
            if not failed['dijkstra_passed']:
                print("\nDijkstra's Algorithm:")
                print("Expected:", failed['dijkstra_expected'])
                print("Actual:", failed['dijkstra_actual'])
            
            if not failed['yen_passed']:
                print("\nYen's Algorithm:")
                print("Expected:", failed['yen_expected'])
                print("Actual:", failed['yen_actual'])

def main():
    # Build the graph
    graph = build_graph("../datasets/Scats Data October 2006.xls")
    
    # Get test cases
    test_cases = load_test_cases()
    
    # Run tests
    results = run_tests(graph, test_cases)
    
    # Print results
    print_test_results(results)

if __name__ == "__main__":
    main() 