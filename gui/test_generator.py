from gui_graph import Graph, build_graph
from gui_algorithms import dijkstra, yen_k_shortest_paths
import json
import os
from typing import List, Dict, Tuple

def create_test_cases() -> List[Dict]:
    """
    Create a list of manually verified test cases.
    These test cases are based on known paths between SCATs that have been verified manually.
    """
    return [
        # Direct connection test is the test that should only has one path even if k is greater than 1
        {
            "name": "Direct connection test 1", 
            "source": "3685",
            "destination": "2000",
            "k": 3,
            "expected_dijkstra": {
                "path": ["3685", "2000"],
                "distance": 0.28
            },
            "expected_yen": [
                {
                    "path": ["3685", "2000"],
                    "distance": 0.28
                }
            ]
        },
        {
            "name": "Direct connection test 2", 
            "source": "2846",
            "destination": "0970",
            "k": 3,
            "expected_dijkstra": {
                "path": ["2846", "0970"],
                "distance": 2.92
            },
            "expected_yen": [
                {
                    "path": ["2846", "0970"],
                    "distance": 2.92
                }
            ]
        },
        {
            "name": "Direct connection test 3", 
            "source": "2827",
            "destination": "2825",
            "k": 3,
            "expected_dijkstra": {
                "path": ["2827", "2825"],
                "distance": 1.43
            },
            "expected_yen": [
                {
                    "path": ["2827", "2825"],
                    "distance": 1.43
                }
            ]
        },
        # Multiple path test is the test that should have multiple paths
        {
            "name": "Multiple path test 1",
            "source": "2000",
            "destination": "4273",
            "k": 3,
            "expected_dijkstra": {
                "path": ["2000", "4043", "4273"],
                "distance": 4.25
            },
            "expected_yen": [
                {
                    "path": ["2000", "4043", "4273"],
                    "distance": 4.25
                },
                {
                    "path": ["2000", "3682", "3804", "4040", "4043", "4273"],
                    "distance": 7.19
                },
                {
                    "path": ["2000", "3682", "3804", "4040", "4272", "4273"],
                    "distance": 7.40
                },
            ]
        },
        {
            "name": "Multiple path test 2",
            "source": "2820",
            "destination": "4030",
            "k": 3,
            "expected_dijkstra": {
                "path": ["2820", "4321", "4030"],
                "distance": 2.87
            },
            "expected_yen": [
                {
                    "path": ["2820", "4321", "4030"],
                    "distance": 2.87
                },
                {
                    "path": ["2820", "4321", "4032", "4030"],
                    "distance": 3.42
                },
                {
                    "path": ["2820", "3662", "4335", "4321", "4030"],
                    "distance": 4.76
                },
            ]
        },
        {
            "name": "Multiple path test 3",
            "source": "4264",
            "destination": "4262",
            "k": 3,
            "expected_dijkstra": {
                "path": ["4264", "4263", "4262"],
                "distance": 1.50
            },
            "expected_yen": [
                {
                    "path": ["4264", "4263", "4262"],
                    "distance": 1.50
                },
                {
                    "path": ["4264", "4263", "3002", "3001", "4262"],
                    "distance": 2.79
                },
                {
                    "path": ["4264", "4270", "4812", "4262"],
                    "distance": 2.82
                },
            ]
        },
        # No path test is the test that should have no path
        {
            "name": "No path test 1",
            "source": "4030",
            "destination": "2827",
            "k": 2,
            "expected_dijkstra": {
                "path": [],
                "distance": 0.0
            },
            "expected_yen": []
        },
        {
            "name": "No path test 2",
            "source": "4057",
            "destination": "2825",
            "k": 2,
            "expected_dijkstra": {
                "path": [],
                "distance": 0.0
            },
            "expected_yen": []
        },
        # Long path test is the test that should have a very long path
        {
            "name": "Long path test 1",
            "source": "2000",
            "destination": "3001",
            "k": 2,
            "expected_dijkstra": {
                "path": ["2000", "3682", "3804", "4040", "4266", "4264", "4263", "3002", "3001"],
                "distance": 8.98
            },
            "expected_yen": [
                {
                    "path": ["2000", "3682", "3804", "4040", "4266", "4264", "4263", "3002", "3001"],
                    "distance": 8.98
                },
                {
                    "path": ["2000", "4043", "4040", "4266", "4264", "4263", "3002", "3001"],
                    "distance": 9.10
                }
            ]
        },
        {
            "name": "Long path test 2",
            "source": "0970",
            "destination": "4821",
            "k": 3,
            "expected_dijkstra": {
                "path": ["0970", "3685", "2000", "3682", "3804", "4040", "4266", "4264", "4263", "3002", "3001", "4821"],
                "distance": 11.75
            },
            "expected_yen": [
                {
                    "path": ["0970", "3685", "2000", "3682", "3804", "4040", "4266", "4264", "4263", "3002", "3001", "4821"],
                    "distance": 11.75
                },
                {
                    "path": ["0970", "3685", "2000", "4043", "4040", "4266", "4264", "4263", "3002", "3001", "4821"],
                    "distance": 11.86
                },
                {
                    "path": ["0970", "3685", "2000", "3682", "3804", "4040", "4272", "4270", "4264", "4263", "3002", "3001", "4821"],
                    "distance": 12.10
                }
            ]
        }
    ]

def save_tests(test_suite: List[Dict], filename: str):
    """
    Save the test suite to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(test_suite, f, indent=2)

def main():
    # Create test directory if it doesn't exist
    if not os.path.exists("algorithm_tests"):
        os.makedirs("algorithm_tests")
    
    # Build the graph
    graph = build_graph("../datasets/Scats Data October 2006.xls")
    
    # Get test cases
    test_suite = create_test_cases()
    
    # Save test cases
    save_tests(test_suite, "algorithm_tests/tests.json")
    
if __name__ == "__main__":
    main()
