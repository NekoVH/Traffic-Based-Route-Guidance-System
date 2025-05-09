from http import cookiejar
from pathlib import Path
import pygame
from astar import astar
#parse_graph and path_coordinates are currently for testing dummy data. May not be in final version
from astar_utils import parse_graph, path_coordinates

def gui_visualisation(path, graph):
    pygame.init()

    #Set up the display window
    #Adjust the screen size when map is available/used
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Path Visualiser")

    #Here is where you would load a map image
    #map_image = pygame.image.load("map_image.png") #Change image name as necessary
    #map_rect = map_image.get_rect()

    #Main GUI Loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        #Reset the screen display
        screen.fill((255, 255, 255)) #White background

        #Render graph and path
        render_graph(screen, graph, path)

        #Update the display
        pygame.display.flip()
        
    #Quit
    pygame.quit()

#For generating test paths with dummy data. May not be in final version.
def dummy_path_generator():
    graph = parse_graph("cade_testing/test_1.txt")
    path, _ = astar(graph, graph.origin, graph.destinations)
    coordinates = path_coordinates(path, graph)
    return coordinates, graph

#Pseudocode for actual path generator
'''
def path_generator(origin, destination):
    graph = parse_graph(???)
    path, _ = astar(graph, origin, destination)
    return path, graph
'''

#Renders the graph
def render_graph(screen, graph, path):
    #Renders the edges (Gray lines)
    
    for edge in graph.edges:
        node1, node2 = edge

        #Grab positions of both nodes
        x1, y1 = graph.nodes[node1]
        x2, y2 = graph.nodes[node2]

        pygame.draw.line(screen, (125, 125, 125), (x1*10, y1*10), (x2*10, y2*10), 2) #remove *10 when using actual data
        
    #Renders the path (Red lines)
    if len(path) > 1:
        for i in range(len(path) - 1):
            pygame.draw.line(screen, (255, 0, 0), path[i], path[i + 1], 5)

    #Renders the nodes (Black circles)
    for node in graph.nodes:
        x, y = graph.nodes[node]    
        pygame.draw.circle(screen, (0, 0, 0), (x*10, y*10), 5) #remove *10 when using actual data

#Run the visualisation GUI
#replace dummy parse when using actual data
path, graph = dummy_path_generator()
gui_visualisation(path, graph)




