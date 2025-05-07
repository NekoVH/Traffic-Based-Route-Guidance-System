from pathlib import Path
import pygame
from astar import astar
#parse_graph and path_coordinates are currently for testing dummy data. May not be in final version
from astar_utils import parse_graph, path_coordinates

def gui_visualisation(path):
    pygame.init()

    #Set up the display window
    #Adjust the screen size when map is available/used
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Path Visualiser")

    #Here is where you would load a map image
    #map_image = pygame.image.load("map_image.png") #Change image name as necessary
    #map_rect = map_image.get_rect()

    #Example path for initial testing
    #path = [(0, 0), (100, 0), (100, 100), (200, 100)]
    path_colour = (255, 0, 0) #Red colour for path
    path_width = 5

    #Main GUI Loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        #Reset the screen display
        screen.fill((255, 255, 255)) #White background

        #Render path
        if len(path) > 1:
            for i in range(len(path) - 1):
                pygame.draw.line(screen, path_colour, path[i], path[i + 1], path_width)

        #Update the display
        pygame.display.flip()
        
    #Quit
    pygame.quit()

#For parsing and generating test paths with dummy data. May not be in final version.
def dummy_testing():
    #Parses Dummy Test data
    graph = parse_graph("cade_testing/test_1.txt")
    path, _ = astar(graph, graph.origin, graph.destinations)
    coordinates = path_coordinates(path, graph)
    return coordinates

#Run the visualisation GUI
gui_visualisation(dummy_testing())




