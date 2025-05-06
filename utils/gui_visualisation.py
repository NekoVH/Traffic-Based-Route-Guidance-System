import pygame

#For the future, could pass through path variables as parameters for setting up the path.
#So from astar, it would be a list
def gui_visualisation():
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
    path = [(0, 0), (100, 0), (100, 100), (200, 100)]
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

#Run the visualisation GUI
gui_visualisation()




