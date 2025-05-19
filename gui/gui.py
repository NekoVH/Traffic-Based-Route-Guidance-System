import tkinter as tk
from tkinter import ttk, messagebox
import math
from gui_algorithms import yen_k_shortest_paths
from gui_graph import Graph, build_graph, Direction
from typing import List, Tuple, Dict

class SCATPathFinder:
    def __init__(self, root):
        self.root = root
        self.root.title("SCAT Path Finder")
        
        # Load and initialize graph
        self.graph = build_graph("../datasets/Scats Data October 2006.xls")
        
        # GUI state
        self.source_scat = tk.StringVar()
        self.dest_scat = tk.StringVar()
        self.show_distances = tk.BooleanVar(value=False)
        self.mode = tk.StringVar(value="shortest")
        self.paths = []
        self.selected_path_index = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main container
        self.main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Graph
        self.graph_frame = ttk.Frame(self.main_container)
        self.main_container.add(self.graph_frame, weight=2)
        
        # Right side - Controls and Results
        self.control_frame = ttk.Frame(self.main_container)
        self.main_container.add(self.control_frame, weight=1)
        
        self.setup_graph()
        self.setup_controls()
        self.setup_results()
        
        self.node_positions = self.calculate_node_positions()
        self.draw_graph()
        
        # Make the window resizable
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
    
    def setup_graph(self):
        # Create canvas for graph with a minimum size
        self.canvas = tk.Canvas(self.graph_frame, bg='white', width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind resize event
        self.canvas.bind('<Configure>', self.on_canvas_resize)
    
    def on_canvas_resize(self, event):
        # Recalculate node positions when canvas is resized
        self.node_positions = self.calculate_node_positions()
        self.draw_graph()
    
    def setup_controls(self):
        # Input fields
        input_frame = ttk.LabelFrame(self.control_frame, text="Input", padding=10)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Source SCAT:").pack(anchor=tk.W)
        self.source_entry = ttk.Entry(input_frame, textvariable=self.source_scat)
        self.source_entry.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="Destination SCAT:").pack(anchor=tk.W)
        self.dest_entry = ttk.Entry(input_frame, textvariable=self.dest_scat)
        self.dest_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Mode selection
        mode_frame = ttk.LabelFrame(self.control_frame, text="Mode", padding=10)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Radiobutton(mode_frame, text="Shortest Path", 
                       variable=self.mode, value="shortest").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Fastest Path", 
                       variable=self.mode, value="fastest").pack(anchor=tk.W)
        
        # Options
        options_frame = ttk.LabelFrame(self.control_frame, text="Options", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Checkbutton(options_frame, text="Show Distances", 
                       variable=self.show_distances,
                       command=self.draw_graph).pack(anchor=tk.W)
        
        # Calculate button
        ttk.Button(self.control_frame, text="Calculate Paths", 
                  command=self.calculate_paths).pack(pady=10)
    
    def setup_results(self):
        # Results frame
        self.results_frame = ttk.LabelFrame(self.control_frame, text="Results", padding=10)
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a canvas with scrollbar for results
        self.results_canvas = tk.Canvas(self.results_frame)
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", 
                                 command=self.results_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.results_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(
                scrollregion=self.results_canvas.bbox("all")
            )
        )
        
        self.results_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.results_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def calculate_node_positions(self) -> Dict[str, Tuple[float, float]]:
        """Calculate screen positions for nodes based on their lat/long"""
        positions = {}
        
        # Get all lat/long values
        all_lats = [way.lat for scat in self.graph.nodes.values() for way in scat.ways.values()]
        all_longs = [way.long for scat in self.graph.nodes.values() for way in scat.ways.values()]
        
        min_lat = min(all_lats)
        max_lat = max(all_lats)
        min_long = min(all_longs)
        max_long = max(all_longs)
        
        # Get current canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Use minimum size if canvas hasn't been drawn yet
        if canvas_width <= 1:
            canvas_width = 800
        if canvas_height <= 1:
            canvas_height = 600
        
        # Add padding to keep nodes away from edges
        padding = 50
        canvas_width -= 2 * padding
        canvas_height -= 2 * padding
        
        for scat_num, scat in self.graph.nodes.items():
            # Calculate average position
            avg_lat = sum(way.lat for way in scat.ways.values()) / len(scat.ways)
            avg_long = sum(way.long for way in scat.ways.values()) / len(scat.ways)
            
            # Convert to screen coordinates with padding
            x = padding + canvas_width * (avg_long - min_long) / (max_long - min_long)
            y = padding + canvas_height * (1 - (avg_lat - min_lat) / (max_lat - min_lat))
            
            positions[scat_num] = (x, y)
            
        return positions
    
    def draw_graph(self):
        self.canvas.delete("all")
        
        # Draw edges
        for connection in self.graph.edges:
            from_pos = self.node_positions[connection.from_scat]
            to_pos = self.node_positions[connection.to_scat]
            
            # Draw the line
            self.canvas.create_line(from_pos[0], from_pos[1], 
                                  to_pos[0], to_pos[1], 
                                  fill='black', width=2)
            
            # Draw arrow for one-way connections
            if connection.from_direction != Direction.UNKNOWN and connection.to_direction != Direction.UNKNOWN:
                self.draw_arrow(from_pos, to_pos)
            
            # Draw distance if enabled
            if self.show_distances.get():
                mid_x = (from_pos[0] + to_pos[0]) / 2
                mid_y = (from_pos[1] + to_pos[1]) / 2
                self.canvas.create_text(mid_x, mid_y, 
                                      text=f"{connection.distance:.1f}km",
                                      fill='black')
        
        # Draw nodes
        for scat_num, pos in self.node_positions.items():
            color = 'red' if self.is_node_in_selected_path(scat_num) else 'black'
            self.canvas.create_oval(pos[0]-5, pos[1]-5, pos[0]+5, pos[1]+5, 
                                  fill=color)
            self.canvas.create_text(pos[0]+10, pos[1]-10, 
                                  text=scat_num, 
                                  fill='black')
    
    def draw_arrow(self, start: Tuple[float, float], end: Tuple[float, float]):
        """Draw an arrow between two points"""
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # Calculate arrow points
        arrow_size = 10
        arrow_x = mid_x - arrow_size * math.cos(angle - math.pi/6)
        arrow_y = mid_y - arrow_size * math.sin(angle - math.pi/6)
        arrow_x2 = mid_x - arrow_size * math.cos(angle + math.pi/6)
        arrow_y2 = mid_y - arrow_size * math.sin(angle + math.pi/6)
        
        self.canvas.create_polygon(mid_x, mid_y, 
                                 arrow_x, arrow_y, 
                                 arrow_x2, arrow_y2, 
                                 fill='black')
    
    def is_node_in_selected_path(self, scat_num: str) -> bool:
        """Check if a node is in the currently selected path"""
        if not self.paths or self.selected_path_index >= len(self.paths):
            return False
        return scat_num in self.paths[self.selected_path_index][0]
    
    def calculate_paths(self):
        source = self.source_scat.get()
        dest = self.dest_scat.get()
        
        # Validate input
        if not source or not dest:
            tk.messagebox.showerror("Error", "Please enter both source and destination SCATs")
            return
        
        if source not in self.graph.nodes or dest not in self.graph.nodes:
            tk.messagebox.showerror("Error", "Invalid SCAT number(s)")
            return
        
        # Calculate paths
        if self.mode.get() == "shortest":
            self.paths = yen_k_shortest_paths(self.graph, source, dest, 5)
        else:
            # Fastest path calculation (We will add this later)
            pass
        
        self.update_results()
        self.draw_graph()
    
    def update_results(self):
        # Clear previous results
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Add new results
        for i, (path, dist) in enumerate(self.paths):
            frame = ttk.Frame(self.scrollable_frame)
            frame.pack(fill=tk.X, pady=5)
            
            # Make the frame clickable
            frame.bind('<Button-1>', lambda e, idx=i: self.select_path(idx))
            
            # Path information - show full path
            path_text = " -> ".join(path)
            ttk.Label(frame, text=f"Path {i+1}: {path_text}").pack(anchor=tk.W)
            
            # Distance/Time
            dist_text = f"Distance: {dist:.2f}km" if self.mode.get() == "shortest" else f"Time: {dist:.2f}min"
            ttk.Label(frame, text=dist_text).pack(anchor=tk.W)
    
    def select_path(self, index: int):
        self.selected_path_index = index
        self.draw_graph()

if __name__ == "__main__":
    root = tk.Tk()
    app = SCATPathFinder(root)
    root.mainloop()





