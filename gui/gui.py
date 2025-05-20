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
        self.root.geometry("1440x900")
        self.root.minsize(1440, 900)
    
    def setup_graph(self):
        # Create canvas for graph with a minimum size
        self.canvas = tk.Canvas(self.graph_frame, bg='white', width=1000, height=800)
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
        
        # Create a canvas with scrollbars for results
        self.results_canvas = tk.Canvas(self.results_frame)
        y_scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", 
                                  command=self.results_canvas.yview)
        x_scrollbar = ttk.Scrollbar(self.results_frame, orient="horizontal", 
                                  command=self.results_canvas.xview)
        
        self.scrollable_frame = ttk.Frame(self.results_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(
                scrollregion=self.results_canvas.bbox("all")
            )
        )
        
        self.results_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=y_scrollbar.set,
                                    xscrollcommand=x_scrollbar.set)
        
        # Pack scrollbars and canvas
        y_scrollbar.pack(side="right", fill="y")
        x_scrollbar.pack(side="bottom", fill="x")
        self.results_canvas.pack(side="left", fill="both", expand=True)
    
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
            canvas_width = 1000
        if canvas_height <= 1:
            canvas_height = 800
        
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
            
            # Check if this edge is in the selected path
            is_in_path = False
            if self.paths and self.selected_path_index < len(self.paths):
                path = self.paths[self.selected_path_index][0]
                for i in range(len(path) - 1):
                    if (path[i] == connection.from_scat and path[i + 1] == connection.to_scat) or \
                       (path[i] == connection.to_scat and path[i + 1] == connection.from_scat):
                        is_in_path = True
                        break
            
            # Draw the line
            color = 'red' if is_in_path else 'black'
            self.canvas.create_line(from_pos[0], from_pos[1], 
                                  to_pos[0], to_pos[1], 
                                  fill=color, width=2)
            
            # Draw arrow only for one-way connections
            if (connection.from_direction != Direction.UNKNOWN and 
                connection.to_direction == Direction.UNKNOWN):
                self.draw_arrow(from_pos, to_pos, color)
            elif (connection.from_direction == Direction.UNKNOWN and 
                  connection.to_direction != Direction.UNKNOWN):
                self.draw_arrow(to_pos, from_pos, color)
            
            # Draw distance if enabled
            if self.show_distances.get():
                mid_x = (from_pos[0] + to_pos[0]) / 2
                mid_y = (from_pos[1] + to_pos[1]) / 2
                # Create a white background for the text
                self.canvas.create_oval(mid_x-25, mid_y-10, mid_x+25, mid_y+10, 
                                      fill='white', outline='white')
                self.canvas.create_text(mid_x, mid_y, 
                                      text=f"{connection.distance:.1f}km",
                                      fill='black',
                                      font=('Arial', 10, 'bold'))
        
        # Draw nodes
        for scat_num, pos in self.node_positions.items():
            color = 'red' if self.is_node_in_selected_path(scat_num) else 'black'
            self.canvas.create_oval(pos[0]-7.5, pos[1]-7.5, pos[0]+7.5, pos[1]+7.5, 
                                  fill=color)
            match scat_num:
                case "2825" | "3001" | "4262" | "4321" | "4335" | "4821" | "3662":
                    self.canvas.create_text(pos[0]-15, pos[1]-15, 
                                        text=scat_num, 
                                        fill='black',
                                        font=('Arial', 10, 'bold'))
                case "3812":
                    self.canvas.create_text(pos[0]-10, pos[1]+10, 
                                        text=scat_num, 
                                        fill='black',
                                        font=('Arial', 10, 'bold'))
                case _:
                    self.canvas.create_text(pos[0]+25, pos[1]-10, 
                                        text=scat_num, 
                                        fill='black',
                                        font=('Arial', 10, 'bold'))
    
    def draw_arrow(self, start: Tuple[float, float], end: Tuple[float, float], color: str = 'black'):
        """Draw an arrow between two points"""
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # Calculate arrow points
        arrow_size = 12
        arrow_x = mid_x - arrow_size * math.cos(angle - math.pi/6)
        arrow_y = mid_y - arrow_size * math.sin(angle - math.pi/6)
        arrow_x2 = mid_x - arrow_size * math.cos(angle + math.pi/6)
        arrow_y2 = mid_y - arrow_size * math.sin(angle + math.pi/6)
        
        self.canvas.create_polygon(mid_x, mid_y, 
                                 arrow_x, arrow_y, 
                                 arrow_x2, arrow_y2, 
                                 fill=color)
    
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
            messagebox.showerror("Error", "Please enter both source and destination SCATs")
            return
        
        if source not in self.graph.nodes or dest not in self.graph.nodes:
            messagebox.showerror("Error", "Invalid SCAT number(s)")
            return
        
        # Calculate paths
        if self.mode.get() == "shortest":
            self.paths = yen_k_shortest_paths(self.graph, source, dest, 5)
        else:
            # Fastest path calculation (We will add this later)
            pass
        
        if not self.paths:
            messagebox.showinfo("No Path Found", f"No valid path found between SCAT {source} and {dest}")
            return
            
        self.update_results()
        self.draw_graph()
    
    def update_results(self):
        # Clear previous results
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Configure styles for result boxes
        style = ttk.Style()
        style.configure('Result.TFrame', background='white')
        style.configure('ResultHover.TFrame', background='#f0f0f0')
        style.configure('Result.TLabel', background='white')
        style.configure('ResultHover.TLabel', background='#f0f0f0')
        
        # Add new results
        for i, (path, dist) in enumerate(self.paths):
            frame = ttk.Frame(self.scrollable_frame, style='Result.TFrame')
            frame.pack(fill=tk.X, pady=5, padx=5)
            
            # Add border and padding
            frame.configure(relief="solid", borderwidth=1)
            
            # Create a container for the content
            content_frame = ttk.Frame(frame, style='Result.TFrame')
            content_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Path information - show full path
            path_text = " → ".join(path)
            path_label = ttk.Label(content_frame, 
                                 text=f"Path {i+1}: {path_text}",
                                 style='Result.TLabel',
                                 wraplength=500)
            path_label.pack(anchor=tk.W, pady=5)
            
            # Distance/Time depends on the mode
            dist_text = f"Distance: {dist:.2f}km" if self.mode.get() == "shortest" else f"Time: {dist:.2f}min"
            dist_label = ttk.Label(content_frame, 
                                 text=dist_text,
                                 style='Result.TLabel')
            dist_label.pack(anchor=tk.W, pady=(0, 5))
            
            # Make everything clickable
            for widget in [frame, content_frame, path_label, dist_label]:
                widget.bind('<Button-1>', lambda e, idx=i: self.select_path(idx))
                widget.bind('<Enter>', lambda e, f=frame, l1=path_label, l2=dist_label: 
                          self.on_hover_enter(f, l1, l2))
                widget.bind('<Leave>', lambda e, f=frame, l1=path_label, l2=dist_label: 
                          self.on_hover_leave(f, l1, l2))
    
    def on_hover_enter(self, frame, label1, label2):
        """Handle mouse enter event"""
        frame.configure(style='ResultHover.TFrame')
        label1.configure(style='ResultHover.TLabel')
        label2.configure(style='ResultHover.TLabel')
    
    def on_hover_leave(self, frame, label1, label2):
        """Handle mouse leave event"""
        frame.configure(style='Result.TFrame')
        label1.configure(style='Result.TLabel')
        label2.configure(style='Result.TLabel')
    
    def select_path(self, index: int):
        self.selected_path_index = index
        self.draw_graph()

if __name__ == "__main__":
    root = tk.Tk()
    app = SCATPathFinder(root)
    root.mainloop()





