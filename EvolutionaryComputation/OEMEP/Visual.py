import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import Evaluate

class MapVisualizer:
    def __init__(self, map_data):
        self.map_data = map_data
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.path_artists = [] # To keep track of drawn paths for easy clearing
        
        # Draw the "Static" map once during initialization
        self._draw_static_map()
        
    def _draw_static_map(self):
        # 1. Draw Sensors
        for s in self.map_data["sensors"]:
            wedge = patches.Wedge(s[0].center(), s[0].r, s[0].thetaL(), s[0].thetaR(), 
                                  alpha=0.3, facecolor='blue', edgecolor='none')
            self.ax.add_patch(wedge)
            self.ax.scatter(s[0].center.x, s[0].center.y, s=1, color='orange')

        # 2. Draw Obstacles
        for o in self.map_data["obstacles"]:
            poly = patches.Polygon(o[0].draw(), closed=True, 
                                   facecolor='darkgrey', edgecolor='black', alpha=0.8)
            self.ax.add_patch(poly)

        # 3. Draw Start and Goal
        self.ax.scatter(self.map_data["start"].x, self.map_data["start"].y, 
                        s=100, color='green', marker='P', label='Start')
        self.ax.scatter(self.map_data["goal"].x, self.map_data["goal"].y, 
                        s=100, color='red', marker='X', label='Goal')
        
        self.ax.set_aspect('equal')
        self.ax.autoscale()
        self.ax.set_title("Evolutionary Path Planning")

    def clear_paths(self):
        """Removes only the path lines from the axis."""
        for artist in self.path_artists:
            artist.remove()
        self.path_artists = []

    def draw_paths(self, paths, use_normalization=True):
        """Draws new paths on the existing stored map."""
        for path in paths:
            # Only normalize if necessary (it's a heavy operation)
            display_path = Evaluate.normalize(self.map_data, path) if use_normalization else path
            
            color = (random.random(), random.random(), random.random())
            xs = [p.x for p in display_path]
            ys = [p.y for p in display_path]
            
            # Draw line and store the artist
            line, = self.ax.plot(xs, ys, linewidth=2, color=color, alpha=0.7)
            # Draw points and store the artist
            scat = self.ax.scatter(xs, ys, s=10, color=color)
            
            self.path_artists.extend([line, scat])
        
        # Refresh the canvas
        self.fig.canvas.draw_idle()
        plt.pause(0.1) # Small pause to allow the UI to update

    def show(self):
        plt.show()

    def save(self, filename):
        self.fig.savefig(f"EvolutionaryComputation/OEMEP/{filename}")