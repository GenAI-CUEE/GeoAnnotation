from math import floor
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tifffile

from plantcv.plantcv.annotate.points import _find_closest_pt

class Find_correspondences:
    """Corrsponding points collection  
    """

    def __init__(self, imgA, imgB, marker_A='x', marker_B="x", marker_color_A="red", marker_color_B="red", figsize=(12, 6)):
        """
        Initialization
        :param img: image data
        :param figsize: desired figure size, (12,6) by default
        :attribute points: list of points as (x,y) coordinates tuples
        """ 
        self.markers = {}
        self.markers[0] = marker_A
        self.markers[1] = marker_B
        self.markers_color = {}
        self.markers_color[0] = marker_color_A
        self.markers_color[1] = marker_color_B
        

        self.fig, self.axs = plt.subplots(1, 2, figsize=figsize) 
        self.axs[0].imshow(imgA[:,:,0:3])  # Display the first three channels (RGB) of the clipped image
        self.axs[0].set_title('Google') # Set a title for the first subplot 

        self.axs[1].imshow(imgB[:,:,0:3])  # Display the first three channels (RGB) of the clipped satellite image
        self.axs[1].set_title('Theos') # Set a title for the second subplot 


        self.points = {}
        self.points[0] = []
        self.points[1] = []


        self.events = {}
        self.events[0] = []
        self.events[1] = []
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)


    def click(self, event, which_axis=0):
    
        if event.button == 1:

            self.axs[which_axis].plot(event.xdata, event.ydata, self.markers[which_axis], c=self.markers_color[which_axis])
            self.points[which_axis].append((floor(event.xdata), floor(event.ydata)))

        else:
            idx_remove, _ = _find_closest_pt((event.xdata, event.ydata), self.points[which_axis])
            # remove the closest point to the user right clicked one
            self.points[which_axis].pop(idx_remove)
            self.axs[which_axis].lines[idx_remove].remove()


    def onclick(self, event):
        """Handle mouse click events"""
        
        if event.inaxes is not None:
            clicked_ax = event.inaxes
            
            if clicked_ax is self.axs[0]:
                self.axs[0].set_title("On")
                self.axs[1].set_title("Off")
                self.events[0].append(event)
                self.click(event, which_axis=0)

            elif clicked_ax is self.axs[1]:
                self.axs[0].set_title("Off")
                self.axs[1].set_title("On")
                self.events[1].append(event)
                self.click(event,  which_axis=1)
                
            self.fig.canvas.draw()
        else:
            print("Clicked outside any axes")
        
         