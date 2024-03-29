from matplotlib import pyplot as plt
import numpy as np

class PyplotVisualizer():

    def __init__(self, limits = 5):
        self.ax = plt.axes(projection='3d')
        self.ax.set(xlim=(-limits, limits), 
                    ylim=(-limits, limits),
                    zlim=(-limits,limits))
        self.ax.view_init(0, vertical_axis='y', azim=200, roll=180)
        self.ax.set_box_aspect([1,1,1])
        self.ax.set_facecolor('black')
        plt.axis('off')

    def draw_camera(self, 
                   translation,
                   orientation,
                   scale = 1,
                   face_color = 'blue',
                   body_color = 'red'):

        # square front face
        sq_face = np.array([[scale, -scale, -scale, scale, scale],
                            [scale, scale, -scale, -scale, scale],
                            [0, 0, 0, 0, 0]])

        # camera body
        body_z = np.array([[0, 0, -scale]]).T

        # transform to proper location and orientation
        sq_face = orientation @ sq_face + translation
        body_z = orientation @ body_z + translation


        # draw
        self.ax.plot3D(sq_face[0,:], sq_face[1,:], sq_face[2,:], face_color)
        for i in range(4):
            self.ax.plot3D([sq_face[0,i].item(), body_z[0]], 
                           [sq_face[1,i].item(), body_z[1]], 
                           [sq_face[2,i].item(), body_z[2]], body_color)

    def show(self):
        plt.show()
