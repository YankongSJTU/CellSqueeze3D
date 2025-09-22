import numpy as np
from math import atan2
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from scipy.spatial import distance
import json
from math import pi,cos,sin
def visualize_cells(he_image, cells,imgname):
    
    mask = np.zeros_like(he_image[:, :, 0], dtype=np.uint8)
    for cell in cells:
        center = (int(cell['center'][0]), int(cell['center'][1]))
        radius = int(cell['radius'])
        cv2.circle(mask, center, radius, 255, -1)   
    for cell in cells:
        for other_cell in cells:
            if other_cell == cell:
                continue
            shared_boundary = calculate_shared_boundary(cell, other_cell)
            if shared_boundary:
                cv2.line(mask,
                         (int(shared_boundary[0][0]), int(shared_boundary[0][1])),
                         (int(shared_boundary[1][0]), int(shared_boundary[1][1])),
                         0, 2)   
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(he_image, contours, -1, (255, 0, 0), 2)
    he_image = draw_shared_boundaries(he_image, cells)
    cv2.imwrite(imgname+'.output_image.png', he_image)
    return he_image
def draw_shared_boundaries(he_image, cells):
    for i, cell in enumerate(cells):
        for j, other_cell in enumerate(cells):
            if i >= j:
                continue   
            shared_boundary = calculate_shared_boundary(cell, other_cell)
            if shared_boundary:
                cv2.line(he_image,
                         (int(shared_boundary[0][0]), int(shared_boundary[0][1])),
                         (int(shared_boundary[1][0]), int(shared_boundary[1][1])),
                         (0, 255, 0), 2)   
    return he_image
def get_cell_boundaries(cells,imgid):
    boundaries = []
    for i, cell in enumerate(cells):
        center = (cell['center'][0], cell['center'][1],cell['center'][2])
        radius = cell['radius']
        nuc_radius=cell['nucleus_radius']
        full_arc = [(0, 2 * pi)]
        shared_boundaries = []
        for j, other_cell in enumerate(cells):
            if i == j:
                continue
            shared_boundary = calculate_shared_boundary(cell, other_cell)
            if shared_boundary:
                intersection1, intersection2 = shared_boundary
                angle1 = atan2(intersection1[1] - center[1], intersection1[0] - center[0])
                angle2 = atan2(intersection2[1] - center[1], intersection2[0] - center[0])
                angle1 = angle1 if angle1 >= 0 else angle1 + 2 * pi
                angle2 = angle2 if angle2 >= 0 else angle2 + 2 * pi
                if angle1 > angle2:   
                    angle1, angle2 = angle2, angle1
                shared_boundaries.append(shared_boundary)
                new_arcs = []
                for arc in full_arc:
                    start_angle, end_angle = arc
                    if angle1 <= start_angle and angle2 >= end_angle:
                        continue   
                    if start_angle < angle1 < end_angle:
                        new_arcs.append((start_angle, angle1))
                    if start_angle < angle2 < end_angle:
                        new_arcs.append((angle2, end_angle))
                full_arc = new_arcs
        if not full_arc:
            full_arc.append((0, 2 * pi))   
        boundary_points = []
        for arc in full_arc:
            start_angle, end_angle = arc
            num_points = max(10, int(50 * (end_angle - start_angle) / (2 * pi)))   
            for t in np.linspace(start_angle, end_angle, num_points):
                x = center[0] + radius * np.cos(t)
                y = center[1] + radius * np.sin(t)
                boundary_points.append((x, y))
        for shared_boundary in shared_boundaries:
            boundary_points.append(shared_boundary[0])
            boundary_points.append(shared_boundary[1])
        boundary_points = sorted(boundary_points, key=lambda p: atan2(p[1] - center[1], p[0] - center[0]))
        boundary_polygon = np.array(boundary_points, dtype=np.int32)
        boundaries.append({
            'center': center,
            'radius': radius,
            'boundary': boundary_polygon.tolist(),
            'nuc_radius':nuc_radius
        })
#    with open(imgid+'cell_boundaries.json', 'w') as f:
#        json.dump(boundaries, f)
    return boundaries

def calculate_shared_boundary(cell1, cell2):
    center1 = np.array(cell1['center'][:2])
    center2 = np.array(cell2['center'][:2])
    radius1 = cell1['radius']
    radius2 = cell2['radius']
    d = np.linalg.norm(center1 - center2)
    if d >= radius1 + radius2 or d == 0:
        return []
    a = (radius1**2 - radius2**2 + d**2) / (2 * d)
    h_squared = radius1**2 - a**2
    if h_squared < 0:
        return []
    h = np.sqrt(h_squared)
    mid_point = center1 + a * (center2 - center1) / d
    dx = (center2[0] - center1[0]) / d
    dy = (center2[1] - center1[1]) / d
    intersection1 = (mid_point[0] + h * dy, mid_point[1] - h * dx)
    intersection2 = (mid_point[0] - h * dy, mid_point[1] + h * dx)
    return [intersection1, intersection2]
from mpl_toolkits.mplot3d import Axes3D
def visualize_3d_cells(cells, save_path='3d_distribution.png'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for cell in cells:
        x, y, z = cell['center']
        radius = cell['radius']
  
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = x + radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = y + radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = z + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                        color='b', alpha=0.3, edgecolor='none')
   
        ax.scatter(x, y, z, color='r', s=20)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Cell Distribution')
    plt.savefig(save_path)
    plt.close()
