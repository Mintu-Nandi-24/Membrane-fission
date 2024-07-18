# This code simulates the cellular membrane invagination
# due to application of force

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#import sys

######################## Parameters ##########################
num_steps = 1000                                     # monte-carlo steps
initial_temperature = 100.0                         # Temperature
cooling_rate = 0.99                                 # Temp cooling rate
kappa = 80                                          # bending modulus (pN.nm)
kappa_gaussian = 0.0 #10.0                          # Gaussian Curvature modulus (pN.nm)
c0 = 0.1                                            # Intrinsic curvature (nm^-1)
kbt = 4.1                                           # KB*T: Thermal energy (pN.nm)
sigma = 0.5 * kbt                                   # surface tension (pN.nm^-1)
force_radial = 2.0                                    # Radial force (pN)
force_z = 10.0                                       # Z force (pN)
center_radius = 0.2                                 # From which deformation occurs (nm)
perturbation_scale = 0.001                          # percentage of perturbation
#tau = 0.5  # Line tension parameters

nv = 31                                             #no of vertices along each direction
length = 1.0
num_vertices_x = nv
num_vertices_y = nv
edge_length = length/nv
num_vertices = nv**2                                #no of total vertices
####################################################################
# Initial flat membrane
x = np.linspace(0, (num_vertices_x) * edge_length, num_vertices_x)
y = np.linspace(0, (num_vertices_y) * edge_length, num_vertices_y)
x, y = np.meshgrid(x, y)
z = np.zeros_like(x)

################## Create triangles based on the grid ##################
triangles = []
vertex_coordinates_set = set()  # Store the unique coordinates of all vertices

for i in range(num_vertices_x-1):
    for j in range(num_vertices_y-1):
        # Define vertices for each quad
        v1 = (x[i, j], y[i, j], z[i, j])
        v2 = (x[i + 1, j], y[i + 1, j], z[i + 1, j])
        v3 = (x[i, j + 1], y[i, j + 1], z[i, j + 1])
        v4 = (x[i + 1, j + 1], y[i + 1, j + 1], z[i + 1, j + 1])

        # Create two triangles from each quad
        triangles.append([v1, v2, v3])
        triangles.append([v2, v4, v3])
        
        # Store the unique coordinates of the vertices
        vertex_coordinates_set.update([v1, v2, v3, v4])

# Convert the set to a list
vertex_coordinates = list(vertex_coordinates_set)
vertices = np.array(vertex_coordinates)

# Store original vertices and triangles
original_vertices = np.copy(vertices)
original_triangles = np.copy(triangles)

####### Calculation of no of triangels and edges ######################
num_triangles = len(triangles)

# Find unique edges
unique_edges = set()
for tri in triangles:
    for i in range(3):
        edge = frozenset([tri[i], tri[(i + 1) % 3]])
        unique_edges.add(edge)

edges = [(tri[i], tri[(i + 1) % 3]) for tri in triangles for i in range(3)]

# Calculate the number of edges
num_edges = len(unique_edges)

#print(num_triangles)
#print(num_edges)

#########calculate_average_edge_length(vertices):
# Calculate pairwise distances between vertices
pairwise_distances = np.linalg.norm(vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :], axis=2)

# Exclude self-distances and duplicate distances (upper triangular part)
tri_upper = np.triu(pairwise_distances, k=1)

# Count the number of distances considered
num_distances = np.sum(tri_upper > 0)

# Calculate the sum of edge lengths
sum_edge_lengths = np.sum(tri_upper)

# Calculate the average edge length
avg_edge_length = sum_edge_lengths / num_distances

######################################################################
# Function to calculate reference area of the initial flat membrane
total_area_i = np.zeros(vertices.shape[0])
    
for i in range(vertices.shape[0]):
    # Find the indices of triangles containing vertex i
    tri_indices = [idx for idx, tri in enumerate(triangles) if i in tri]

    # Calculate the total area and reference area at vertex i
    area_i = 0.0
    for tri_idx in tri_indices:
        tri = triangles[tri_idx]
       # Get the vertices of the triangle
        index_i = np.where(np.array(tri) == i)[0][0]  # Find the index of i in the triangle
        index_j1 = (index_i + 1) % 3
        index_j2 = (index_i + 2) % 3

        A, B, C = vertices[tri]

        # Calculate the cross product and area
        cross_product = np.cross(B - A, C - A)
        area_i += 0.5 * np.linalg.norm(cross_product)

    total_area_i[i] = area_i
    
reference_area = total_area_i.sum()  # Use the initial total area as the reference area

######################################################################
################ Visualize the flat membrane #########################
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the triangles
# for tri in triangles:
#     poly3d = Poly3DCollection([tri], facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.9)
#     ax.add_collection3d(poly3d)
    
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Initial Flat Membrane')

# plt.show()
######################################################################
# # Apply intrinsic curvature
# # Calculate the distance of each vertex from the center
# vertex_distance = np.linalg.norm(vertices[:, :2] - 0.5, axis=1)

# # Intrinsic curvature force component
# intrinsic_curvature_force = c0 * np.exp(-vertex_distance**2 / (2 * center_radius**2))

# # Apply forces to induce intrinsic curvature
# vertices[:, 2] += intrinsic_curvature_force

# # Clip z values to ensure non-negative values
# vertices[:, 2] = np.maximum(vertices[:, 2], 0.0)


# Apply intrinsic curvature
# Calculate the distance of each vertex from the center
vertex_distance = np.linalg.norm(vertices[:, :2] - 0.5, axis=1)

# Intrinsic curvature force component
intrinsic_curvature_force = -c0 * np.exp(-vertex_distance**2 / (2 * center_radius**2))

# Apply forces to induce intrinsic curvature
# For positive curvature, deform in +z direction
# For negative curvature, deform in -z direction
vertices[:, 2] += intrinsic_curvature_force if c0 > 0 else -intrinsic_curvature_force

# Clip z values to ensure non-negative values
#vertices[:, 2] = np.maximum(vertices[:, 2], 0.0)

######################################################################
# Calculate the distance of each vertex from the center
#vertex_distance = np.linalg.norm(vertices[:, :2] - 0.5, axis=1)

# Z-force component
z_component = force_z * np.exp(-vertex_distance**2 / (2 * center_radius**2))

# Inward radial force component
radial_component = -force_radial * np.exp(-vertex_distance**2 / (2 * center_radius**2))

# Apply forces to uplift the center and shrink the bud radius
vertices[:, 2] += z_component
vertices[:, :2] += radial_component[:, np.newaxis] * (vertices[:, :2] - 0.5)

# Clip z values to ensure non-negative values
#vertices[:, 2] = np.maximum(vertices[:, 2], 0.0)

############################################################################
# Update the triangles with deformed vertices
deformed_triangles = []
for i in range(0, len(triangles), 2):
    v1, v2, v3 = triangles[i]
    v4, v5, v6 = triangles[i + 1]
      
    deformed_triangles.append([vertex_coordinates.index(tuple(v1)), vertex_coordinates.index(tuple(v2)), vertex_coordinates.index(tuple(v3))])
    deformed_triangles.append([vertex_coordinates.index(tuple(v4)), vertex_coordinates.index(tuple(v5)), vertex_coordinates.index(tuple(v6))])

triangles = deformed_triangles

##########################################################################
# Function to calculate total area
total_area = np.zeros(vertices.shape[0])
    
for i in range(vertices.shape[0]):
    # Find the indices of triangles containing vertex i
    tri_indices = [idx for idx, tri in enumerate(triangles) if i in tri]

    # Calculate the total area and reference area at vertex i
    area_i = 0.0
    for tri_idx in tri_indices:
        tri = triangles[tri_idx]
        # Get the vertices of the triangle
        index_i = np.where(np.array(tri) == i)[0][0]  # Find the index of i in the triangle
        index_j1 = (index_i + 1) % 3
        index_j2 = (index_i + 2) % 3

        A, B, C = vertices[tri]

        # Calculate the cross product and area
        cross_product = np.cross(B - A, C - A)
        area_i += 0.5 * np.linalg.norm(cross_product)

    total_area[i] = area_i
    
    
# Calculate the gradient of the surface tension energy with respect to vertices
surface_tension_gradient = np.zeros_like(vertices)
for i in range(vertices.shape[0]):
    # Find the indices of triangles containing vertex i
    tri_indices = [idx for idx, tri in enumerate(triangles) if i in tri]

    # Calculate the gradient of the surface tension energy at vertex i
    gradient_i = np.zeros(3)
    for tri_idx in tri_indices:
        tri = triangles[tri_idx]
        # Get the vertices of the triangle
        index_i = np.where(np.array(tri) == i)[0][0]  # Find the index of i in the triangle
        index_j1 = (index_i + 1) % 3
        index_j2 = (index_i + 2) % 3

        A, B, C = vertices[tri]

        # Calculate the cross product and area
        cross_product = np.cross(B - A, C - A)
        area_i = 0.5 * np.linalg.norm(cross_product)

        # Calculate the gradient contribution from the current triangle
        gradient_i += sigma * np.sign(total_area[i] - reference_area) * area_i * np.cross(B - A, C - A)

    surface_tension_gradient[i] = gradient_i

# Update the vertices based on the surface tension force
vertices += surface_tension_gradient

###########################################################################
# # Line Tension 
# # Identify the middle vertices based on a Y-range
# middle_vertices_indices = np.where((vertices[:, 1] >= 0.4) & (vertices[:, 1] <= 0.6))[0]

# # Print the indices and check if any vertices are selected
# #print("Middle Vertices Indices:", middle_vertices_indices)

# # Check if there are any vertices within the specified Y-range
# if len(middle_vertices_indices) > 0:
#     middle_vertices = vertices[middle_vertices_indices]

#     # Calculate the line tension force on the middle vertices
#     line_tension_force = -tau * (middle_vertices - 0.5)

#     # Apply the line tension force to the middle vertices
#     vertices[middle_vertices_indices] += line_tension_force
# #else:
# #    print("No vertices found in the specified Y-range.")

# ... (previous code)

# # Identify the middle vertices at the center circle of the bud
# center_circle_radius = 0.1  # Adjust the radius as needed
# center_circle_center = np.array([0.5, 0.5, 0.5])  # Adjust the center coordinates as needed
# middle_vertices_indices = np.where(np.linalg.norm(vertices[:, :2] - center_circle_center[:2], axis=1) <= center_circle_radius)[0]

# # Check if there are any vertices within the center circle
# if len(middle_vertices_indices) > 0:
#     # Calculate the direction vector towards the center
#     direction_vector = center_circle_center - vertices

#     # Normalize the direction vector
#     norm_direction_vector = direction_vector / np.linalg.norm(direction_vector, axis=1)[:, np.newaxis]

#     # Define the magnitude of the line tension force
#     line_tension_magnitude = tau

#     # Calculate the line tension force vector
#     line_tension_force = line_tension_magnitude * norm_direction_vector

#     # Apply the line tension force to the vertices on the center circle
#     vertices[middle_vertices_indices] += line_tension_force[middle_vertices_indices]

#sys.exit()
##########################################################################
# # Update the triangles with deformed vertices
# deformed_triangles = []
# for i in range(0, len(triangles), 2):
#     v1, v2, v3 = triangles[i]
#     v4, v5, v6 = triangles[i + 1]
      
#     deformed_triangles.append([vertex_coordinates.index(tuple(v1)), vertex_coordinates.index(tuple(v2)), vertex_coordinates.index(tuple(v3))])
#     deformed_triangles.append([vertex_coordinates.index(tuple(v4)), vertex_coordinates.index(tuple(v5)), vertex_coordinates.index(tuple(v6))])

# triangles = deformed_triangles
##########################################################################
################ Visualize the deformed membrane #########################
# plt.clf()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], cmap='viridis')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Deformed Membrane with forces')
# plt.show()


# # Visualize the deformed membrane with the same triangulations as the initial flat membrane
# plt.clf()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the triangles with deformed vertices
# for tri in triangles:
#     poly3d = Poly3DCollection([vertices[tri]], facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.9)
#     ax.add_collection3d(poly3d)

# # Plot the connecting edges in the deformed membrane
# for edge in unique_edges:
#     v1, v2 = edge
#     ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 'k-')

# # Plot the deformed vertices
# #ax.scatter([v[0] for v in vertices], [v[1] for v in vertices], [v[2] for v in vertices], c='blue', marker='o')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Deformed Membrane with forces')

# plt.show()
######################################################################
# Calculate the mean curvature at each vertex
mean_curvature = np.zeros(vertices.shape[0])

for i in range(vertices.shape[0]):
    # Find neighboring vertices
    neighbors = [j for j in range(vertices.shape[0]) if j != i]

    # Calculate the gradient of the area (F_i)
    area_gradient = np.zeros(3)
    for tri in triangles:
        if i in tri:
            # Identify the indices of the current vertex in the triangle
            indices = [idx for idx, vertex_idx in enumerate(tri) if vertex_idx == i]
            if indices:
                index_i = indices[0]
                index_j1 = (index_i + 1) % 3
                index_j2 = (index_i + 2) % 3

                normal = np.cross(vertices[tri[index_j1]] - vertices[i], vertices[tri[index_j2]] - vertices[i])
                area_gradient += normal / np.linalg.norm(normal)

    # Calculate the gradient of the volume (N_i)
    volume_gradient = np.zeros(3)
    for j in neighbors:
        volume_gradient += vertices[j] - vertices[i]

    # Calculate the mean curvature at vertex i
    volume_gradient_norm = np.linalg.norm(volume_gradient)
    if volume_gradient_norm > 1e-10:  # Avoid division by zero
        mean_curvature[i] = 0.5 * np.dot(area_gradient, volume_gradient / volume_gradient_norm)
    else:
        mean_curvature[i] = 0.0

# Print or visualize the mean curvature values
#print('mean curvature', mean_curvature)

# ########################################################################
# Calculate the total area Ai for each vertex
total_area = np.zeros(vertices.shape[0])
# Calculate Gaussian curvature at each vertex
gaussian_curvature = np.zeros(vertices.shape[0])

for i in range(vertices.shape[0]):
    # Find the indices of triangles containing vertex i
    tri_indices = [idx for idx, tri in enumerate(triangles) if i in tri]
    
    normal_sum = np.zeros(3)
    area_sum = 0.0

    # Calculate the total area Ai
    for tri_idx in tri_indices:
        tri = triangles[tri_idx]
        # Get the vertices of the triangle
        index_i = np.where(np.array(tri) == i)[0][0]  # Find the index of i in the triangle
        index_j1 = (index_i + 1) % 3
        index_j2 = (index_i + 2) % 3

        A, B, C = vertices[tri]

        # Calculate the cross product and area
        cross_product = np.cross(B - A, C - A)
        area = 0.5 * np.linalg.norm(cross_product)
        
        # Calculate the normal and update sums
        normal_sum += cross_product / (2 * area)
        area_sum += area

    total_area[i] = area_sum
    
    # Calculate the Gaussian curvature at vertex i
    norm_normal_sum = np.linalg.norm(normal_sum)
    if norm_normal_sum > 1e-10 and area_sum > 1e-10:  # Avoid division by zero
        gaussian_curvature[i] = 1.0 / area_sum * norm_normal_sum
    else:
        gaussian_curvature[i] = 0.0

# #########################################################################
# Calculate the bending energy
bending_energy = 0.5 * np.sum((1/3) * kappa * (mean_curvature - c0)**2 * total_area)

# Calculate energy due to Gaussian curvature
gaussian_curvature_energy = 0.5 * np.sum(kappa_gaussian * gaussian_curvature**2 * total_area)


# Print or visualize the bending energy
#print("Bending Energy:", bending_energy)

# Calculate the surface tension energy
surface_tension_energy = np.sum(sigma * np.abs(total_area - reference_area))

# Print or visualize the surface tension energy
#print("Surface Tension Energy:", surface_tension_energy)

# Calculate the energy due to radial force
radial_energy = 0.5 * np.sum(force_radial**2 * np.linalg.norm(vertices[:, :2] - 0.5, axis=1)**2)

# Print or visualize the radial energy
#print("Radial Energy:", radial_energy)

# Calculate the energy due to z-component force
z_energy = 0.5 * np.sum(force_z**2 * np.exp(-vertex_distance**2 / (center_radius**2)))

# Print or visualize the z-energy
#print("Z-component Energy:", z_energy)

# Calculate the line tension energy
#line_tension_energy = np.sum(tau * np.linalg.norm(middle_vertices - 0.5, axis=1))
##line_tension_energy = np.sum(tau * np.linalg.norm(line_tension_force[middle_vertices_indices], axis=1))


# Total energy including bending, surface tension, radial, and z-component energies
total_energy1 = bending_energy + gaussian_curvature_energy + radial_energy + z_energy + surface_tension_energy #+ line_tension_energy

# Print or visualize the total energy
#print("Total Energy:", bending_energy)

best_vertices = vertices
best_energy = total_energy1
best_triangles = triangles

#########################################################################
x1 = np.zeros(num_steps)
x2 = np.zeros(num_steps)
x3 = np.zeros(num_steps)

temperature = initial_temperature

# start of monte carlo loop
for step in range(num_steps):
    
    # Generate random displacements from a cube [-0.05 * L, 0.05 * L]^3
    perturbation = perturbation_scale * (2.0 * np.random.rand(len(vertices), 3) - 1.0) * avg_edge_length
    # Apply perturbation to vertices
    vertices += perturbation

    ## Perform edge flipping
    edge_index = np.random.randint(0, len(triangles))
    edge = triangles[edge_index][:2]

    # Find triangles sharing the edge
    edge_triangles = [i for i, simplex in enumerate(triangles) if set(edge).issubset(simplex)]

    if len(edge_triangles) == 2:
        # Find vertices opposite to the edge in each triangle
        v_opposite1 = list(set(triangles[edge_triangles[0]]) - set(edge))[0]
        v_opposite2 = list(set(triangles[edge_triangles[1]]) - set(edge))[0]

        # Ensure the vertices form a convex quadrilateral
        if v_opposite2 not in triangles[edge_triangles[0]] and v_opposite1 not in triangles[edge_triangles[1]]:
            # Update the triangles to flip the edge
            triangles[edge_triangles[0]] = [v_opposite2, edge[0], v_opposite1, edge[1]]
            triangles[edge_triangles[1]] = [v_opposite1, edge[1], v_opposite2, edge[0]]

    old_vertices = vertices
    old_triangles = triangles
    
    ###################################################################
    # Calculate the total area Ai for each vertex
    total_area = np.zeros(vertices.shape[0])
    # Calculate Gaussian curvature at each vertex
    gaussian_curvature = np.zeros(vertices.shape[0])

    for i in range(vertices.shape[0]):
        # Find the indices of triangles containing vertex i
        tri_indices = [idx for idx, tri in enumerate(triangles) if i in tri]
        
        normal_sum = np.zeros(3)
        area_sum = 0.0

        # Calculate the total area Ai
        for tri_idx in tri_indices:
            tri = triangles[tri_idx]
            # Get the vertices of the triangle
            index_i = np.where(np.array(tri) == i)[0][0]  # Find the index of i in the triangle
            index_j1 = (index_i + 1) % 3
            index_j2 = (index_i + 2) % 3
            
            A = vertices[tri[index_j1]]
            B = vertices[tri[index_j2]]
            C = vertices[tri[index_i]]

            # Calculate the cross product and area
            cross_product = np.cross(B - A, C - A)
            area = 0.5 * np.linalg.norm(cross_product)
            
            # Calculate the normal and update sums
            normal_sum += cross_product / (2 * area)
            area_sum += area

        total_area[i] = area_sum
        
        # Calculate the Gaussian curvature at vertex i
        norm_normal_sum = np.linalg.norm(normal_sum)
        if norm_normal_sum > 1e-10 and area_sum > 1e-10:  # Avoid division by zero
            gaussian_curvature[i] = 1.0 / area_sum * norm_normal_sum
        else:
            gaussian_curvature[i] = 0.0 
    ######################################################################    
    # Calculate the mean curvature at each vertex
    mean_curvature = np.zeros(vertices.shape[0])

    for i in range(vertices.shape[0]):
        # Find neighboring vertices
        neighbors = [j for j in range(vertices.shape[0]) if j != i]

        # Calculate the gradient of the area (F_i)
        area_gradient = np.zeros(3)
        for tri in triangles:
            if i in tri:
                # Identify the indices of the current vertex in the triangle
                indices = [idx for idx, vertex_idx in enumerate(tri) if vertex_idx == i]
                if indices:
                    index_i = indices[0]
                    index_j1 = (index_i + 1) % 3
                    index_j2 = (index_i + 2) % 3

                    normal = np.cross(vertices[tri[index_j1]] - vertices[i], vertices[tri[index_j2]] - vertices[i])
                    area_gradient += normal / np.linalg.norm(normal)

        # Calculate the gradient of the volume (N_i)
        volume_gradient = np.zeros(3)
        for j in neighbors:
            volume_gradient += vertices[j] - vertices[i]

        # Calculate the mean curvature at vertex i
        volume_gradient_norm = np.linalg.norm(volume_gradient)
        if volume_gradient_norm > 1e-10:  # Avoid division by zero
            mean_curvature[i] = 0.5 * np.dot(area_gradient, volume_gradient / volume_gradient_norm)
            #print('pass')
        else:
            mean_curvature[i] = 0.0

    # Print or visualize the mean curvature values
    #print(mean_curvature,vertices)
    
    ## calulation of energies
    # Calculate the bending energy
    bending_energy = 0.5 * np.sum((1/3) * kappa * (mean_curvature - c0)**2 * total_area)
    # Calculate energy due to Gaussian curvature
    gaussian_curvature_energy = 0.5 * np.sum(kappa_gaussian * gaussian_curvature**2 * total_area)
    # Calculate the surface tension energy
    surface_tension_energy = np.sum(sigma * np.abs(total_area - reference_area))
    # Calculate the energy due to radial force
    radial_energy = 0.5 * np.sum(force_radial**2 * np.linalg.norm(vertices[:, :2] - 0.5, axis=1)**2)
    # Calculate the energy due to z-component force
    z_energy = 0.5 * np.sum(force_z**2 * np.exp(-vertex_distance**2 / (center_radius**2)))
    # Identify the middle vertices based on a Y-range
    ##middle_vertices_indices = np.where((vertices[:, 1] >= 0.4) & (vertices[:, 1] <= 0.6))[0]
    # Calculate the line tension energy
    #line_tension_energy = np.sum(tau * np.linalg.norm(middle_vertices - 0.5, axis=1))
    total_energy = bending_energy + gaussian_curvature_energy + radial_energy + z_energy + surface_tension_energy #+ line_tension_energy

    delta_energy = total_energy - total_energy1
    
    #print(total_energy, total_energy1)

    if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
        if total_energy < total_energy1:
            best_vertices = vertices
            best_energy = total_energy
            best_triangles = triangles
            total_energy1 = best_energy
            vertices = best_vertices
            triangles = best_triangles
            print(step + 1, "ACCEPT+BEST", delta_energy, best_energy)
        else:
            total_energy1 = best_energy
            vertices = vertices
            triangles = triangles
            #print(step + 1, "ACCEPT+METRO", delta_energy, best_energy)
    else:
        total_energy1 = total_energy1
        vertices = old_vertices
        triangles = old_triangles
        #print(step + 1, "REJECT", delta_energy, best_energy)

    temperature *= cooling_rate

    x1[step] = step + 1
    x2[step] = best_energy
    x3[step] = temperature
    
# Calculate the radius and height for each vertex
radius_final = np.linalg.norm(best_vertices[:, :2] - 0.5, axis=1)
height_final = best_vertices[:, 2]

# Get indices to sort height in descending order
height_sort_indices = np.argsort(height_final)[::-1]

# Arrange radius_final and height_final using the sorted indices
radius_final_sorted = radius_final[height_sort_indices]
height_final_sorted = height_final[height_sort_indices]

reversed_radius = -radius_final_sorted[::-1]
reversed_height = height_final_sorted[::-1]
combined_radius = np.concatenate((reversed_radius, radius_final_sorted))
combined_height = np.concatenate((reversed_height, height_final_sorted))
combined_data = np.column_stack((combined_height, combined_radius))
np.savetxt('fig-height-vs-radius-fz10-fr2-sigp5-cp1.dat', combined_data, comments='')
#############
#sys.exit()

# Assuming best_triangles is a list of triangles, where each triangle is a list of three vertices
expected_length = 3
# Filter triangles to ensure each triangle has the expected length
best_triangles = [tri for tri in best_triangles if len(tri) == expected_length]
# Convert the list of triangles to a NumPy array
triangles_array = np.array(best_triangles)
# Ensure all triangles have the same length by reshaping the array
triangles_array = triangles_array.reshape(-1, expected_length)
# Save the vertices and triangles to files
np.savetxt('vertices-fz10-fr2-sigp5-cp1.dat', best_vertices, comments='')
np.savetxt('triangles-fz10-fr2-sigp5-cp1.dat', triangles_array, comments='')


# Visualize the 3D configuration with colored triangles
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Calculate color gradient based on the Z-coordinate of triangle centroids
centroid_z = np.zeros(len(triangles))

for i, tri in enumerate(triangles):
    centroid_z[i] = np.mean(best_vertices[tri, 2])

# Now you can use centroid_z in your visualization code
colors = plt.cm.viridis((centroid_z - min(centroid_z)) / (max(centroid_z) - min(centroid_z)))

for i, tri in enumerate(triangles):
    poly3d = Poly3DCollection([best_vertices[tri]], facecolors=[colors[i]], linewidths=1, edgecolors='c', alpha=0.9)
    ax.add_collection3d(poly3d)

# Plot the connecting edges in the deformed membrane
for edge in unique_edges:
    v1, v2 = edge
    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 'k-')

ax.set_zlim(0, 10)
ax.set_axis_off()
plt.savefig('fig-3d-tri-fz10-fr2-sigp5-cp1.pdf', format='pdf', bbox_inches='tight')
plt.show()


## SA steps vs energy
plt.plot(x1, x2, '-')
plt.xlabel('SA Steps')
plt.ylabel('Energy (pN.nm)')
plt.savefig('fig-energy-fz10-fr2-sigp5-cp1.pdf', format='pdf', bbox_inches='tight')
plt.show()

## Temp vs energy
plt.plot(x3, x2, '-')
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.show()

