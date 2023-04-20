import ex2 as ex
import numpy as np
import pdb

# load a dataset 
filename = 'data/dlr.g2o'
graph = ex.read_graph_g2o(filename)

# visualize the dataset
# 41 pose
# 36 landmarks
# -> 195 state variables
ex.plot_graph(graph)

total_error = ex.compute_global_error(graph)
print("Initial total error: ", total_error)

ex.run_graph_slam(graph, 100)