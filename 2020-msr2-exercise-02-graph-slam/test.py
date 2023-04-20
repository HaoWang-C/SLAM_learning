import ex2 as ex
import numpy as np

# load a dataset 
filename = 'data/simulation-pose-pose.g2o'
graph = ex.read_graph_g2o(filename)

# visualize the dataset
# ex.plot_graph(graph)
print('Loaded graph with {} nodes and {} edges'.format(len(graph.nodes), len(graph.edges)))

total_error = ex.compute_global_error(graph)
print("Initial total error: ", total_error)

ex.run_graph_slam(graph, 100)