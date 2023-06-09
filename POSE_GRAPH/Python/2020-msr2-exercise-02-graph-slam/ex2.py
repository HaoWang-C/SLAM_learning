import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import pdb
from numpy.linalg import inv

# Helper functions to get started


class Graph:
    def __init__(self, x, nodes, edges, lut):
        self.x = x
        self.nodes = nodes
        self.edges = edges
        self.lut = lut


def read_graph_g2o(filename):
    """ This function reads the g2o text file as the graph class 

    Parameters
    ----------
    filename : string
        path to the g2o file

    Returns
    -------
    graph: Graph contaning information for SLAM 

    """
    Edge = namedtuple(
        'Edge', ['Type', 'fromNode', 'toNode', 'measurement', 'information'])
    edges = []
    nodes = {}
    with open(filename, 'r') as file:
        for line in file:
            data = line.split()

            if data[0] == 'VERTEX_SE2':
                nodeId = int(data[1])
                pose = np.array(data[2:5], dtype=np.float32)
                nodes[nodeId] = pose

            elif data[0] == 'VERTEX_XY':
                nodeId = int(data[1])
                loc = np.array(data[2:4], dtype=np.float32)
                nodes[nodeId] = loc

            elif data[0] == 'EDGE_SE2':
                Type = 'P'
                fromNode = int(data[1])
                toNode = int(data[2])
                measurement = np.array(data[3:6], dtype=np.float32)
                uppertri = np.array(data[6:12], dtype=np.float32)
                information = np.array(
                    [[uppertri[0], uppertri[1], uppertri[2]],
                     [uppertri[1], uppertri[3], uppertri[4]],
                     [uppertri[2], uppertri[4], uppertri[5]]])
                edge = Edge(Type, fromNode, toNode, measurement, information)
                edges.append(edge)

            elif data[0] == 'EDGE_SE2_XY':
                Type = 'L'
                fromNode = int(data[1])
                toNode = int(data[2])
                measurement = np.array(data[3:5], dtype=np.float32)
                uppertri = np.array(data[5:8], dtype=np.float32)
                information = np.array([[uppertri[0], uppertri[1]],
                                        [uppertri[1], uppertri[2]]])
                edge = Edge(Type, fromNode, toNode, measurement, information)
                edges.append(edge)

            else:
                print('VERTEX/EDGE type not defined')

    # compute state vector and lookup table
    lut = {}
    x = []
    offset = 0
    for nodeId in nodes:
        lut.update({nodeId: offset})
        offset = offset + len(nodes[nodeId])
        x.append(nodes[nodeId])
    x = np.concatenate(x, axis=0)

    # collect nodes, edges and lookup in graph structure
    graph = Graph(x, nodes, edges, lut)
    print('Loaded graph with {} nodes and {} edges'.format(
        len(graph.nodes), len(graph.edges)))

    return graph


def v2t(pose):
    """This function converts SE2 pose from a vector to transformation  

    Parameters
    ----------
    pose : 3x1 vector
        (x, y, theta) of the robot pose

    Returns
    -------
    T : 3x3 matrix
        Transformation matrix corresponding to the vector
    """
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    T = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return T


def t2v(T):
    """This function converts SE2 transformation to vector for  

    Parameters
    ----------
    T : 3x3 matrix
        Transformation matrix for 2D pose

    Returns
    -------
    pose : 3x1 vector
        (x, y, theta) of the robot pose
    """
    x = T[0, 2]
    y = T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    v = np.array([x, y, theta])
    return v


def plot_graph(g):

    # initialize figure
    plt.figure(1)
    plt.clf()

    # get a list of all poses and landmarks
    poses, landmarks = get_poses_landmarks(g)

    # plot robot poses
    if len(poses) > 0:
        poses = np.stack(poses, axis=0)
        plt.plot(poses[:, 0], poses[:, 1], 'bo')

    # plot landmarks
    if len(landmarks) > 0:
        landmarks = np.stack(landmarks, axis=0)
        plt.plot(landmarks[:, 0], landmarks[:, 1], 'r*')

    # plot edges/constraints
    poseEdgesP1 = []
    poseEdgesP2 = []
    landmarkEdgesP1 = []
    landmarkEdgesP2 = []

    for edge in g.edges:
        fromIdx = g.lut[edge.fromNode]
        toIdx = g.lut[edge.toNode]
        if edge.Type == 'P':
            poseEdgesP1.append(g.x[fromIdx:fromIdx + 3])
            poseEdgesP2.append(g.x[toIdx:toIdx + 3])

        elif edge.Type == 'L':
            landmarkEdgesP1.append(g.x[fromIdx:fromIdx + 3])
            landmarkEdgesP2.append(g.x[toIdx:toIdx + 2])

    poseEdgesP1 = np.stack(poseEdgesP1, axis=0)
    poseEdgesP2 = np.stack(poseEdgesP2, axis=0)
    plt.plot(np.concatenate((poseEdgesP1[:, 0], poseEdgesP2[:, 0])),
             np.concatenate((poseEdgesP1[:, 1], poseEdgesP2[:, 1])), 'b')

    landmarkEdgesP1 = np.stack(landmarkEdgesP1, axis=0)
    landmarkEdgesP2 = np.stack(landmarkEdgesP2, axis=0)
    plt.plot([landmarkEdgesP1[:, 0], landmarkEdgesP2[:, 0]],
             [landmarkEdgesP1[:, 1], landmarkEdgesP2[:, 1]], 'r--', alpha=0.3)

    plt.draw()
    plt.pause(1)

    return


def get_poses_landmarks(g):
    poses = []
    landmarks = []

    for nodeId in g.nodes:
        dimension = len(g.nodes[nodeId])
        offset = g.lut[nodeId]

        if dimension == 3:
            pose = g.x[offset:offset + 3]
            poses.append(pose)
        elif dimension == 2:
            landmark = g.x[offset:offset + 2]
            landmarks.append(landmark)

    return poses, landmarks


def run_graph_slam(g, numIterations):

    LM_lambda = 0.01
    # perform optimization
    for i in range(numIterations):

        total_error = compute_global_error(g)
        print("total error: ", total_error)
        print("lambda: ", LM_lambda)
        x_old = g.x

        # compute the incremental update dx of the state vector
        dx = linearize_and_solve(g, LM_lambda)
        # apply the solution to the state vector g.x
        g.x = (g.x.reshape(g.x.shape[0], 1) + dx).reshape(g.x.shape[0],)

        # update or not depending on LM_lambda
        new_total_error = compute_global_error(g)
        if (new_total_error < total_error):
            LM_lambda /= 2
        else:
            LM_lambda *= 2
            g.x = x_old

        # plot graph
        plot_graph(g)
        # compute and print global error
        # terminate procedure if change is less than 10e-4
        if (total_error < 10e-4):
            break


def compute_global_error(g):
    """ This function computes the total error for the graph. 

    Parameters
    ----------
    g : Graph class

    Returns
    -------
    Fx: scalar
        Total error for the graph
    """
    Fx = 0
    for edge in g.edges:

        # pose-pose constraint
        if edge.Type == 'P':

            # compute idx for nodes using lookup table
            fromIdx = g.lut[edge.fromNode]
            toIdx = g.lut[edge.toNode]

            # get node state for the current edge
            x1 = g.x[fromIdx:fromIdx + 3]
            x2 = g.x[toIdx:toIdx + 3]

            # get measurement and information matrix for the edge
            z12 = edge.measurement
            info12 = edge.information

            # (TODO) compute the error due to this edge
            e12 = t2v(inv(v2t(z12)) @ inv(v2t(x1)) @ v2t(x2)).reshape(3, 1)
            Fx += np.transpose(e12) @ info12 @ e12

        # pose-pose constraint
        elif edge.Type == 'L':

            # compute idx for nodes using lookup table
            fromIdx = g.lut[edge.fromNode]
            toIdx = g.lut[edge.toNode]

            # get node states for the current edge
            x = g.x[fromIdx:fromIdx + 3]
            l = g.x[toIdx:toIdx + 2]

            # get measurement and information matrix for the edge
            z = edge.measurement
            info12 = edge.information

            # (TODO) compute the error due to this edge
            zil_hat = compute_l_in_x_frame(x, l)
            eil = zil_hat - z.reshape(2, 1)
            Fx += np.transpose(eil) @ info12 @ eil

    return Fx


def compute_l_in_x_frame(x, l):
    trans = x[0:2]
    R = v2t(x)[0:2, 0:2]
    z_hat = np.transpose(R) @ (l-trans)

    return z_hat.reshape(2, 1)


def linearize_and_solve(g, LM_lambda):
    """ This function solves the least-squares problem for one iteration
        by linearizing the constraints 

    Parameters
    ----------
    g : Graph class

    Returns
    -------
    dx : Nx1 vector 
         change in the solution for the unknowns x
    """

    # initialize the sparse H and the vector b
    H = np.zeros((len(g.x), len(g.x)))
    b = np.zeros(len(g.x)).reshape(1, len(g.x))

    # set flag to fix gauge
    needToAddPrior = True
    Fx = 0

    # compute the addend term to H and b for each of our constraints
    print('linearize and build system')
    index = 0
    for edge in g.edges:
        index += 1
        print("index: ", index, "/17605")
        # pose-pose constraint
        if edge.Type == 'P':

            # compute idx for nodes using lookup table
            fromIdx = g.lut[edge.fromNode]
            toIdx = g.lut[edge.toNode]
            # g.lut -> (nodeID, Starting_Index_Of_StateVariable)

            # get node state for the current edge
            x_i = g.x[fromIdx:fromIdx + 3]
            x_j = g.x[toIdx:toIdx + 3]

            # (TODO) compute the error and the Jacobians
            e, A, B = linearize_pose_pose_constraint(
                x_i, x_j, edge.measurement)

            # (TODO) compute the terms
            b_i = np.transpose(e) @ edge.information @ A
            b_j = np.transpose(e) @ edge.information @ B
            H_ii = np.transpose(A) @ edge.information @ A
            H_ij = np.transpose(A) @ edge.information @ B
            H_ji = np.transpose(B) @ edge.information @ A
            H_jj = np.transpose(B) @ edge.information @ B

            # (TODO) add the terms to H matrix and b
            b[:, fromIdx:fromIdx + 3] += b_i
            b[:, toIdx:toIdx + 3] += b_j

            H[fromIdx:fromIdx + 3, fromIdx:fromIdx + 3] += H_ii
            H[fromIdx:fromIdx + 3, toIdx:toIdx + 3] += H_ij
            H[toIdx:toIdx + 3, fromIdx:fromIdx + 3] += H_ji
            H[toIdx:toIdx + 3, toIdx:toIdx + 3] += H_jj

            # Add the prior for one pose of this edge
            # This fixes one node to remain at its current location
            if needToAddPrior:
                H[fromIdx:fromIdx + 3, fromIdx:fromIdx +
                  3] = H[fromIdx:fromIdx + 3,
                         fromIdx:fromIdx + 3] + 1000 * np.eye(3)
                needToAddPrior = False

        # pose-pose constraint
        elif edge.Type == 'L':

            # compute idx for nodes using lookup table
            fromIdx = g.lut[edge.fromNode]
            toIdx = g.lut[edge.toNode]

            # get node states for the current edge
            x = g.x[fromIdx:fromIdx + 3]
            l = g.x[toIdx:toIdx + 2]

            # (TODO) compute the error and the Jacobians
            e, A, B = linearize_pose_landmark_constraint(
                x, l, edge.measurement)

            # (TODO) compute the terms
            b_i = np.transpose(e) @ edge.information @ A
            b_j = np.transpose(e) @ edge.information @ B
            H_ii = np.transpose(A) @ edge.information @ A
            H_ij = np.transpose(A) @ edge.information @ B
            H_ji = np.transpose(B) @ edge.information @ A
            H_jj = np.transpose(B) @ edge.information @ B

            # (TODO )add the terms to H matrix and b
            b[:, fromIdx:fromIdx + 3] += b_i
            b[:, toIdx:toIdx + 2] += b_j

            H[fromIdx:fromIdx + 3, fromIdx:fromIdx + 3] += H_ii
            H[fromIdx:fromIdx + 3, toIdx:toIdx + 2] += H_ij
            H[toIdx:toIdx + 2, fromIdx:fromIdx + 3] += H_ji
            H[toIdx:toIdx + 2, toIdx:toIdx + 2] += H_jj
            H += LM_lambda * np.eye(H.shape[0], H.shape[1])

    # solve system
    L = np.linalg.cholesky(H)
    # dx = np.linalg.solve(H, np.transpose(-b))

    return dx


def linearize_pose_pose_constraint(x1, x2, z):
    """Compute the error and the Jacobian for pose-pose constraint

    Parameters
    ----------
    x1 : 3x1 vector
         (x,y,theta) of the first robot pose
    x2 : 3x1 vector
         (x,y,theta) of the second robot pose
    z :  3x1 vector
         (x,y,theta) of the measurement

    Returns
    -------
    e  : 3x1
         error of the constraint
    A  : 3x3
         Jacobian wrt x1
    B  : 3x3
         Jacobian wrt x2
    """
    # compute the error e = (z^-1) * (x1^-1) * (x2)
    e = t2v(inv(v2t(z)) @ inv(v2t(x1)) @ v2t(x2)).reshape(3, 1)

    # compute the A and B
    # Get Ri and Rij
    R_i = v2t(x1)[0:2, 0:2]
    R_ij = v2t(z)[0:2, 0:2]
    t_i = x1[0:2].reshape(2, 1)
    t_j = x2[0:2].reshape(2, 1)

    # d (R_i)^T / d theta_i
    D_of_R = derivative_of_R_transpose_wrt_theta(x1[2])

    ii = -np.transpose(R_ij) @ np.transpose(R_i)
    ij = np.transpose(R_ij) @ D_of_R @ (t_j - t_i)
    row_1 = np.concatenate((ii, ij), axis=1)
    row_2 = np.array([0, 0, -1]).reshape(1, 3)
    A = np.concatenate((row_1, row_2), axis=0)

    row_1_B = np.concatenate(
        (np.transpose(R_ij) @ np.transpose(R_i), np.array([0, 0]).reshape(2, 1)), axis=1)
    B = np.concatenate((row_1_B, np.array([0, 0, 1]).reshape(1, 3)), axis=0)

    return e, A, B


def derivative_of_R_transpose_wrt_theta(theta):
    # d (R_i)^T / d theta_i
    c = np.cos(theta)
    s = np.sin(theta)

    T = np.array([[-s, c], [-c, -s,]])
    return T


def linearize_pose_landmark_constraint(x, l, z):
    """Compute the error and the Jacobian for pose-landmark constraint

    Parameters
    ----------
    x : 3x1 vector
        (x,y,theta) og the robot pose
    l : 2x1 vector
        (x,y) of the landmark
    z : 2x1 vector
        (x,y) of the measurement

    Returns
    -------
    e : 2x1 vector
        error for the constraint
    A : 2x3 Jacobian wrt x
    B : 2x2 Jacobian wrt l
    """

    zil_hat = compute_l_in_x_frame(x, l)
    e = zil_hat - z.reshape(2, 1)

    R_i = v2t(x)[0:2, 0:2]
    # d (R_i)^T / d theta_i
    D_of_R = derivative_of_R_transpose_wrt_theta(x[2])
    t = x[0:2]

    A = np.concatenate((-np.transpose(R_i), D_of_R @
                       (l-t).reshape(2, 1)), axis=1)
    B = np.transpose(R_i)

    return e, A, B
