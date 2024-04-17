#!/usr/bin/env python

from generate_graph import generate_graph
import networkx as nx
import numpy as np
from utility import *
import os


def default_input_nodes(G, network_type):
    N = nx.number_of_nodes(G)
    if network_type == 'square' or network_type == 'tri' or network_type == 'hex2':
        L = int(np.sqrt(N))
        input_nodes = [(0, L//2), (L//2, 0), (L-1, L//2), (L//2, L-1)]
    elif network_type == 'hex':
        input_nodes = [(0, 12), (7, 12), (0, 4), (7, 4)]
    elif network_type == 'ring':
        input_nodes = [N//6,2*N//6,4*N//6,5*N//6]
    else:
        input_nodes = np.random.choice(G.nodes(), 4)
    return input_nodes

def default_input_edges(G, network_type):
    N = nx.number_of_nodes(G)
    if network_type == 'square' or network_type == 'tri' or network_type == 'hex2':
        L = int(np.sqrt(N))
        input_edges = [((0, L//2-1),(0, L//2)), ((L//2-1, 0),(L//2, 0)), ((L-1, L//2-1),(L-1, L//2)), ((L//2-1, L-1),(L//2, L-1))]
    elif network_type == 'ring':
        input_edges = [(N//6-1,N//6),(2*N//6-1,2*N//6),(4*N//6-1,4*N//6),(5*N//6-1,5*N//6)]
    else:
        input_edges = np.random.choice(G.edges(), 4)
    return input_edges

def generate_training_data(G, network_type, high_V, low_V, filename, num_samples=-1, num_inputs=4, ran=0, noise=0.1, midpoint_list=None, num_choices = 1):
    N = nx.number_of_nodes(G)
    nodes_list = list(G.nodes())
    if midpoint_list is None:
        midpoint_list = default_input_nodes(G, network_type)
    midpoints = midpoint_list
    print(midpoints)
    for i in range(ran):
        nx.double_edge_swap(G)
    edges = list(G.edges)
    if num_samples == -1:
        num_samples = len(edges)
    V_original = get_V(G, high_V, low_V)

    #num_choices = 1
    with open(filename, 'w') as f:
        for i in range(num_samples):
            if i == num_samples//4:
                print('0.25 complete')
            # Randomly disconnect an edge
            #edge_to_remove = random.choice(edges)
            edge_to_remove = [edges[i]]
            #edge_to_remove = [edges[i] for i in np.random.choice(num_samples,num_choices)]
            # Calculate voltage changes
            dV = get_dV(G, V_original, edge_to_remove, high_V, low_V)
            # Prepare inputs: dV for each midpoint
            inputs = []
            for n in midpoints:
                node_index = nodes_list.index(n)
                inputs.append(dV[node_index])
            # Prepare output: index of disconnected edge
            output = []
            for i, edge in enumerate(edge_to_remove):

              output.append(edges.index(edge))
            # Write to file
            for _ in range(10):
                ran_in = np.array(inputs)*(1+(np.random.rand(num_inputs)-0.5)*2*noise)
                f.write(','.join(map(str, ran_in)) + ',' + str(output[0]) + '\n')

def setup_training_data(L=10, network_type = 'square', high_V = 0, low_V = -1, midpoint_list=None, name = str):

    G = generate_graph(network_type, L)
    nodes_list = list(G.nodes())

    #print(G)
    if not os.path.exists(f'training_data_{network_type}_{name}.txt'):
        generate_training_data(G, network_type, high_V, low_V, f'training_data_{network_type}_{name}.txt',
                            num_samples=-1, num_inputs=4, ran=0, midpoint_list=midpoint_list)
    #plot_inputs(G, network_type, high_V, low_V, midpoint_list)