#!/usr/bin/env python

import sys
import os
sys.path.append(os.getcwd()) 
import generate_graph
import generate_data
import utility
import DNN_training

network_type = "square"
name = "L31"
L = 31
N = 240
p = 1
m = 16

high_V = 0
#low_V = N//2
low_V = -1

#G = generate_graph.generate_graph(network_type, N=N, p=p, m=m)
G = generate_graph.generate_graph(network_type, L=L)

generate_data.generate_training_data(G, network_type, high_V, low_V, f'training_data_{network_type}_{name}.txt',num_samples=-1, num_inputs=4)
DNN_training.run_experiment(network_type=network_type, name=name, exp_num = '0', epochs=20000, patience=500)