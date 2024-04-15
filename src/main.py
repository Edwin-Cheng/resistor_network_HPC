#!/usr/bin/env python

import generate_graph
import generate_data
import utility
import DNN_training

network_type = "square"
L = 31

DNN_training.run_experiment(L=L, network_type=network_type, name="square_L31", exp_num='0')