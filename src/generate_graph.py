#!/usr/bin/env python

import networkx as nx
import numpy as np

def generate_graph(network_type,L = 10,m = 4,N = 100, p=0.01, k=5, ring_links=[1], periodic=False):
    if network_type == 'square':
        G = nx.grid_graph((L,L), periodic= periodic)
    elif network_type == 'hex':
        G = nx.hexagonal_lattice_graph(2*m-1,2*m-1, with_positions =True)
        cx, cy = m-0.5, 2*m -(m%2) #center
        nodes_to_remove = [n for n in G.nodes if (abs(cx-n[0]) + abs(cy-n[1])) > 2*m]
        G.remove_nodes_from(nodes_to_remove)
    elif network_type == 'hex2':
        G = nx.hexagonal_lattice_graph(L//2,L, with_positions =True, periodic= periodic)
    elif network_type == 'tri':
        G = nx.triangular_lattice_graph(L-1,2*(L-1), with_positions =True)
    elif network_type == 'per_att':
        G = nx.empty_graph(create_using=nx.MultiGraph)
        for i in range(3):
            G.add_edges_from([(0,1),(1,2)])
        for t in range(3,N):
            G.add_node(t)
            degrees = np.array([d for n,d in G.degree()])
            degrees[-1] = 1
            for i in range(m):
                P = degrees/np.sum(degrees)
                #print(degrees)
                n = np.random.choice(np.arange(t+1),p=P)
                G.add_edge(t,n)
                degrees[n] += 1
    elif network_type == 'ER':
        G = nx.fast_gnp_random_graph(N, p)
        G.remove_nodes_from(list(nx.isolates(G)))
        G.remove_edges_from(nx.selfloop_edges(G))
    elif network_type == 'uni_con':
        ks = k*np.ones(N, dtype=np.int8)
        G = nx.configuration_model(ks)
        G.remove_edges_from(nx.selfloop_edges(G))
    elif network_type == 'ring':
        G = nx.circulant_graph(N, ring_links)
        for n in range(nx.number_of_nodes(G)):
            if p>np.random.rand(): G.add_edge(n,int((n+m)%N))
    elif network_type == 'cube':
        G = nx.grid_graph((L,L,L), periodic= periodic)
    return G