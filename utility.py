import networkx as nx
import numpy as np


def supply_V(V, i, A, V_in):
    A[i, :] = np.zeros(len(V_in))
    A[i, i] = 1
    V_in[i] = V

def get_V(G, high_V, low_V):
    C = nx.to_numpy_array(G)
    N = nx.number_of_nodes(G)
    V_in = np.zeros(N)
    A = -C
    np.fill_diagonal(A, np.maximum(np.sum(C, axis=1), 1))
    supply_V(1, high_V, A, V_in)
    supply_V(0, low_V, A, V_in)
    V = np.linalg.solve(A, V_in)
    return V

def get_I(G, high_V, low_V):
    nodes_list = list(G.nodes())
    V = get_V(G, high_V, low_V)
    I = np.zeros(G.number_of_edges())
    for i, e in enumerate(G.edges()):
        u, v = e
        u = nodes_list.index(u)
        v = nodes_list.index(v)
        I[i] = V[u]-V[v]
    return I

def get_dI(G, I, edge, high_V, low_V ):
    nodes_list = list(G.nodes())
    G_copy = G.copy()
    G_copy.remove_edge(*edge)
    V_new = get_V(G_copy, high_V, low_V)
    dI = np.zeros(G.number_of_edges())
    for i, e in enumerate(G.edges()):
        if e != edge:
            u, v = e
            u = nodes_list.index(u)
            v = nodes_list.index(v)
            dI[i] = V_new[u]-V_new[v]-I[i]
        else:
            dI[i] = -I[i]
    return dI

def get_dV(G, V, edge, high_V, low_V):
    G_copy = G.copy()
    G_copy.remove_edges_from(edge)
    V_new = get_V(G_copy, high_V, low_V)
    return V_new - V

def get_node_dV(G, V, n, high_V, low_V):
    nodelist = list(G.nodes())
    G_copy = G.copy()
    connect_edges = G.edges(n)
    G_copy.remove_edges_from(connect_edges)
    C = nx.to_numpy_array(G_copy)
    N = nx.number_of_nodes(G_copy)
    V_in = np.zeros(N)
    A = -C
    np.fill_diagonal(A, np.maximum(np.sum(C, axis=1), 1))
    supply_V(1, high_V, A, V_in)
    supply_V(0, low_V, A, V_in)
    supply_V(0, nodelist.index(n), A, V_in)
    V_new = np.linalg.solve(A, V_in)
    return V_new-V


def para_up(e1, d):
    (x1a,y1a),(x1b,y1b) = e1
    x2a = x1a
    x2b = x1b
    y2a = y2b = y1a+d
    e2 = ((x2a,y2a),(x2b,y2b))
    return e2

def sym_xy(e1):
    (x1a,y1a),(x1b,y1b) = e1
    e2 = ((y1a,x1a),(y1b,x1b))
    return e2

def reflect(e1,L):
    l=L-1
    (x1a,y1a),(x1b,y1b) = e1
    e2 = ((l-y1a,l-x1a),(l-y1b,l-x1b))
    return e2

def pair(n, L):
    x, y = n
    l=L-1
    n2 = (l-x,l-y)
    return n2

def names_to_index(G, names):
    nodes_list = list(G.nodes())
    indexs = np.zeros(len(names))
    i = 0
    for n in names:
        indexs[i] = nodes_list.index(n)
        i += 1
    return indexs

def edges_to_index(G, edges):
    edge_index = np.zeros(len(edges))
    for i, e in enumerate(edges):
        u, v = e
        edge_index[i] = tuple(names_to_index([u,v]))
    return edge_index


def get_distance_distribution(G):
    nodes = G.nodes()
    length = dict(nx.all_pairs_shortest_path_length(G))
    ls = []
    for u in nodes:
        for v in nodes:
            if u!=v:
                ls.append(length[u][v])
    r, n_r = np.unique(np.array(ls), return_counts=True)
    return r, n_r

def get_dim(G,cutoff = 5):
    r, n_r = get_distance_distribution(G)
    #discard = int(len(r)/3)*2
    x, y = np.log(r[0:cutoff]), np.log(n_r[0:cutoff])
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m+1

def get_avg_dist(G, nodes):
    d =0
    n =len(nodes)
    pairs = 0
    for i in range(n-1):
        for j in range(i+1,n):
            d += nx.shortest_path_length(G, nodes[i], nodes[j])
            pairs +=1
    return d/pairs

def get_avg_degree(G, nodes):
    k =0
    n =len(nodes)
    for i in nodes:
        k += G.degree(i)
    return k/n

def get_avg_closeness(G, nodes):
    c =0
    n =len(nodes)
    for i in nodes:
        c += nx.closeness_centrality(G, i)
    return c/n

def get_avg_betweenness(G, nodes):
    c =0
    betweenness = nx.betweenness_centrality(G)
    n =len(nodes)
    for i in nodes:
        c += betweenness[i]
    return c/n

def get_num_shortest_path(G, high_V, low_V):
    n_p = 0
    for p in nx.all_shortest_paths(G,high_V,low_V):
        n_p+=1
    return n_p

def getNeighbors(G, edge):
    u,v = edge
    edges = []
    for n in G.neighbors(u):
        edges.append((n,u))
    for n in G.neighbors(v):
        edges.append((n,v))
    return edges

def get_negihbor_edges(G, edges):
    edgelist = []
    for e in edges:
        edgelist.extend(getNeighbors(G, e))
    edgelist = list(dict.fromkeys(edgelist))
    for e in edges:
        edgelist.remove(e)
    return edgelist