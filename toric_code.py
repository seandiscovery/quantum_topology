"""
An implementation of A. Kitaev's toric code (https://arxiv.org/pdf/quant-ph/9707021.pdf).
Final project for CS269Q at Stanford University.
Authors: Richard Mu, Sean Mullane, and Chris Yeh
(c) 2019
"""
from typing import Tuple, List

import networkx as nx
import numpy as np
from pyquil.quilatom import QubitPlaceholder

def sort_edge(edge: Tuple, L: int, mod: bool = True) -> Tuple[Tuple]:
    '''Verifies that the given edge (n1, n2) is valid and orients the edge such
    that n2 is always east or south of n1.

    Args
    - edge: tuple of (n1, n2), where n1 and n2 each are tuples (r, c)
    - L: int, size of grid with periodic boundary conditions
    - mod: whether to return the edge with coordinates modulo L

    Returns
    - edge: same as edge, except n2 is always east or south of n1
    '''
    error = ValueError('Invalid edge')

    n1, n2 = edge
    if n1 == n2:
        raise error

    n1_r, n1_c = n1
    n2_r, n2_c = n2

    if any([n1_r < 0, n1_c < 0, n2_r < 0, n2_c < 0]):
        raise error

    if abs(n2_r - n1_r) > 1:
        if n1_c != n2_c:
            raise error

        if n1_r == 0:
            n1_r += L
        elif n2_r == 0:
            n2_r += L
        else:
            raise error
    if abs(n2_c - n1_c) > 1:
        if n1_r != n2_r:
            raise error

        if n1_c == 0:
            n1_c += L
        elif n2_c == 0:
            n2_c += L
        else:
            raise error

    if (abs(n2_r - n1_r) > 1) or (abs(n2_c - n1_c) > 1):
        raise error

    if (n1_r > n2_r) or (n1_c > n2_c):
        n1_r, n1_c, n2_r, n2_c = n2_r, n2_c, n1_r, n1_c

    if mod:
        n1 = (n1_r % L, n1_c % L)
        n2 = (n2_r % L, n2_c % L)
    else:
        n1 = (n1_r, n1_c)
        n2 = (n2_r, n2_c)
    return (n1, n2)


def dual_edge_to_primal_edge(dual_edge, L):
    (n1, n2) = sort_edge(dual_edge, L, mod=False)
    (n1_r, n1_c) = n1
    (n2_r, n2_c) = n2

    n1_r += 0.5
    n1_c += 0.5
    n2_r += 0.5
    n2_c += 0.5
    q_r = 0.5 * (n1_r + n2_r)
    q_c = 0.5 * (n1_c + n2_c)
    if n1_r == n2_r:
        n3_r = q_r - 0.5
        n3_c = q_c
        n4_r = q_r + 0.5
        n4_c = q_c
    elif n1_c == n2_c:
        n3_r = q_r
        n3_c = q_c - 0.5
        n4_r = q_r
        n4_c = q_c + 0.5
    else:
        raise ValueError('Invalid dual edge')
    return ((int(n3_r % L), int(n3_c % L)), (int(n4_r % L), int(n4_c % L)))


def construct_toric_code(L: int) -> Tuple[nx.Graph]:
    """ Constructs a toric code as a NetworkX graph structure.

    :param L: Number of physical qubits on one side of the square lattice
    :returns: Primal graph and dual graph in a List
    """
    # Step 1: generate an L x L grid, where L is the number of physical qubits
    # on one side of the grid

    # Using the NetworkX package; allows us to generate an L x L lattice
    # with periodic boundary conditions. See:
    # https://networkx.github.io/documentation/stable/reference/generated/networkx.generators.lattice.grid_2d_graph.html
    primal_graph = nx.generators.lattice.grid_2d_graph(L, L, periodic=True)
    for edge in primal_graph.edges:
        primal_graph.edges[edge]['data_qubit'] = QubitPlaceholder()
    for node in primal_graph.nodes:
        primal_graph.nodes[node]['ancilla_qubit'] = QubitPlaceholder()

    dual_graph = nx.generators.lattice.grid_2d_graph(L, L, periodic=True)
    for edge in dual_graph.edges:
        dual_graph.edges[edge]['data_qubit'] = primal_graph.edges[dual_edge_to_primal_edge(edge, L)]['data_qubit']
    for node in dual_graph.nodes:
        dual_graph.nodes[node]['ancilla_qubit'] = QubitPlaceholder()

    distance_graph = nx.Graph()
    for edge in primal_graph.edges:
        edge = sort_edge(edge, L)
        distance_graph.add_node(edge)
        distance_graph.nodes[edge]['data_qubit'] = primal_graph.edges[edge]['data_qubit']
    for n1 in distance_graph.nodes:
        for n2 in distance_graph.nodes:
            if n1 == n2: continue

            if (n1[0] == n2[0]) or (n1[0] == n2[1]) or (n1[1] == n2[0]) or (n1[1] == n2[1]):
                distance_graph.add_edge(n1, n2)
    # Step 2: Add physical qubits and stabilizer generators to lattice
    # Add qubits to edges

    # Construct Z-stabilizer generators

    # Construct X-stabilizer generators

    return primal_graph, dual_graph, distance_graph


def toric_error_correction() -> List[nx.Graph]:
	""" Given a toric code, applies an error model and attempts to correct errors.
	"""
	pass


def operator_distance(G: nx.Graph, distance_G: nx.Graph, L: int, o1: Tuple,
                      o2: Tuple):
    '''
    Args
    - G: nx.Graph
    - distance_G: nx.Graph
    - o1/o2: tuple (r, c), vertex in graph G representing an operator

    Returns
    - shortest_pathlen: int, number of qubits between the 2 operators
    - shortest_path: list of edges (qubits)
    '''
    edges1 = G.edges(o1)
    edges2 = G.edges(o2)
    assert len(edges1) == 4, len(edges1)
    assert len(edges2) == 4, len(edges2)

    shortest_pathlen = np.inf
    shortest_path = None

    for e1 in edges1:
        e1 = sort_edge(e1, L)
        for e2 in edges2:
            e2 = sort_edge(e2, L)

            # path includes source and target
            # pathlen does not count source
            path = nx.shortest_path(distance_G, source=e1, target=e2)
            nqubits = len(path)
            if nqubits < shortest_pathlen:
                shortest_pathlen = nqubits
                shortest_path = path

    return shortest_pathlen, shortest_path


def mwpm(G, distance_G, L, errors):
    '''
    Args
    - G: nx.Graph, edges are qubits
    - distance_G: nx.Graph
    - errors: list of vertices (operators) in G where errors were observed

    Returns
    - correction_paths: list of paths, each path is a list of edges in G
    '''
    mwpm_G = nx.Graph()
    mwpm_G.add_nodes_from(errors)

    paths = {}
    for o1 in errors:
        for o2 in errors:
            if o1 == o2: continue

            nqubits, path = operator_distance(G, distance_G, L, o1, o2)
            mwpm_G.add_edge(o1, o2, weight=1/nqubits)
            paths[(o1, o2)] = path
            paths[(o2, o1)] = path[::-1]

    matching = nx.max_weight_matching(mwpm_G, maxcardinality=True)
    assert nx.is_perfect_matching(mwpm_G, matching)
    correction_paths = [paths[edge] for edge in matching]
    return correction_paths



