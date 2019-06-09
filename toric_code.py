"""
An implementation of A. Kitaev's toric code (https://arxiv.org/pdf/quant-ph/9707021.pdf).
Final project for CS269Q at Stanford University.
Authors: Richard Mu, Sean Mullane, and Chris Yeh
(c) 2019
"""
from typing import Tuple, List

import networkx as nx
import numpy as np
from pyquil import Program
from pyquil.gates import CNOT, H, MEASURE, X, Z
from pyquil.quil import address_qubits
from pyquil.quilatom import QubitPlaceholder
from pyquil.api import QVMConnection


# Typing
Node = Tuple[int]
Edge = Tuple[Node]

qvm = QVMConnection()


def sort_edge(edge: Edge, L: int, mod: bool = True) -> Edge:
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


def dual_edge_to_primal_edge(dual_edge: Edge, L: int) -> Edge:
    """Converts an edge from the dual graph to an edge in the primal graph."""
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
    :returns: Primal graph and dual graph, and distance graph (for the MWPM algorithm)
    """
    # Using the NetworkX package; allows us to generate an L x L lattice
    # with periodic boundary conditions. See:
    # https://networkx.github.io/documentation/stable/reference/generated/networkx.generators.lattice.grid_2d_graph.html
    primal_graph = nx.generators.lattice.grid_2d_graph(L, L, periodic=True)

    for edge in primal_graph.edges:
        # Add data qubits to edges
        primal_graph.edges[edge]['data_qubit'] = QubitPlaceholder()
    for node in primal_graph.nodes:
        # Add ancilla qubits to nodes
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

    return primal_graph, dual_graph, distance_graph


def operator_distance(G: nx.Graph, distance_G: nx.Graph, L: int, o1: Node,
                      o2: Node):
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


def mwpm(G: nx.Graph, distance_G: nx.Graph, L: int, errors: List[Node]) -> List[List[Edge]]:
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
    assert nx.matching.is_perfect_matching(mwpm_G, matching)
    correction_paths = [paths[edge] for edge in matching]
    return correction_paths


def apply_operation(paths: List[List[Edge]], G: nx.Graph, gate) -> Program:
    '''Returns a program that applies a gate to every qubit in each path.
    '''
    pq = Program()
    for path in paths:
        for edge in path:  # each `edge` is the name of a qubit
            qubit = G.edges[edge]['data_qubit']
            pq += gate(qubit)
    return pq

##### Syndrome Extraction #####

def get_number() -> int:
    """ Generate unique memory ids for the X and Z syndrome extraction functions.
    :returns: Random integer in range (0, 1e10); chance of collision = 1e(-20)
    """
    return np.random.randint(0, 2**30)

def X_syndrome_extraction(primal_qubits: List[QubitPlaceholder]) -> Program:
    """ Runs the syndrome extraction circuit for the X-stabilizers.
    Detects phase-flip errors on the lattice.
    :param primal_qubits: List of ancilla and data qubits for the extraction.
    We assume the following input format:
            qubits = [ancilla, north, west, east, south]
    where "ancilla" is the ancilla qubit on the vertex of the primal graph,
    and "north", ... "south" are the data qubits to the north, ... south of
    the ancilla qubit. Note that we assume the ancilla is initialized to ancilla = |0>.
    :returns: Pyquil Program representing the syndrome extraction process
    """
    pq = Program()
    ro_X = pq.declare('ro', 'BIT', 1) # Do we need to avoid namespace conflicts here?

    # Initialize the ancilla
    pq += H(primal_qubits[0])
    # Perform the circuit
    for i in range(1, len(primal_qubits)):
        pq += CNOT(primal_qubits[0], primal_qubits[i])
    # Measure in the X-basis
    pq += H(primal_qubits[0])
    pq += MEASURE(primal_qubits[0], ro_X[0])

    return pq

def Z_syndrome_extraction(dual_qubits: List[QubitPlaceholder]) -> Program:
    """ Runs the syndrome extraction circuit for the Z-stabilizers.
    Detects bit-flip errors on the lattice.
    :param dual_qubits: List of ancilla and data qubits for the extraction.
    Assumed to have an identical format to the "primal_qubits" parameter
    in "X_syndrome_extraction" above. Note that the ancilla qubits live on the
    nodes of the dual graph (plaquette faces of the primal graph). Also note that
    we assume the ancilla is initialized to ancilla = |0>.
    :returns: +/- 1 to indicate whether an error has been detected
    """
    pq = Program()
    ro_Z = pq.declare('ro', 'BIT', 1)

    # Perform the circuit
    for i in range(1, len(dual_qubits)):
        pq += CNOT(dual_qubits[i], dual_qubits[0])
    # Measure in the Z-basis
    pq += MEASURE(dual_qubits[0], ro_Z[0])
    return pq

def weighted_flip(p):
    """ Flips a weighted coin; heads (0) with probability
    1 - p, tails (1) with probability p.
    :param p: Probability of tails (1)
    :returns: 0 for tails/failure, 1 for heads/success
    """
    flip = np.random.random()
    if flip < p:
        return 1
    else:
        return 0

def nwes(node, L):
    """ Ensures that the edges from a node are returned in the order 
    north, west, east, south for syndrome extraction.  
    """
    r, c = node
    N = (((r-1)%L, c), node)
    W = ((r, (c-1)%L), node)
    E = (node, (r, (c+1)%L))
    S = (node, ((r+1)%L, c))
    return [N, W, E, S]

def syndrome_extraction(G: nx.Graph, L: int, pq: Program, op: str) -> List[Node]:
    '''
    Args
    - G: graph
    - L: int
    - pq: Program
    - op: str, one of ['X', 'Z']
    Returns
    - faulty_nodes: list of nodes, representing plaquette or vertex operators
        that measured a -1 eigenvalue
    '''
    assert op in ['X', 'Z']

    faulty_nodes = []
    for node in G.nodes:  # each node is a plaquette or vertex operator
        # neighbors = sorted(G.edges(node))  # qubits that the operator acts on
        neighbors = nwes(node, L)

        # Extract the necessary qubits
        qubits = [G.nodes[node]["ancilla_qubit"]]
        for edge in neighbors:
            qubits.append(G.edges[edge]["data_qubit"])

        if op == 'X':
            syndrome = pq + X_syndrome_extraction(qubits)
        else:
            syndrome = pq + Z_syndrome_extraction(qubits)
        syndrome = address_qubits(syndrome)
        result = qvm.run(syndrome)
        if result[0][0]:
            # Error detected
            faulty_nodes.append(node)

    return faulty_nodes

def simulate_error(primal: nx.Graph, dual: nx.Graph, p=None, phase_flips=None, bit_flips=None):
    if p is None:
        assert phase_flips is not None
        assert bit_flips is not None
    else:
        # Randomly choose which qubits will have bit/phase flip errors
        # Working under the independent noise model; since the toric code is a CSS code,
        # we can analyze bit and phase flip errors seperately

        assert phase_flips is None
        assert bit_flips is None

        phase_flips = set()
        for edge in primal.edges:
            if weighted_flip(p):
                phase_flips.add(edge)
        bit_flips = set()
        for edge in dual.edges:
            if weighted_flip(p):
                bit_flips.add(edge)

    primal_pq = Program()
    dual_pq = Program()

    # Apply the errors we selected above to the necessary qubits
    for p_edge in primal.edges:
        if p_edge in phase_flips:
            primal_pq += Z(primal.edges[p_edge]['data_qubit'])
    for d_edge in dual.edges:
        if d_edge in bit_flips:
            dual_pq += X(dual.edges[d_edge]['data_qubit'])

    return primal_pq, phase_flips, dual_pq, bit_flips


def measure_all_qubits(G, pq):
    num_qubits = len(G.edges)

    ro = pq.declare('ro', 'BIT', num_qubits)
    edge_list = list(G.edges)
    for i, edge in enumerate(edge_list):  # sort for determinism
        pq += MEASURE(G.edges[edge]['data_qubit'], ro[i])
    pq = address_qubits(pq)
    result = qvm.run(pq)[0]
    for edge, bit in zip(edge_list, result):
        G.edges[edge]['value'] = bit


def main():
    L = 3
    p = 0.05
    primal_G, dual_G, distance_G = construct_toric_code(L)

    # generate programs that initialize qubits to valid codeword
    empty_pq = Program()
    primal_faulty_nodes = syndrome_extraction(G=primal_G, L=L, pq=empty_pq, op='X')
    dual_faulty_nodes = syndrome_extraction(G=dual_G, L=L, pq=empty_pq, op='Z')
    print(primal_faulty_nodes)

    correction_paths = mwpm(primal_G, distance_G, L=L, errors=primal_faulty_nodes)
    primal_pq = apply_operation(paths=correction_paths, G=primal_G, gate=X)

    correction_paths = mwpm(dual_G, distance_G, L=L, errors=dual_faulty_nodes)
    dual_pq = apply_operation(paths=correction_paths, G=dual_G, gate=Z)

    measure_all_qubits(primal_G, primal_pq)
    measure_all_qubits(dual_G, dual_pq)

    ascii_print(primal_G, L)

    # apply noise to qubits
    primal_error_pq, phase_flips, dual_error_pq, bit_flips = simulate_error(primal_G, dual_G, p)
    primal_pq += primal_error_pq
    dual_pq += dual_error_pq

    # determining faulty nodes for error correction
    primal_faulty_nodes = syndrome_extraction(G=primal_G, pq=primal_pq, op='X')
    dual_faulty_nodes = syndrome_extraction(G=dual_G, pq=dual_pq, op='Z')

    correction_paths = mwpm(primal_G, distance_G, L=L, errors=primal_faulty_nodes)
    primal_pq += apply_operation(paths=correction_paths, G=primal_G, gate=X)

    correction_paths = mwpm(dual_G, distance_G, L=L, errors=dual_faulty_nodes)
    dual_pq += apply_operation(paths=correction_paths, G=dual_G, gate=Z)

    # add measurement operators to validate error correction
    measure_all_qubits(primal_G, primal_pq)
    measure_all_qubits(dual_G, dual_pq)

    ascii_print(primal_G, L)


def ascii_print(G: nx.Graph, L: int):
    '''
    Args
    - G: nx.Graph, where each edge has a 'value' field
    '''
    # does not show the wrap-around row
    x = np.zeros([2*L, 2*L], dtype=object)

    # all of the row qubits
    for r in range(L):
        for c in range(L):
            x[2*r, 2*c] = '+'
            x[2*r + 1, 2*c + 1] = '.'

            n1 = (r, c)
            n2 = (r, (c+1) % L)
            edge = (n1, n2)
            x[2*r, 2*c + 1] = str(G.edges[edge]['value'])

            n1 = (r, c)
            n2 = ((r+1) % L, c)
            edge = (n1, n2)
            x[2*r + 1, 2*c] = str(G.edges[edge]['value'])

    s = [' '.join(list(row)) for row in x]
    s = '\n'.join(s)
    print(s)

main()