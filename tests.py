from toric_code import *


def test_dual_edge_to_primal_edge():
    L = 4
    expected_dual_to_primal = {
        ((0,0),(3,0)): ((0,0),(0,1)),
        ((0,0),(0,3)): ((0,0),(1,0)),
        ((1,1),(2,1)): ((2,1),(2,2)),
        ((1,1),(1,2)): ((1,2),(2,2)),
        ((0,2),(0,3)): ((0,3),(1,3))
    }
    for dual_edge, expected_primal_edge in expected_dual_to_primal.items():
        primal_edge = dual_edge_to_primal_edge(dual_edge, L)
        print(f'dual: {dual_edge}, expected: {expected_primal_edge}, got: {primal_edge}')
        assert primal_edge == expected_primal_edge

        reversed_dual_edge = (dual_edge[1], dual_edge[0])
        primal_edge = dual_edge_to_primal_edge(reversed_dual_edge, L)
        print(f'dual: {reversed_dual_edge}, expected: {expected_primal_edge}, got: {primal_edge}')
        assert primal_edge == expected_primal_edge


def test_operator_distance():
    '''Tests Figures 2.10 and 2.13 from
    Browne, "Topological Codes and Computatation"
    '''
    L = 5
    primal_G, dual_G, distance_G = construct_toric_code(L)

    # Figure 2.10, upper-left
    nqubits, path = operator_distance(G=primal_G, distance_G=distance_G, L=L,
                                      o1=(2, 2), o2=(2, 3))
    assert nqubits == 1
    assert len(path) == 1
    assert path[0] == ((2, 2), (2, 3))

    # Figure 2.10, upper-right
    nqubits, path = operator_distance(G=primal_G, distance_G=distance_G, L=L,
                                      o1=(2, 2), o2=(2, 4))
    assert nqubits == 2
    assert len(path) == 2
    assert path[0] == ((2, 2), (2, 3))
    assert path[1] == ((2, 3), (2, 4))

    # Figure 2.13d - top/bottom
    nqubits, path = operator_distance(G=primal_G, distance_G=distance_G, L=L,
                                      o1=(1, 2), o2=(4, 2))
    assert nqubits == 2
    assert len(path) == 2
    assert ((0, 2), (1, 2)) in path
    assert ((4, 2), (0, 2)) in path

    # Figure 2.13d - bottom/top
    nqubits, path = operator_distance(G=primal_G, distance_G=distance_G, L=L,
                                      o1=(4, 2), o2=(1, 2))
    assert nqubits == 2
    assert len(path) == 2
    assert ((0, 2), (1, 2)) in path
    assert ((4, 2), (0, 2)) in path

    # Figure 2.13d - left/right
    nqubits, path = operator_distance(G=primal_G, distance_G=distance_G, L=L,
                                      o1=(2, 1), o2=(2, 4))
    assert nqubits == 2
    assert len(path) == 2
    assert ((2, 4), (2, 0)) in path
    assert ((2, 0), (2, 1)) in path

    # Figure 2.10, upper-left, shift Z's down-right half for X error on dual
    nqubits, path = operator_distance(G=dual_G, distance_G=distance_G, L=L,
                                      o1=(2, 2), o2=(2, 3))
    assert nqubits == 1
    assert len(path) == 1
    assert path[0] == ((2, 2), (2, 3))


def test_mwpm():
    '''Tests Figures 2.10 and 2.13 from
    Browne, "Topological Codes and Computatation"
    '''
    L = 5
    primal_G, dual_G, distance_G = construct_toric_code(L)

    # Figure 2.10, upper-left
    errors = [(2,2), (2,3)]
    correction_paths = mwpm(primal_G, distance_G, L, errors)
    assert len(correction_paths) == 1
    for path in correction_paths:
        for edge in path:
            assert edge in primal_G.edges
    correction_paths = [tuple(path) for path in correction_paths]
    assert (((2,2), (2,3)),) in correction_paths

    # Figure 2.13d
    errors = [(1, 2), (2, 1), (2, 4), (4, 2)]
    correction_paths = mwpm(primal_G, distance_G, L, errors)
    assert len(correction_paths) == 2
    for path in correction_paths:
        for edge in path:
            assert edge in primal_G.edges
    correction_paths = [tuple(path) for path in correction_paths]
    assert (((0, 2), (1, 2)), ((4, 2), (0, 2))) in correction_paths
    assert (((2, 4), (2, 0)), ((2, 0), (2, 1))) in correction_paths

    # Figure 2.13d, shift Z's down-right half for X error on dual
    errors = [(1, 2), (2, 1), (2, 4), (4, 2)]
    correction_paths = mwpm(dual_G, distance_G, L, errors)
    assert len(correction_paths) == 2
    for path in correction_paths:
        for edge in path:
            assert edge in primal_G.edges
    correction_paths = [tuple(path) for path in correction_paths]
    assert (((0, 2), (1, 2)), ((4, 2), (0, 2))) in correction_paths
    assert (((2, 4), (2, 0)), ((2, 0), (2, 1))) in correction_paths


test_dual_edge_to_primal_edge()
test_operator_distance()
test_mwpm()
