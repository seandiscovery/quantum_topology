"""
An implementation of A. Kitaev's toric code (https://arxiv.org/pdf/quant-ph/9707021.pdf). 
Final project for CS269Q at Stanford University.
Authors: Richard Mu, Sean Mullane, and Chris Yeh 
(c) 2019 
"""
from typing import Tuple, List
import networkx as nx

def construct_toric_code(L: int) -> List[nx.Graph]:
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
	dual_graph = nx.generators.lattice.grid_2d_graph(L, L, periodic=True)
	
	# Step 2: Add physical qubits and stabilizer generators to lattice 
	# Add qubits to edges  

	# Construct Z-stabilizer generators 

	# Construct X-stabilizer generators   

	return [primal_graph, dual_graph] 

def toric_error_correction() -> List[nx.Graph]: 
	""" Given a toric code, applies an error model and attempts to correct errors. 
	"""
	pass 





