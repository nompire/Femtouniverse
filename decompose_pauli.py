#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sp
import itertools, functools

op_I = sp.eye(2)
op_Z = sp.dia_matrix([[1,0],[0,-1]])
op_X = sp.dia_matrix([[0,1],[1,0]])
op_Y = -1j * op_Z @ op_X
pauli_ops = { "I" : op_I, "Z" : op_Z, "X": op_X, "Y" : op_Y }

# kronecker (tensor) product of two sparse matrices
# returns a sparse matrix in "dictionary of keys" format (to eliminate zeros)
def sp_kron_dok(mat_A, mat_B): return sp.kron(mat_A, mat_B, format = "dok")

def to_pauli_vec(mat):
    pauli_vec = {} # the dictionary we are saving

    mat_vec = np.array(mat).ravel()
    num_qubits = int(np.log2(np.sqrt(mat_vec.size)))

    for pauli_string in itertools.product(pauli_ops.keys(), repeat = num_qubits):
        # construct this pauli string as a matrix
        ops = [ pauli_ops[tag] for tag in pauli_string ]
        op = functools.reduce(sp_kron_dok, ops)

        # compute an inner product, same as tr(A @ B) but faster
        op_vec = op.reshape((1,4**num_qubits))
        coefficient = ( op_vec * mat_vec ).sum() / 2**num_qubits
        if coefficient != 0:
            pauli_vec["".join(pauli_string)] = coefficient

    return pauli_vec

#mat = [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]] # for example...






