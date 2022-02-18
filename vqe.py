#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:57:44 2021

@author: noumanbutt
"""

#import numpy as np
import time

from decompose_pauli import to_pauli_vec
import numpy as np
from A1plus import get_eigenvalues
from qiskit import Aer

from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.circuit.library import RealAmplitudes

from qiskit.opflow.primitive_ops import PauliOp

from qiskit.quantum_info.operators import Pauli




def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)


lmax=20
nmax=10
flag=0
M=16
gs=[0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.0,2.2,2.4,2.6]
n=0

E0_vqe=np.zeros((14))

n_qubit=np.log2(16)

for g in gs:
 print("g = ",g)
 H,evals=get_eigenvalues(g,M,lmax,nmax,flag)  
 ref_value=evals[0]
 print("Exact value ",ref_value)

 pauli_vec = to_pauli_vec(H)
 
 print("# of Pauli-Strings: ",len(pauli_vec))

 # of Is = log_2(M)
 
 H=PauliOp(Pauli(label='IIII'),0.0)


 for pauli_string in pauli_vec.keys():
    coefficient = pauli_vec[pauli_string]
    
    H += PauliOp(Pauli(label=pauli_string),coefficient)
    
 
 
 seed = (int) (10000*np.random.rand()) 
 iterations = 300
 algorithm_globals.random_seed = seed
 backend = Aer.get_backend('statevector_simulator')
 qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

 counts = []
 values = []

 ansatz =  TwoLocal(rotation_blocks='ry', entanglement_blocks='cz',reps=8)
 spsa = SPSA(maxiter=iterations)
 time0 = time.time()
 vqe = VQE(ansatz, optimizer=spsa, callback=store_intermediate_result,quantum_instance=qi)
 result = vqe.compute_minimum_eigenvalue(operator=H)
 E0_vqe[n] = result.eigenvalue.real
 #print(vqe._get_eigenstate())
 time1 = time.time()
 n += 1
 print("Time taken to run VQE: ", time1-time0)
 print(f'VQE on Aer qasm simulator (no noise): {result.eigenvalue.real:.5f}')
 print(f'Delta from reference energy value is {(result.eigenvalue.real - ref_value):.5f}')
 