#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:18:56 2022

@author: noumanbutt
"""
#import numpy as np
import time
import matplotlib.pyplot as plt
from decompose_pauli import to_pauli_vec
import numpy as np
from A1plus import get_eigenvalues
from qiskit import Aer

from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal,EfficientSU2
#from qiskit.circuit.library import RealAmplitudes

from qiskit.opflow.primitive_ops import PauliOp

from qiskit.quantum_info.operators import Pauli



from qiskit.ignis.mitigation.measurement import CompleteMeasFitter


#import os
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeLima

# device backend -- can be changed easily by calling another device constructor
device_backend = FakeLima()


noise_model = None
device = QasmSimulator.from_backend(device_backend)
coupling_map = device.configuration().coupling_map
noise_model = NoiseModel.from_backend(device)
basis_gates = noise_model.basis_gates

print(noise_model)
print()



def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)

counts = []
values = []

lmax=20
nmax=10
flag=0
M=8
gs=[2.6]#,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.0,2.2,2.4,2.6]
n=0

E0_vqe=np.zeros((14))



for g in gs:
 print("g = ",g)
 H,evals=get_eigenvalues(g,M,lmax,nmax,flag)  
 ref_value=evals[0]
 print("Exact value ",ref_value)

 pauli_vec = to_pauli_vec(H)
 
 print("# of Pauli-Strings: ",len(pauli_vec))

 # of Is = log_2(M)
 
 H=PauliOp(Pauli(label='III'),0.0)


 for pauli_string in pauli_vec.keys():
    coefficient = pauli_vec[pauli_string]
    
    H += PauliOp(Pauli(label=pauli_string),coefficient)
    
 
 
 seed = (int) (10000*np.random.rand()) 
 iterations = 300
 algorithm_globals.random_seed = seed
 backend = Aer.get_backend('aer_simulator')
 
 
 # Backend and quantum instance for plain vqe -- 
 #backend = Aer.get_backend('statevector_simulator')
 #qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
 
 # Backend and quantum instance for vqe with noise
 #backend = Aer.get_backend('aer_simulator')
 # qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed,coupling_map=coupling_map,noise_model=noise_model)
 
 
 # Backend and quantum instance for vqe with noise and error mitiagation
 backend = Aer.get_backend('aer_simulator')
 qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed,
                         coupling_map=coupling_map, noise_model=noise_model,
                         measurement_error_mitigation_cls=CompleteMeasFitter,
                         cals_matrix_refresh_period=30)
 
 
 
 ansatz =  TwoLocal(rotation_blocks='ry', entanglement_blocks='cz',reps=2)
 # hardware efficient anstaz
 ansatz = EfficientSU2(3, su2_gates=['ry', 'x'], entanglement='full', reps=2)
 
 spsa = SPSA(maxiter=iterations,learning_rate=0.01,perturbation=0.2)
 time0 = time.time()
 vqe = VQE(ansatz, optimizer=spsa, callback=store_intermediate_result,quantum_instance=qi)
 result = vqe.compute_minimum_eigenvalue(operator=H)
 E0_vqe[n] = result.eigenvalue.real
 
 time1 = time.time()
 n += 1
 print("Time taken to run VQE: ", time1-time0)
 print(f'VQE on Aer qasm simulator (with noise + error mitigation): {result.eigenvalue.real:.5f}')
 print(f'Delta from reference energy value is {(result.eigenvalue.real - ref_value):.5f}')

plt.plot(counts,values)
plt.xlabel("Eval Count")
plt.ylabel("Energy")

#plt.title("VQE Convergence without noise") # Plain VQE
#plt.title(" VQE Convergence with  noise") # VQE with noise
plt.title("VQE Convergence with  noise and errror mitigation") # VQE with noise and error mitigation

plt.show() 