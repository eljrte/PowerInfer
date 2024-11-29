#!/usr/bin/env python
# coding=utf-8
import argparse
from cvxopt.glpk import ilp
import numpy as np
from cvxopt import matrix
import torch
import pickle

def solve_gpu_split(
    activation_path: str,
    neuron: int,
    capacity: int,
    layer: int,
    batch: int,
    threshold: int,
):
    # Processing activation data 
    #这是按batch聚簇的 
    values = []
    for i in range(layer):
        # Load and sort activation data for each layer
        freq = torch.load(f"{activation_path}/activation_{i}.pt")
        freq, _ = torch.sort(freq, descending=True)
        freq = freq * -1.0
        freq = freq.view(-1, batch)
        freq = freq.sum(dim=1)
        freq = freq.tolist()
        values += freq

    # Padding zero values for additional constraints
    for i in range(layer):
        values += [0.0]
    c = np.array(values, dtype=float)
    c = matrix(c)

    # Setting capacity and neuron count per batch                   #不懂为什么要 除以一个batch
    CAP = capacity
    CAP = int(CAP / batch)
    neuron = int(neuron / batch)
    coeff = []
    h = []

    # Constraint 1: Total neuron activation constraint                这是从纯数量的角度来看的
    lst = []
    for i in range(neuron * layer):
        lst.append(1)
    for i in range(layer):
        lst.append(0)
    coeff.append(lst)
    h.append(CAP)

    # Constraint 2: Threshold constraint for GPU split per layer
    #确保第 i层神经元分配的总激活值满足最低阈值。   原文第7个不等式
    for i in range(layer):
        lst = [0] * (neuron * layer + layer)
        for j in range(neuron):
            lst[i * neuron + j] = -1
        lst[neuron * layer + i] = int(threshold / batch)
        coeff.append(lst)
        h.append(0)

    # Constraint 3: Upper bound on neuron activations
    #原文第8个不等式
    for i in range(layer):
        lst = [0] * (neuron * layer + layer)
        for j in range(neuron):
            lst[i * neuron + j] = 1
        lst[neuron * layer + i] = -1000000  # Arbitrary large negative number as an upper bound
        coeff.append(lst)
        h.append(0)

    # Convert lists to matrix format for ILP solver
    coeff = np.array(coeff, dtype=float)
    G = matrix(coeff)
    h = np.array(h, dtype=float)
    h = matrix(h)

    # Define the set of integer and binary variables
    I = set(range(neuron * layer + layer))
    B = set()

    # Solving the ILP problem
    (status, x) = ilp(c, G, h, None, None, B, I, options={'tm_lim' : 30000}) # with 30s timeout
    print(f"ILP Status: {status}")
    ans = list(x)
    print(f"Total Activation Units: {sum(ans)}")

    #估计
    aligned_lst = []
    for i in range(layer):
        aligned_lst.append(sum(ans[i * neuron:i * neuron + neuron] * batch))

    return aligned_lst


