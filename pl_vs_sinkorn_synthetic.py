import argparse
import numpy as np
import cupy as cp
import torch
import ot
import ot.gpu
import os
import time

from scipy.spatial.distance import cdist
from matching import matching_cpu, matching_gpu

def cost_prep(n = 100, seed = 0):
    np.random.seed(seed)
    W = np.random.rand(n,n)
    return W

def rand_points(n = 100, seed = 0):
    np.random.seed(seed)
    a = np.random.rand(n,2)
    b = np.random.rand(n,2)
    cost = cdist(a, b, 'sqeuclidean')
    return a, b, cost

def computeDistanceMatrixGrid(n, dist):
    grid = [None]*(n*n)
    iter = 0
    for i in range(n):
        for j in range(n):
            grid[iter] = (i,j)
            iter += 1
    M = cdist(grid, grid, dist)
    return M

def get_sinkorn_reg(a, b, cost, target_loss, d = 1e-2):
    reg_rt = 1
    reg_lt = 1e-5
    reg_mid = (reg_lt+reg_rt)/2
    while(True):
        G = ot.sinkhorn(cp.array(a), cp.array(b), cp.array(cost), reg_mid, method='sinkhorn', numItermax=100000000)
        # cur_loss = ot.sinkhorn2(cp.array(a), cp.array(b), cp.array(cost), reg_mid, method='sinkhorn', numItermax=100000000)
        cur_loss = np.sum(G.get()*cost)
        d_loss = cur_loss - target_loss
        if(np.absolute(d_loss)<=d):
            break
        if(np.absolute(reg_rt-reg_mid)<=1e-6 or np.absolute(reg_lt-reg_mid)<=1e-6):
            print("choose boundary value")
            break
        if(d_loss > 0):
            reg_rt = reg_mid
            reg_mid = (reg_mid + reg_lt)/2
        else:
            reg_lt = reg_mid
            reg_mid = (reg_mid + reg_rt)/2
    
    return reg_mid

parser = argparse.ArgumentParser()
parser.add_argument('--nexp', type=int, default=30)
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--delta', type=float, default=0.1)
args = parser.parse_args()
print(args)

NUM_EXPERIMENTS = int(args.nexp)
n = int(args.n)
delta = args.delta

emd_time = []
emd_loss = []

sink_time = []
sink_loss = []

pl_time = []
pl_iter = []
pl_loss = []

sink_gpu_time = []
sink_gpu_loss = []

pl_gpu_torch_time = []
pl_gpu_torch_iter = []
pl_gpu_torch_loss = []

pl_gpu_time = []
pl_gpu_iter = []
pl_gpu_loss = []

for i in range(NUM_EXPERIMENTS):
    a, b, cost = rand_points(n, i)
    a = np.ones(n)/n
    b = np.ones(n)/n
    C = cost.max()
    
    #######test on cpu###########
    start = time.time()
    ot_loss_emd = ot.emd2(a, b, cost, processes=1, numItermax=100000000)
    end = time.time()
    emd_time.append(end-start)
    emd_loss.append(ot_loss_emd)

    start = time.time()
    Mb, yA, yB, ot_loss, iteration = matching_cpu(cost, C, delta)
    end = time.time()
    pl_time.append(end-start)
    pl_loss.append(ot_loss/n)
    pl_iter.append(iteration)

    reg = get_sinkorn_reg(a, b, cost, ot_loss/n, d = 1e-5)

    start = time.time()
    ot_loss_sink = ot.sinkhorn2(a, b, cost, reg, method='sinkhorn', numItermax=100000000)
    end = time.time()
    sink_time.append(end-start)
    sink_loss.append(ot_loss_sink)

    #######test on gpu###########
    W = cp.array(cost)

    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device2 = torch.device("cpu")
        # print(device)
        cost = torch.tensor(cost, device=device, requires_grad=False)
        C = torch.tensor([C], device=device, requires_grad=False)
        delta_tensor = torch.tensor([delta], device=device, requires_grad=False)
        cost_cpu = cost.to(device2)
        start = time.time()
        # Mb, yA, yB, ot_loss, iteration
        # Mb, yA, yB, ot_pyt_loss, iteration = matching_torch_(cost, C, delta, device=device)
        Mb, yA, yB, ot_pyt_loss, iteration = matching_gpu(cost, cost_cpu, C, delta_tensor, device=device)
        torch.cuda.synchronize()
        end = time.time()
        pl_gpu_torch_time.append(end-start)
        pl_gpu_torch_loss.append(ot_pyt_loss/n)
        pl_gpu_torch_iter.append(iteration)

    cost = cost_cpu.numpy()
    reg = get_sinkorn_reg(a, b, cost, ot_pyt_loss.numpy()/n, d = 1e-5)

    start = time.time()
    G = ot.gpu.sinkhorn(cp.array(a), cp.array(b), W, reg, method='sinkhorn', numItermax=100000000)
    ot_pal_loss_sinkhorn = np.sum(G*cost)
    end = time.time()
    sink_gpu_time.append(end-start)
    sink_gpu_loss.append(ot_pal_loss_sinkhorn)

print("random data")
print("problem size {}".format(n))
print("*********test on cpu************")    
print("emd standard took {}({}) seconds with loss {}({})".format(np.mean(emd_time), np.std(emd_time), np.mean(emd_loss), np.std(emd_loss)))
print("sinkhorn-cpu took {}({}) seconds with loss {}({})".format(np.mean(sink_time), np.std(sink_time), np.mean(sink_loss), np.std(sink_loss)))
print("push-relabel-cpu took {}({}) seconds with loss {}({}), iter {}({})".format(np.mean(pl_time), np.std(pl_time), np.mean(pl_loss), np.std(pl_loss), np.mean(pl_iter), np.std(pl_iter)))
print("*********test on gpu************")  
print("sinkhorn-gpu took {}({}) seconds with loss {}({})".format(np.mean(sink_gpu_time), np.std(sink_gpu_time), np.mean(sink_gpu_loss), np.std(sink_gpu_loss)))
print("push-relabel-gpu took {}({}) seconds with loss {}({}), iter {}({})".format(np.mean(pl_gpu_torch_time), np.std(pl_gpu_torch_time), np.mean(pl_gpu_torch_loss), np.std(pl_gpu_torch_loss), np.mean(pl_gpu_torch_iter), np.std(pl_gpu_torch_iter)))