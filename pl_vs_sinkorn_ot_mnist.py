"""
This is a script that conducts the experiments of optimal transport algorithm shown in our paper. 
CPU and GPU based implementation are compared with corresponding sinkorn methods with MNIST data
"""

import argparse
import numpy as np
import cupy as cp
import torch
import tensorflow as tf
import ot
import ot.gpu
import os
import time

from scipy.spatial.distance import cdist
from transport import transport_tensor

def mnist_data_prep(n=1000, seed = 0, eps=1):
    """
    This function creates MNIST experiment dataset for a bipartite matching algorithm.
    This function returns two arrays (A and B) of randomly selected MNIST images. 
    Array A contains 0-4 MNIST digits, array B contains 5-9 MNIST digits.
    """
    if os.path.exists('./mnist.npy'):
        mnist = np.load('./mnist.npy') # 60k x 28 x 28
        mnist_labels = np.load('./mnist_labels.npy').ravel().astype(int) # 60k x 1 # originally UINT8
    else:
        (mnist, mnist_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        np.save('mnist.npy', mnist)
        np.save('mnist_labels.npy', mnist_labels)

    np.random.seed(seed)
    k = 1 - eps # rate of inlier (eps : outlier)

    total = np.arange(len(mnist_labels))
    indx_a = total[mnist_labels < 5]
    indx_b = total[mnist_labels > 4]

    indx_a = np.random.permutation(indx_a)[:n]
    indx_b = np.random.permutation(indx_b)[:n]

    a  = mnist[indx_a, :, :]
    b  = mnist[indx_b, :, :]

    # im2double
    a = a/255.0
    b = b/255.0
    a = a.reshape(-1, 784)
    b = b.reshape(-1, 784)
    a = a / a.sum(axis=1, keepdims=1)
    b = b / b.sum(axis=1, keepdims=1)

    return a, b

def c_dist(b, a):
    return cdist(a, b, metric='minkowski', p=1)

def get_two_mnist_digits(a_num, b_num, seed = 0):
    if os.path.exists('mnist.npy'):
        mnist = np.load('mnist.npy') # 60k x 28 x 28
        mnist_labels = np.load('mnist_labels.npy').ravel().astype(int) # 60k x 1 # originally UINT8

    np.random.seed(seed)
    a_indx = np.where(mnist_labels == a_num)
    b_indx = np.where(mnist_labels == b_num)
    a = mnist[np.random.choice(a_indx[0], size=1),:,:]
    b = mnist[np.random.choice(b_indx[0], size=1),:,:]

    a = a/255
    b = b/255
    a[np.where(a == 0)] = 0.000001;
    b[np.where(b == 0)] = 0.000001;
    a = a.reshape(-1, 784)
    b = b.reshape(-1, 784)
    a = np.squeeze(a/np.sum(a))
    b = np.squeeze(b/np.sum(b))

    return a, b

def computeDistanceMatrixGrid(n):
    grid = [None]*(n*n)
    iter = 0
    for i in range(n):
        for j in range(n):
            grid[iter] = (i,j)
            iter += 1
    M = cdist(grid, grid, metric='minkowski', p=1)
    return M

def get_sinkorn_reg(a, b, cost, target_loss, d = 1e-6):
    """
    This function selects regularization parameter for sinkhorn algorithm using binary searching.
    """
    reg_rt = 1
    reg_lt = 1e-5
    reg_mid = (reg_lt+reg_rt)/2
    a_cupy = cp.array(a)
    b_cupy = cp.array(b)
    cost_cupy = cp.array(cost)
    while(True):
        start = time.time()
        G = ot.gpu.sinkhorn(a_cupy, b_cupy, cost_cupy, reg_mid, method='sinkhorn', numItermax=100000000, to_numpy=False)
        end = time.time()
        cur_loss = cp.sum(G*cost_cupy)
        # G = ot.sinkhorn(a, b, cost, reg_mid, method='sinkhorn', numItermax=100000000)
        # cur_loss = np.sum(G*cost)
        d_loss = cur_loss.get() - target_loss
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
        print("current iteration searching reg: d_time={}s | d_loss={} | new_reg={}".format(end-start, d_loss, reg_mid))
    print("search done!!!")
    return reg_mid

parser = argparse.ArgumentParser()
parser.add_argument('--nexp', type=int, default=1)
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--delta', type=float, default=0.5)
args = parser.parse_args()
print(args)

NUM_EXPERIMENTS = int(args.nexp)
n = int(args.n)
delta = args.delta
sink_reg = 1

emd_time = []
emd_loss = []

sink_time = []
sink_loss = []

pl_time = []
pl_iter = []
pl_loss = []
pl_zero_slack_length = []

sink_gpu_time = []
sink_gpu_loss = []

pl_gpu_torch_time = []
pl_gpu_torch_iter = []
pl_gpu_torch_loss = []
pl_gpu_zero_slack_length = []

# cost = computeDistanceMatrixGrid(28)
# C = cost.max()
# cost /= C

for i in range(NUM_EXPERIMENTS):
    X, Y = mnist_data_prep(n, seed = i, eps=1)
    DA = np.ones(n)/n
    SB = np.ones(n)/n
    cost = c_dist(X, Y)
    C = cost.max()
    cost /= C
    # DA, SB = get_two_mnist_digits(2,8,i)
    # m = 28
    
    # #######test on cpu###########
    start = time.time()
    ot_loss_emd = ot.emd2(DA, SB, cost, processes=1, numItermax=1e9)
    end = time.time()
    emd_time.append(end-start)
    emd_loss.append(ot_loss_emd)

    with torch.no_grad():
        device = torch.device("cpu")
        cost_tensor = torch.tensor(cost, device=device, requires_grad=False)
        delta_tensor = torch.tensor([delta], device=device, requires_grad=False)
        start = time.time()
        F, yA, yB, total_cost, iteration, zero_slack_length = transport_tensor(DA, SB, cost_tensor, delta_tensor, device=device)
        end = time.time()
        torch.cuda.synchronize()
        pl_time.append(end-start)
        pl_loss.append(total_cost.cpu().numpy())
        pl_iter.append(iteration)
        pl_zero_slack_length.append(np.mean(zero_slack_length))

    reg = get_sinkorn_reg(DA, SB, cost, total_cost.cpu().numpy(), d = 1e-5)

    start = time.time()
    ot_loss_sink = ot.sinkhorn2(DA, SB, cost, reg, method='sinkhorn', numItermax=100000000)
    end = time.time()
    sink_time.append(end-start)
    sink_loss.append(ot_loss_sink)

    #######test on gpu###########
    W = cp.array(cost)
    
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cost_tensor = torch.tensor(cost, device=device, requires_grad=False)
        delta_tensor = torch.tensor([delta], device=device, requires_grad=False)
        start = time.time()
        F, yA, yB, total_cost, iteration, zero_slack_length = transport_tensor(DA, SB, cost_tensor, delta_tensor, device=device)
        end = time.time()
        torch.cuda.synchronize()
        pl_gpu_torch_time.append(end-start)
        pl_gpu_torch_loss.append(total_cost.cpu().numpy())
        pl_gpu_torch_iter.append(iteration)
        pl_gpu_zero_slack_length.append(np.mean(zero_slack_length))

    # reg = get_sinkorn_reg(DA, SB, cost, total_cost.cpu().numpy(), d = 1e-5)

    DA_cupy = cp.array(DA)
    SB_cupy = cp.array(SB)
    start = time.time()
    G = ot.gpu.sinkhorn(DA_cupy, SB_cupy, W, reg, method='sinkhorn', numItermax=100000000, to_numpy=False)
    ot_pal_loss_sinkhorn = cp.sum(G*W)
    end = time.time()
    sink_gpu_time.append(end-start)
    sink_gpu_loss.append(ot_pal_loss_sinkhorn.get())

print("mnist data a04 b59")
print("*********test on cpu************")    
print("emd standard took {}({}) seconds with loss {}({})".format(np.mean(emd_time), np.std(emd_time), np.mean(emd_loss), np.std(emd_loss)))
print("sinkhorn-cpu took {}({}) seconds with loss {}({})".format(np.mean(sink_time), np.std(sink_time), np.mean(sink_loss), np.std(sink_loss)))
print("push-relabel-cpu took {}({}) seconds with loss {}({}), iter {}({})".format(np.mean(pl_time), np.std(pl_time), np.mean(pl_loss), np.std(pl_loss), np.mean(pl_iter), np.std(pl_iter)))
print("pl_cpu_zero_slack_length = {}".format(np.mean(pl_zero_slack_length)))
print("*********test on gpu************")  
print("sinkhorn-gpu took {}({}) seconds with loss {}({})".format(np.mean(sink_gpu_time), np.std(sink_gpu_time), np.mean(sink_gpu_loss), np.std(sink_gpu_loss)))
print("push-relabel-gpu took {}({}) seconds with loss {}({}), iter {}({})".format(np.mean(pl_gpu_torch_time), np.std(pl_gpu_torch_time), np.mean(pl_gpu_torch_loss), np.std(pl_gpu_torch_loss), np.mean(pl_gpu_torch_iter), np.std(pl_gpu_torch_iter)))
print("pl_gpu_zero_slack_length = {}".format(np.mean(pl_gpu_zero_slack_length)))

results = np.concatenate((emd_time,emd_loss,sink_time,sink_loss,pl_time,pl_loss,pl_iter,pl_zero_slack_length,sink_gpu_time,sink_gpu_loss,pl_gpu_torch_time,pl_gpu_torch_loss,pl_gpu_torch_iter,pl_gpu_zero_slack_length), axis=-1)
np.save('mnist_results_n_{}_nexp_{}_delta_{}.npy'.format(n, NUM_EXPERIMENTS, delta), results)