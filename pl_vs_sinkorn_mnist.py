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
from matching import matching_cpu, matching_gpu

def mnist_data_prep(n=1000, seed = 0, eps=1):
    ############ Creating pure and contaminated mnist dataset ############

    # KAIYI PATH : OT-profile/outlier_detection_robot_vs_gtot
    if os.path.exists('./mnist.npy'):
        mnist = np.load('./mnist.npy') # 60k x 28 x 28
        mnist_labels = np.load('./mnist_labels.npy').ravel().astype(int) # 60k x 1 # originally UINT8
    else:
        (mnist, mnist_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        np.save('mnist.npy', mnist)
        np.save('mnist_labels.npy', mnist_labels)

    np.random.seed(seed)
    k = 1 - eps # rate of inlier (eps : outlier)

    # print(mnist_labels[:20])

    total = np.arange(len(mnist_labels))
    indx_a = total[mnist_labels < 5]
    indx_b = total[mnist_labels > 4]

    indx_a = np.random.permutation(indx_a)[:n]
    indx_b = np.random.permutation(indx_b)[:n]
    # indx_a = np.random.permutation(total)[:n]
    # indx_b = np.random.permutation(total)[:n]

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

def get_sinkorn_reg(a, b, cost, target_loss, d = 1e-2):
    reg_rt = 1
    reg_lt = 1e-5
    reg_mid = (reg_lt+reg_rt)/2
    while(True):
        G = ot.gpu.sinkhorn(cp.array(a), cp.array(b), cp.array(cost), reg_mid, method='sinkhorn', numItermax=100000000)
        cur_loss = np.sum(G*cost)
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
sink_reg = 1

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
    X, Y = mnist_data_prep(n, seed = i, eps=1)
    a = np.ones(n)/n
    b = np.ones(n)/n
    cost = c_dist(X, Y)
    C = cost.max()
    
    # #######test on cpu###########
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
        cost = torch.tensor(cost, device=device, requires_grad=False)
        C = torch.tensor([C], device=device, requires_grad=False)
        delta_tensor = torch.tensor([delta], device=device, requires_grad=False)
        cost_cpu = cost.to(device2)
        start = time.time()
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

print("mnist data a04 b59")
print("problem size {}".format(n))
print("*********test on cpu************")    
print("emd standard took {}({}) seconds with loss {}({})".format(np.mean(emd_time), np.std(emd_time), np.mean(emd_loss), np.std(emd_loss)))
print("sinkhorn-cpu took {}({}) seconds with loss {}({})".format(np.mean(sink_time), np.std(sink_time), np.mean(sink_loss), np.std(sink_loss)))
print("push-relabel-cpu took {}({}) seconds with loss {}({}), iter {}({})".format(np.mean(pl_time), np.std(pl_time), np.mean(pl_loss), np.std(pl_loss), np.mean(pl_iter), np.std(pl_iter)))
print("*********test on gpu************")  
print("sinkhorn-gpu took {}({}) seconds with loss {}({})".format(np.mean(sink_gpu_time), np.std(sink_gpu_time), np.mean(sink_gpu_loss), np.std(sink_gpu_loss)))
print("push-relabel-gpu took {}({}) seconds with loss {}({}), iter {}({})".format(np.mean(pl_gpu_torch_time), np.std(pl_gpu_torch_time), np.mean(pl_gpu_torch_loss), np.std(pl_gpu_torch_loss), np.mean(pl_gpu_torch_iter), np.std(pl_gpu_torch_iter)))
