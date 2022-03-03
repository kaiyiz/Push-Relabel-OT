import argparse
import numpy as np
import cupy as cp
import torch
import tensorflow as tf
from scipy.spatial.distance import cdist
import ot
import ot.gpu
import os
import time

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

def matching_cpu(W, C, delta):
    n = W.shape[0]
    S = (4*W//(delta)).astype(int) 
    cost = (4*W//(delta)).astype(int)
    yB = np.ones(n, dtype=int)
    yA = np.zeros(n, dtype=int)
    Mb = np.ones(n, dtype=int) * -1
    Ma = np.ones(n, dtype=int) * -1
    f = n
    iteration = 0

    while f > n*delta/C:
        #need to test if needs to search Mb in gpu
        #T(ind_b_free transfer to cpu)-(T(search Mb in cpu)-T(search Mb in gpu)) ? 0
        Mb_gpu = Mb
        ind_b_free = np.where(Mb_gpu==-1) 
        ind_S_zero = np.where(S[ind_b_free]==0)
        ind_b_free_cpu = ind_b_free
        ind_S_zero_cpu = ind_S_zero
        ind_b_not_visited = np.full(n, True, dtype=bool) # boolean array
        ind_a_not_visited = np.full(n, True, dtype=bool)
        edges_released = ([],[])
        edges_pushed = ([],[])
        ind_b_not_pushed = ([])

        cur_S_zero_pt = 0
        for ind_b_tent in ind_b_free_cpu[0]:
            pushed = False
            while(cur_S_zero_pt < len(ind_S_zero_cpu[0]) and ind_b_tent == ind_b_free[0][ind_S_zero_cpu[0][cur_S_zero_pt]]):
                ind_a_tent = ind_S_zero_cpu[1][cur_S_zero_pt]
                cur_S_zero_pt += 1
                if ind_b_not_visited[ind_b_tent] and ind_a_not_visited[ind_a_tent]:
                    pushed = True
                    if(Ma[ind_a_tent] == -1):
                        f -= 1
                    else:
                        Mb[Ma[ind_a_tent]] = -1
                        edges_released[0].append(Ma[ind_a_tent])
                        edges_released[1].append(ind_a_tent)
                    edges_pushed[0].append(ind_b_tent)
                    edges_pushed[1].append(ind_a_tent)
                    ind_b_not_visited[ind_b_tent] = False
                    ind_a_not_visited[ind_a_tent] = False
                    Ma[ind_a_tent] = ind_b_tent
                    Mb[ind_b_tent] = ind_a_tent
                    yA[ind_a_tent] -= 1
            if not pushed:
                yB[ind_b_tent] += 1
                ind_b_not_pushed.append(ind_b_tent)

        edges_released_gpu = edges_released
        edges_pushed_gpu = edges_pushed
        ind_b_not_pushed_gpu = ind_b_not_pushed
        S[edges_released_gpu] += 1
        S[edges_pushed_gpu] -= 1
        S[:,edges_pushed_gpu[1]] += 1
        S[ind_b_not_pushed_gpu, :] -= 1
        iteration += 1

    # for ind_b in range(n):
    #     for ind_a in range(n):
    #         feasibility_check(ind_b, ind_a, yB, yA, Ma, cost)

    ind_a = 0
    for ind_b in range(n):
        if Mb[ind_b] == -1:
            while Ma[ind_a] != -1:
                ind_a += 1
            Mb[ind_b] = ind_a
            Ma[ind_a] = ind_b

    # matching_check(Ma, Mb)

    total_cost = 0
    for ind_b in range(n):
        total_cost += W[ind_b, Mb[ind_b]]
    return Mb, yA, yB, total_cost, iteration

def matching_gpu(W, W_cpu, C, delta, device):
    device2 = torch.device("cpu")
    dtyp = torch.int32

    # W_cpu = W.to(device2)
    n = W.shape[0]

    S = (3*W/(delta)).type(dtyp).to(device)
    # cost = (3*W/(delta)).type(dtyp).to(device)
    yB = np.ones(n, dtype=int)
    yA = np.zeros(n, dtype=int)
    Mb = np.ones(n, dtype=int) * -1
    Ma = np.ones(n, dtype=int) * -1

    range_n = np.arange(n)

    f = n
    iteration = 0

    ng = torch.tensor([n], device=device)[0]
    ones_ = torch.ones(ng, device=device)

    n = W.shape[0]
    zero = torch.tensor([0], device=device, dtype=dtyp, requires_grad=False)[0]
    one = torch.tensor([1], device=device, dtype=dtyp, requires_grad=False)[0]
    m_one = torch.tensor([-1], device=device, dtype=dtyp, requires_grad=False)[0]
    m_two = torch.tensor([-2], device=device, dtype=dtyp, requires_grad=False)[0]

    while f > n*delta/C:
        # need to test if needs to search Mb in gpu
        # T(ind_b_free transfer to cpu)-(T(search Mb in cpu)-T(search Mb in gpu)) ? 0
        # ind_b_free_cpu = np.where(Mb==-1)
        # ind_S_zero = cp.where(S[cp.array(ind_b_free_cpu[0])]==0)
        Mb_gpu = torch.tensor(Mb, device=device) #cp.array(Mb)
        ind_b_free = torch.where(Mb_gpu == m_one) #cp.where(Mb_gpu==-1)
        ind_S_zero = torch.where(S[ind_b_free] == zero) #cp.where(S[ind_b_free]==0)

        # ind_b_free_cpu = ind_b_free[0].get()
        # ind_S_zero_cpu = (ind_S_zero[0].get(),ind_S_zero[1].get())
        ind_b_free_cpu = ind_b_free[0].cpu().numpy()
        ind_S_zero_cpu = (ind_S_zero[0].cpu().numpy(),ind_S_zero[1].cpu().numpy())

        ind_b_not_visited = np.full(n, True, dtype=bool) # boolean array
        ind_a_not_visited = np.full(n, True, dtype=bool)
        edges_released = ([],[])
        edges_pushed = ([],[])
        ind_b_not_pushed = ([])

        cur_S_zero_pt = 0
        for ind_b_tent in ind_b_free_cpu:
            pushed = False
            while(cur_S_zero_pt < len(ind_S_zero_cpu[0]) and ind_b_tent == ind_b_free_cpu[ind_S_zero_cpu[0][cur_S_zero_pt]]):
                ind_a_tent = ind_S_zero_cpu[1][cur_S_zero_pt]
                cur_S_zero_pt += 1
                if ind_b_not_visited[ind_b_tent] and ind_a_not_visited[ind_a_tent]:
                    pushed = True
                    if(Ma[ind_a_tent] == -1):
                        f -= 1
                    else:
                        Mb[Ma[ind_a_tent]] = -1
                        edges_released[0].append(Ma[ind_a_tent])
                        edges_released[1].append(ind_a_tent)
                    edges_pushed[0].append(ind_b_tent)
                    edges_pushed[1].append(ind_a_tent)
                    ind_b_not_visited[ind_b_tent] = False
                    ind_a_not_visited[ind_a_tent] = False
                    Ma[ind_a_tent] = ind_b_tent
                    Mb[ind_b_tent] = ind_a_tent
                    yA[ind_a_tent] -= 1
            if not pushed:
                yB[ind_b_tent] += 1
                ind_b_not_pushed.append(ind_b_tent)

        # edges_released_gpu = cp.array(edges_released, dtype = int)
        # edges_pushed_gpu = cp.array(edges_pushed, dtype = int)
        # ind_b_not_pushed_gpu = cp.array(ind_b_not_pushed, dtype = int)
        edges_released_gpu = torch.tensor(edges_released, dtype = torch.long, device=device)
        edges_pushed_gpu = torch.tensor(edges_pushed, dtype = torch.long, device=device)
        ind_b_not_pushed_gpu = torch.tensor(ind_b_not_pushed, dtype = torch.long, device=device)


        S[edges_released_gpu[0], edges_released_gpu[1]] += one
        S[edges_pushed_gpu[0],edges_pushed_gpu[1]] -= one
        S[:,edges_pushed_gpu[1]] += one
        S[ind_b_not_pushed_gpu, :] -= one
        iteration += 1
        # break

    # for ind_b in range(n):
    #     for ind_a in range(n):
    #         feasibility_check(ind_b, ind_a, yB, yA, Ma, cost)

    ind_a = 0
    for ind_b in range(n):
        if Mb[ind_b] == -1:
            while Ma[ind_a] != -1:
                ind_a += 1
            Mb[ind_b] = ind_a
            Ma[ind_a] = ind_b

    # matching_check(Ma, Mb)

    total_cost = 0
    for ind_b in range(n):
        total_cost += W_cpu[ind_b, Mb[ind_b]]
    return Mb, yA, yB, total_cost, iteration

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
