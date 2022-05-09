from os import times_result
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
from scipy.spatial.distance import cdist
import time
from line_profiler import LineProfiler

def transport_cpu_sparse(DA, SB, C, delta):
    start = time.time()
    n = C.shape[0]
    yA = np.zeros(n, dtype=int)
    yB = np.ones(n, dtype=int)
    PushableA = np.zeros((n,2), dtype=int) # better name? store number of copies with smaller/larger dual weights
    #PushableA[:,0] -> avaliable mass of A
    #PushableA[:,1] -> unavaliable mass of A 
    F = csr_matrix(C.shape, dtype=int)
    # F = np.zeros(C.shape, dtype=int)
    yFA = csc_matrix(C.shape, dtype=int)
    # C_scaled = ((4*C)//delta).astype(int)
    S = ((4*C)//delta).astype(int) #forward slack between yA and yB 

    max_C = np.max(C)
    alpha = 6 * n * max_C / delta

    FreeA = np.ceil(DA * alpha).astype(int)
    PushableA[:,0] = FreeA
    FreeB = (SB * alpha).astype(int)
    f = np.sum(FreeB) #flow remaining to push
    iteration = 0
    time_log = []
    print("variable init: {}s".format(time.time()-start))

    start_out = time.time()

    # main loop
    while f > n:
        time_temp = []
        FreeB_gpu = FreeB
        ind_b_free = np.squeeze(np.argwhere(FreeB_gpu>0),1) 
        # flow_validation_a_sparse(F, PushableA, FreeA)
        # flow_validation_b(f, FreeB)
        # slack_validation(yB, yA, S, C_scaled)
        # feasibilty_validation_sparse(yFA, yB, yA, F, C_scaled)
        ind_zero_slack = np.argwhere(S[ind_b_free,:]==0) 
        ind_b_free_cpu = ind_b_free
        ind_zero_slack_cpu = ind_zero_slack
        edges_full_released = ([],[])
        edges_part_released = ([],[])
        edges_pushed = ([],[])
        flow_pushed = ([])
        flow_released_ind_a = np.zeros(n, dtype=int)
        flow_partial_released = ([])
        ind_b_not_exhausted = ([])
        # ind_a_exhausted = ([])
        ind_a_exhausted = np.zeros(n, dtype=bool)

        cur_S_zero_pt = 0

        # log pushed edges
        start = time.time()
        if len(ind_zero_slack[0]) > 0:
            ind_zero_slack_b_range = find_ind_range(ind_zero_slack[:,0])

        # for ind_b in ind_b_free_cpu:
        for ind_b_free_index in range(len(ind_b_free_cpu)):
            ind_b = ind_b_free_cpu[ind_b_free_index]
            b_exhausted = False
            try:
                cur_S_zero_pt = ind_zero_slack_b_range[ind_b_free_index][0]
            except:
                ind_b_not_exhausted.append(ind_b)
                continue
                
            # while cur_S_zero_pt < len(ind_zero_slack_cpu) and ind_b == ind_b_free_cpu[ind_zero_slack_cpu[cur_S_zero_pt,0]]:
            while not b_exhausted and cur_S_zero_pt < len(ind_zero_slack_cpu) and ind_b == ind_b_free_cpu[ind_zero_slack_cpu[cur_S_zero_pt,0]]:
                ind_a = ind_zero_slack_cpu[cur_S_zero_pt,1]
                a_exhausted = ind_a_exhausted[ind_a]
                # a_exhausted = ind_a in ind_a_exhausted # 5%
                cur_S_zero_pt += 1
                # if not b_exhausted and not a_exhausted:
                if not a_exhausted:
                    flow_to_push = min(FreeB[ind_b], PushableA[ind_a,0])
                    # push
                    FreeB[ind_b] -= flow_to_push
                    flow_pushed.append(flow_to_push)
                    # relabel
                    if FreeA[ind_a] > 0:
                        # It's guaranteed only push to either free or occuplied demands(A)
                        f -= flow_to_push
                        FreeA[ind_a] -= flow_to_push
                    else:
                        flow_released_ind_a[ind_a] += flow_to_push
                    # maintain variables
                    if flow_to_push == PushableA[ind_a,0]:
                        ind_a_exhausted[ind_a] = True
                        # ind_a_exhausted.append(ind_a)
                    if FreeB[ind_b] == 0:
                        b_exhausted = True
                    PushableA[ind_a,1] += flow_to_push
                    PushableA[ind_a,0] -= flow_to_push
                    edges_pushed[0].append(ind_b)
                    edges_pushed[1].append(ind_a)
            if not b_exhausted:
                ind_b_not_exhausted.append(ind_b)

        time_temp.append(time.time()-start)
        start = time.time()

        FreeB_release = np.zeros(n, dtype=int)
        ind_a_release = np.nonzero(flow_released_ind_a)[0]

        # log released edges
        all_ind_b_relabel = np.nonzero(np.transpose(yFA[:,ind_a_release]) == yA[ind_a_release][:,None])
        all_ind_b_relabel = (all_ind_b_relabel[1],ind_a_release[all_ind_b_relabel[0]])
        if len(all_ind_b_relabel[1]) > 0:
            ind_b_range = find_ind_range(all_ind_b_relabel[1])

        for ind_a in ind_a_release:
        # for ind_a in range(n): #5.9%
            # if ind_a_exhausted[ind_a] and yA[ind_a] < 0:
            if ind_a_exhausted[ind_a]:
            # if ind_a in ind_a_exhausted and yA[ind_a] < 0: #17.8%
                # List relabel candidates of B, select edges to release
                # ind_b_relabel = np.argwhere(yFA[:,ind_a] == yA[ind_a])
                ind_b_relabel = all_ind_b_relabel[0][ind_b_range[ind_a][0]:ind_b_range[ind_a][1]+1]
                # log full released edges for slack & flow update
                if len(ind_b_relabel) > 0:
                    # release_edge_b = ind_b_relabel[:,0].tolist()
                    release_edge_b = ind_b_relabel.tolist()
                    release_edge_a = len(ind_b_relabel)*[ind_a]
                    edges_full_released[0].extend(release_edge_b)
                    edges_full_released[1].extend(release_edge_a)
                    # FreeB_release[ind_b_relabel[:,0].tolist()] += np.squeeze(np.transpose(np.array(F[(release_edge_b, release_edge_a)])),1)
                    FreeB_release[ind_b_relabel.tolist()] += np.squeeze(np.transpose(np.array(F[(release_edge_b, release_edge_a)])),1)
                # full_release_validate(ind_a, ind_b_relabel, F, flow_released_ind_a)
            # elif PushableA[ind_a,0]>0 and yA[ind_a] < 0 and flow_released_ind_a[ind_a]>0: #10.5%
            else:
                # List relabel candidates of B, select ind_b to full/partial release
                # yFA_temp = yFA[:,ind_a]
                # ind_b_relabel = np.argwhere(yFA[:,ind_a] == yA[ind_a])
                ind_b_relabel = all_ind_b_relabel[0][ind_b_range[ind_a][0]:ind_b_range[ind_a][1]+1]
                # ind_b_full_released, ind_b_part_released = partial_release_a_sparse(F, ind_a, ind_b_relabel[:,0], flow_released_ind_a[ind_a])
                ind_b_full_released, ind_b_part_released = partial_release_a_sparse(F, ind_a, ind_b_relabel, flow_released_ind_a[ind_a])
                full_released_flow_sum = 0
                part_flow_ind_a = 0
                # log full/partial released edges for slack & flow update
                if ind_b_full_released is not None:
                    release_edge_b = ind_b_full_released.tolist()
                    release_edge_a = len(ind_b_full_released)*[ind_a]
                    edges_full_released[0].extend(release_edge_b)
                    edges_full_released[1].extend(release_edge_a)
                    full_released_flows = np.squeeze(np.transpose(np.array(F[(release_edge_b,release_edge_a)])),1)
                    full_released_flow_sum = np.sum(full_released_flows)
                    FreeB_release[release_edge_b] += full_released_flows
                if ind_b_part_released is not None:
                    edges_part_released[0].append(ind_b_part_released)
                    edges_part_released[1].append(ind_a)
                    part_flow_ind_a = flow_released_ind_a[ind_a] - full_released_flow_sum
                    flow_partial_released.append(part_flow_ind_a)
                    FreeB_release[ind_b_part_released] += part_flow_ind_a
                # part_release_validate(ind_a, ind_b_full_released, ind_b_part_released, F, flow_released_ind_a, part_flow_ind_a)

        time_temp.append(time.time()-start)
        start = time.time()

        # update variables
        edges_full_released_gpu = edges_full_released
        edges_part_released_gpu = edges_part_released
        edges_pushed_gpu = edges_pushed
        ind_b_not_exhausted_gpu = ind_b_not_exhausted
        ind_a_exhausted_gpu = ind_a_exhausted

        PushableA[ind_a_exhausted, 0] = PushableA[ind_a_exhausted, 1]
        PushableA[ind_a_exhausted, 1] = 0

        # release mass in B and flow (release goes before push)
        FreeB += FreeB_release
        if len(edges_full_released_gpu[0]) > 0:
            F[edges_full_released_gpu] = 0
            yFA[edges_full_released_gpu] = 0 # set to default value when no flow
        if len(edges_part_released_gpu[0]) > 0:
            F[edges_part_released_gpu] -= np.array(flow_partial_released, dtype = int)

        # push flow
        F[edges_pushed_gpu] += np.array(flow_pushed, dtype = int) # not efficient
        yFA[edges_pushed_gpu] = yA[edges_pushed_gpu[1]] - 1 # update dual weight of a on pushed edges 
        yA[ind_a_exhausted_gpu] -= 1

        # update slack
        S[:,ind_a_exhausted_gpu] += 1 # No need to update the slack of pushed edge unless corresponding vertex A exhausted
        S[ind_b_not_exhausted_gpu, :] -= 1
        yB[ind_b_not_exhausted_gpu] += 1

        time_temp.append(time.time()-start)
        # S_yFA = C_scaled + 1 - yFA - yB[:,np.newaxis]
        start = time.time()

        # self release
        ind_f = np.nonzero(F[ind_b_not_exhausted_gpu,:]!=0)
        ind_self_release = (np.array(ind_b_not_exhausted_gpu)[ind_f[0]], ind_f[1])
        # S_yFA = C_scaled + 1 - yFA - yB[:,np.newaxis] # S != S_yFA, can reduce time by maintaining S_yFA 7.9%
        # ind_temp = np.nonzero(S_yFA[ind_b_not_exhausted_gpu[ind_f[0]],ind_f[1]]==0)
        # ind_self_release = (ind_b_not_exhausted_gpu[ind_f[0][ind_temp]], ind_f[1][ind_temp])
        # ind_self_release = np.nonzero((S_yFA==0) & (F!=0)) # 16.9%
        if len(ind_self_release[0]) > 0:
            flow_release = np.squeeze(np.array(F[ind_self_release]))
            # Need to consider the opreatioins in places that are indexed more than once  
            np.add.at(PushableA[:,1], ind_self_release[1], flow_release)
            np.subtract.at(PushableA[:,0], ind_self_release[1], flow_release)
            ind_a_exhausted = np.nonzero(PushableA[:,0]==0)[0]
            PushableA[ind_a_exhausted, 0] = PushableA[ind_a_exhausted, 1]
            PushableA[ind_a_exhausted, 1] = 0
            yFA[ind_self_release] = yA[ind_self_release[1]] - 1
            yA[ind_a_exhausted] -= 1
            S[:,ind_a_exhausted] += 1
        iteration += 1

        time_temp.append(time.time()-start)
        time_log.append(time_temp)
    print("main loop: {}s".format(time.time()-start_out))

    start = time.time()
    ind_a_left = np.nonzero(FreeA > 0)[0].tolist()
    ind_a_left_next = ind_a_left.copy()
    ind_b_left = np.nonzero(FreeB > 0)[0].tolist()
    for ind_b in ind_b_left:
        for ind_a in ind_a_left:
            flow_to_push = min(FreeB[ind_b], FreeA[ind_a])
            F[ind_b, ind_a] += flow_to_push
            FreeB[ind_b] -= flow_to_push
            FreeA[ind_a] -= flow_to_push
            # f -= flow_to_push
            if FreeA[ind_a] == 0:
                ind_a_left_next.remove(ind_a)
            if FreeB[ind_b] == 0:
                break
        ind_a_left = ind_a_left_next
        ind_a_left_next = ind_a_left.copy()
    print("arbitrary matching: {}s".format(time.time()-start))

    start = time.time()
    F = F/alpha
    total_cost = np.sum(F*C)
    print("gen outputs: {}s".format(time.time()-start))
    return F, yA, yB, total_cost, iteration, time_log

def transport_cpu(DA, SB, C, delta):
    start = time.time()
    n = C.shape[0]
    yA = np.zeros(n, dtype=int)
    yB = np.ones(n, dtype=int)
    PushableA = np.zeros((n,2), dtype=int) # better name? store number of copies with smaller/larger dual weights
    #PushableA[:,0] -> avaliable mass of A
    #PushableA[:,1] -> unavaliable mass of A 
    F = np.zeros(C.shape, dtype=int)
    yFA = np.zeros(C.shape, dtype=int)
    # C_scaled = ((4*C)//delta).astype(int)
    S = ((4*C)//delta).astype(int) #forward slack between yA and yB 
    # S1 = np.floor_divide(np.multiply(C, 4), delta)

    max_C = np.max(C)
    alpha = 6 * n * max_C / delta

    FreeA = np.ceil(DA * alpha).astype(int)
    PushableA[:,0] = FreeA
    FreeB = (SB * alpha).astype(int)
    f = np.sum(FreeB) #flow remaining to push
    iteration = 0
    time_log = []
    print("variable init: {}s".format(time.time()-start))

    start_out = time.time()

    # main loop
    while f > n:
        time_temp = []
        FreeB_gpu = FreeB
        ind_b_free = np.squeeze(np.argwhere(FreeB_gpu>0),1) 
        # flow_validation_a(F, PushableA, FreeA)
        # flow_validation_b(f, FreeB)
        # slack_validation(yB, yA, S, C_scaled)
        # feasibilty_validation(yFA, yB, yA, F, C_scaled)
        ind_zero_slack = np.argwhere(S[ind_b_free,:]==0) 
        ind_b_free_cpu = ind_b_free
        ind_zero_slack_cpu = ind_zero_slack
        edges_full_released = ([],[])
        edges_part_released = ([],[])
        edges_pushed = ([],[])
        flow_pushed = ([])
        flow_released_ind_a = np.zeros(n, dtype=int)
        flow_partial_released = ([])
        ind_b_not_exhausted = ([])
        # ind_a_exhausted = ([])
        ind_a_exhausted = np.zeros(n, dtype=bool)

        cur_S_zero_pt = 0

        # log pushed edges
        start = time.time()
        if len(ind_zero_slack[0]) > 0:
            ind_zero_slack_b_range = find_ind_range(ind_zero_slack[:,0])

        # for ind_b in ind_b_free_cpu:
        for ind_b_free_index in range(len(ind_b_free_cpu)):
            ind_b = ind_b_free_cpu[ind_b_free_index]
            b_exhausted = False
            try:
                cur_S_zero_pt = ind_zero_slack_b_range[ind_b_free_index][0]
            except:
                ind_b_not_exhausted.append(ind_b)
                continue
            # while cur_S_zero_pt < len(ind_zero_slack_cpu) and ind_b == ind_b_free_cpu[ind_zero_slack_cpu[cur_S_zero_pt,0]]:
            while not b_exhausted and cur_S_zero_pt < len(ind_zero_slack_cpu) and ind_b == ind_b_free_cpu[ind_zero_slack_cpu[cur_S_zero_pt,0]]:
                ind_a = ind_zero_slack_cpu[cur_S_zero_pt,1]
                a_exhausted = ind_a_exhausted[ind_a]
                # a_exhausted = ind_a in ind_a_exhausted # 5%
                cur_S_zero_pt += 1

                if not b_exhausted and not a_exhausted:
                    flow_to_push = min(FreeB[ind_b], PushableA[ind_a,0])
                    # push
                    FreeB[ind_b] -= flow_to_push
                    flow_pushed.append(flow_to_push)
                    # relabel
                    if FreeA[ind_a] > 0:
                        # It's guaranteed only push to either free or occuplied demands(A)
                        f -= flow_to_push
                        FreeA[ind_a] -= flow_to_push
                    else:
                        flow_released_ind_a[ind_a] += flow_to_push
                    # maintain variables
                    if flow_to_push == PushableA[ind_a,0]:
                        ind_a_exhausted[ind_a] = True
                        # ind_a_exhausted.append(ind_a)
                    if FreeB[ind_b] == 0:
                        b_exhausted = True
                    PushableA[ind_a,1] += flow_to_push
                    PushableA[ind_a,0] -= flow_to_push
                    edges_pushed[0].append(ind_b)
                    edges_pushed[1].append(ind_a)
            if not b_exhausted:
                ind_b_not_exhausted.append(ind_b)

        time_temp.append(time.time()-start)
        start = time.time()

        FreeB_release = np.zeros(n, dtype=int)
        ind_a_release = np.nonzero(flow_released_ind_a)[0]
        # log released edges
        all_ind_b_relabel = np.nonzero(np.transpose(yFA[:,ind_a_release]) == yA[ind_a_release][:,None])
        all_ind_b_relabel = (all_ind_b_relabel[1],ind_a_release[all_ind_b_relabel[0]])
        if len(all_ind_b_relabel[1]) > 0:
            ind_b_range = find_ind_range(all_ind_b_relabel[1])

        for ind_a in ind_a_release:
        # for ind_a in range(n): #5.9%
            # if ind_a_exhausted[ind_a] and yA[ind_a] < 0:
            if ind_a_exhausted[ind_a]:
            # if ind_a in ind_a_exhausted and yA[ind_a] < 0: #17.8%
                # List relabel candidates of B, select edges to release
                ind_b_relabel = all_ind_b_relabel[0][ind_b_range[ind_a][0]:ind_b_range[ind_a][1]+1]
                # ind_b_relabel = np.argwhere(yFA[:,ind_a] == yA[ind_a])
                # log full released edges for slack & flow update
                if len(ind_b_relabel) > 0:
                    release_edge_b = ind_b_relabel.tolist()
                    release_edge_a = len(ind_b_relabel)*[ind_a]
                    edges_full_released[0].extend(release_edge_b)
                    edges_full_released[1].extend(release_edge_a)
                    FreeB_release[ind_b_relabel.tolist()] += F[(release_edge_b, release_edge_a)]
                # full_release_validate(ind_a, ind_b_relabel, F, flow_released_ind_a)
            # elif PushableA[ind_a,0]>0 and yA[ind_a] < 0 and flow_released_ind_a[ind_a]>0: #10.5%
            else:
                # List relabel candidates of B, select ind_b to full/partial release
                # ind_b_relabel = np.argwhere(yFA[:,ind_a] == yA[ind_a])
                ind_b_relabel = all_ind_b_relabel[0][ind_b_range[ind_a][0]:ind_b_range[ind_a][1]+1]
                ind_b_full_released, ind_b_part_released = partial_release_a(F, ind_a, ind_b_relabel, flow_released_ind_a[ind_a])
                full_released_flow_sum = 0
                part_flow_ind_a = 0
                # log full/partial released edges for slack & flow update
                if ind_b_full_released is not None:
                    release_edge_b = ind_b_full_released.tolist()
                    release_edge_a = len(ind_b_full_released)*[ind_a]
                    edges_full_released[0].extend(release_edge_b)
                    edges_full_released[1].extend(release_edge_a)
                    full_released_flows = F[(release_edge_b,release_edge_a)]
                    full_released_flow_sum = np.sum(full_released_flows)
                    FreeB_release[release_edge_b] += full_released_flows
                if ind_b_part_released is not None:
                    edges_part_released[0].append(ind_b_part_released)
                    edges_part_released[1].append(ind_a)
                    part_flow_ind_a = flow_released_ind_a[ind_a] - full_released_flow_sum
                    flow_partial_released.append(part_flow_ind_a)
                    FreeB_release[ind_b_part_released] += part_flow_ind_a
                # part_release_validate(ind_a, ind_b_full_released, ind_b_part_released, F, flow_released_ind_a, part_flow_ind_a)

        time_temp.append(time.time()-start)
        start = time.time()

        # update variables
        edges_full_released_gpu = edges_full_released
        edges_part_released_gpu = edges_part_released
        edges_pushed_gpu = edges_pushed
        ind_b_not_exhausted_gpu = ind_b_not_exhausted
        ind_a_exhausted_gpu = ind_a_exhausted

        PushableA[ind_a_exhausted, 0] = PushableA[ind_a_exhausted, 1]
        PushableA[ind_a_exhausted, 1] = 0

        # release mass in B and flow (release goes before push)
        FreeB += FreeB_release
        if len(edges_full_released_gpu[0]) > 0:
            F[edges_full_released_gpu] = 0
            yFA[edges_full_released_gpu] = 0 # set to default value when no flow
        if len(edges_part_released_gpu[0]) > 0:
            F[edges_part_released_gpu] -= np.array(flow_partial_released, dtype = int)

        # push flow
        F[edges_pushed_gpu] += np.array(flow_pushed, dtype = int) # not efficient
        yFA[edges_pushed_gpu] = yA[edges_pushed_gpu[1]] - 1 # update dual weight of a on pushed edges 
        yA[ind_a_exhausted_gpu] -= 1

        # update slack
        S[:,ind_a_exhausted_gpu] += 1 # No need to update the slack of pushed edge unless corresponding vertex A exhausted
        S[ind_b_not_exhausted_gpu, :] -= 1
        yB[ind_b_not_exhausted_gpu] += 1

        time_temp.append(time.time()-start)
        # S_yFA = C_scaled + 1 - yFA - yB[:,np.newaxis]
        start = time.time()

        # self release
        ind_f = np.nonzero(F[ind_b_not_exhausted_gpu,:]!=0)
        ind_self_release = (np.array(ind_b_not_exhausted_gpu)[ind_f[0]], ind_f[1])
        # S_yFA = C_scaled + 1 - yFA - yB[:,np.newaxis] # S != S_yFA, can reduce time by maintaining S_yFA 7.9%
        # ind_temp = np.nonzero(S_yFA[ind_b_not_exhausted_gpu[ind_f[0]],ind_f[1]]==0)
        # ind_self_release = (ind_b_not_exhausted_gpu[ind_f[0][ind_temp]], ind_f[1][ind_temp])
        # ind_self_release = np.nonzero((S_yFA==0) & (F!=0)) # 16.9%
        if len(ind_self_release[0]) > 0:
            flow_release = np.squeeze(np.array(F[ind_self_release]))
            # Need to consider the opreatioins in places that are indexed more than once  
            np.add.at(PushableA[:,1], ind_self_release[1], flow_release)
            np.subtract.at(PushableA[:,0], ind_self_release[1], flow_release)
            ind_a_exhausted = np.nonzero(PushableA[:,0]==0)[0]
            PushableA[ind_a_exhausted, 0] = PushableA[ind_a_exhausted, 1]
            PushableA[ind_a_exhausted, 1] = 0
            yFA[ind_self_release] = yA[ind_self_release[1]] - 1
            yA[ind_a_exhausted] -= 1
            S[:,ind_a_exhausted] += 1
        iteration += 1

        time_temp.append(time.time()-start)
        time_log.append(time_temp)
    print("main loop: {}s".format(time.time()-start_out))

    start = time.time()
    ind_a_left = np.nonzero(FreeA > 0)[0].tolist()
    ind_a_left_next = ind_a_left.copy()
    ind_b_left = np.nonzero(FreeB > 0)[0].tolist()
    for ind_b in ind_b_left:
        for ind_a in ind_a_left:
            flow_to_push = min(FreeB[ind_b], FreeA[ind_a])
            F[ind_b, ind_a] += flow_to_push
            FreeB[ind_b] -= flow_to_push
            FreeA[ind_a] -= flow_to_push
            # f -= flow_to_push
            if FreeA[ind_a] == 0:
                ind_a_left_next.remove(ind_a)
            if FreeB[ind_b] == 0:
                break
        ind_a_left = ind_a_left_next
        ind_a_left_next = ind_a_left.copy()
    print("arbitrary matching: {}s".format(time.time()-start))

    start = time.time()
    F = F/alpha
    total_cost = np.sum(F*C)
    print("gen outputs: {}s".format(time.time()-start))
    return F, yA, yB, total_cost, iteration, time_log
    
def find_ind_range(ind):
    n = len(ind)
    st = 0
    cur = 0
    val = ind[st]
    ret = dict()
    while(cur < n):
        if ind[cur] != val:
            ret[val] = (st, cur-1)
            st = cur
            val = ind[st]
        cur += 1
    ret[val] = (st, cur-1)
    return ret

def partial_release_a(F, ind_a, ind_b_relabel, flow_to_push):
    cumsum_flow = np.cumsum(F[ind_b_relabel, ind_a])
    part_released_ind = np.argwhere(cumsum_flow > flow_to_push)[0][0]
    if part_released_ind == 0:
        ind_b_full_released = None
        ind_b_part_released = ind_b_relabel[part_released_ind]
    else:
        ind_b_full_released = ind_b_relabel[0:part_released_ind]
        ind_b_part_released = ind_b_relabel[part_released_ind]

    return ind_b_full_released, ind_b_part_released

def partial_release_a_sparse(F, ind_a, ind_b_relabel, flow_to_push):
    F_temp = F[ind_b_relabel, ind_a].toarray()
    cumsum_flow = np.cumsum(F_temp)
    # cumsum_flow = np.cumsum(F[ind_b_relabel, ind_a])
    part_released_ind = np.argwhere(cumsum_flow > flow_to_push)[0][0]
    if part_released_ind == 0:
        ind_b_full_released = None
        ind_b_part_released = ind_b_relabel[part_released_ind]
    else:
        ind_b_full_released = ind_b_relabel[0:part_released_ind]
        ind_b_part_released = ind_b_relabel[part_released_ind]

    return ind_b_full_released, ind_b_part_released

def full_release_validate(ind_a, ind_b_relabel, F, flow_released_ind_a):
    edges_released = (ind_b_relabel[:,0].tolist(), len(ind_b_relabel)*[ind_a])
    if not flow_released_ind_a[ind_a] == np.sum(F[edges_released]):
        print("the sum of full released edges not equals to released flow log")

def full_release_validate_sparse(ind_a, ind_b_relabel, F, flow_released_ind_a):
    edges_released = (ind_b_relabel[:,0].tolist(), len(ind_b_relabel)*[ind_a])
    if not flow_released_ind_a[ind_a] == np.sum(F[edges_released]):
        print("the sum of full released edges not equals to released flow log")

def part_release_validate(ind_a, ind_b_full_released, ind_b_part_released, F, flow_released_ind_a, part_flow_ind_a):
    if ind_b_full_released is not None:
        edges_released_full = (ind_b_full_released.tolist(), len(ind_b_full_released)*[ind_a])
        flow_released_full = np.sum(F[edges_released_full])
    else:
        flow_released_full = 0
    
    if ind_b_part_released is not None:
        edges_released_part = (ind_b_part_released, ind_a)
        if part_flow_ind_a > F[edges_released_part]:
            print("the part release flow larger than actual flow in edge")
    
    if not flow_released_ind_a[ind_a] == flow_released_full + part_flow_ind_a:
        print("the sum of part released edges not equals to released flow log")

def part_release_validate_sparse(ind_a, ind_b_full_released, ind_b_part_released, F, flow_released_ind_a, part_flow_ind_a):
    if ind_b_full_released is not None:
        edges_released_full = (ind_b_full_released.tolist(), len(ind_b_full_released)*[ind_a])
        flow_released_full = np.sum(F[edges_released_full])
    else:
        flow_released_full = 0
    
    if ind_b_part_released is not None:
        edges_released_part = (ind_b_part_released, ind_a)
        if part_flow_ind_a > F[edges_released_part]:
            print("the part release flow larger than actual flow in edge")
    
    if not flow_released_ind_a[ind_a] == flow_released_full + part_flow_ind_a:
        print("the sum of part released edges not equals to released flow log")

def flow_validation_a(F, PushableA, FreeA):
    if (np.sum(F,0) + FreeA != np.sum(PushableA, 1)).any():
        print("flow not validate for type A")

def flow_validation_a_sparse(F, PushableA, FreeA):
    if (np.squeeze(np.transpose(np.array(np.sum(F, 0) + FreeA)), 1) != np.sum(PushableA, 1)).any():
        print("flow not validate for type A")

def flow_validation_b(f, FreeB):
    if f != np.sum(FreeB):
        print("flow not validate for type B")

def feasibilty_validation(yFA, yB, yA, F, C):
    zero_f_ind = np.nonzero(F == 0)
    nonzero_f_ind = np.nonzero(F > 0)
    if len(zero_f_ind[0]) > 0 and (yA[zero_f_ind[1]] + yB[zero_f_ind[0]] > C[zero_f_ind] + 1).any():
        print("first feasibility condition not valid (F=0)")
    if len(nonzero_f_ind[0]) > 0 and (yFA[nonzero_f_ind] + yB[nonzero_f_ind[0]] > C[nonzero_f_ind] + 1).any():
        print("first feasibility condition not valid (F>0)")
    if len(nonzero_f_ind[0]) > 0 and (yFA[nonzero_f_ind] + yB[nonzero_f_ind[0]] < C[nonzero_f_ind]).any():
        print("second feasibility condition not valid")

def feasibilty_validation_sparse(yFA, yB, yA, F, C):
    zero_f_ind = np.nonzero(F == 0)
    nonzero_f_ind = np.nonzero(F > 0)
    if len(zero_f_ind[0]) > 0 and (yA[zero_f_ind[1]] + yB[zero_f_ind[0]] > C[zero_f_ind] + 1).any():
        print("first feasibility condition not valid (F=0)")
    if len(nonzero_f_ind[0]) > 0 and (yFA[nonzero_f_ind] + yB[nonzero_f_ind[0]] > C[nonzero_f_ind] + 1).any():
        print("first feasibility condition not valid (F>0)")
    if len(nonzero_f_ind[0]) > 0 and (yFA[nonzero_f_ind] + yB[nonzero_f_ind[0]] < C[nonzero_f_ind]).any():
        print("second feasibility condition not valid")

def slack_validation(yB, yA, S, C):
    # Only need to check S with yA and yB
    if (S != C + 1 - yB[:,np.newaxis] - yA[np.newaxis, :]).any():
        print("slack not valid")

def rand_inputs(n = 100, seed = 0):
    """
    This function creates synthetic experiment data by randomly generating points in a 2d unit square.
    """
    np.random.seed(seed)
    a = np.random.rand(n,2)
    b = np.random.rand(n,2)
    cost = cdist(a, b, 'sqeuclidean')
    DA = np.random.rand(n)
    SB = np.random.rand(n)
    return DA, SB, cost


DA, SB, cost = rand_inputs(10000, 0)
DA = DA/np.sum(DA)
SB = SB/np.sum(SB)
C = cost.max()
cost /= C
delta = 0.01

# lp = LineProfiler()
# lp_wrapper = lp(transport_cpu)
# Mb, yA, yB, ot_loss, iteration, time_log = lp_wrapper(DA, SB, cost, delta)
# lp.print_stats()

# lp = LineProfiler()
# lp_wrapper = lp(transport_cpu_sparse)
# Mb, yA, yB, ot_loss, iteration, time_log = lp_wrapper(DA, SB, cost, delta)
# lp.print_stats()

start = time.time()
Mb, yA, yB, ot_loss, iteration, time_log = transport_cpu(DA, SB, cost, delta)
# Mb, yA, yB, ot_loss, iteration, time_log = transport_cpu_sparse(DA, SB, cost, delta)
end = time.time()
dt = end-start
print("ot function total: {}s".format(time.time()-start))
print("in main loop: edge push logging | edge release logging | variable update | self release")
print(np.sum(time_log,0))
print("main loop total:")
print(np.sum(time_log))