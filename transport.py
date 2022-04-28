import numpy as np
from scipy.sparse import csc_matrix, coo_matrix

# def transport_cpu(DA, SB, C, delta):

#     n = C.shape[0]
#     F = csc_matrix(C.shape, dtype=int)
#     yFA = csc_matrix(C.shape, dtype=int)
#     yFB = csc_matrix(C.shape, dtype=int)
#     yA = np.zeros(n, dtype=int)
#     yB = np.zeros(n, dtype=int)
#     S = (C//(4*delta)).astype(int)

#     FreeA = DA
#     FreeB = SB
#     f = np.sum(SB)

#     iteration = 0

#     while f > n:
#         a_pushed = np.full(n, False)
#         for ind_b in range(n):
#             if FreeB[ind_b] > 0:
#                 ind_a_zero_slack = np.argwhere(S[ind_b]==0)
#                 for ind_a in ind_a_zero_slack[0]:
#                     if FreeA[ind_a] > 0:
#                         flow_to_push = min(FreeB[ind_b], FreeA[ind_a])
#                         F[ind_b, ind_a] += flow_to_push
#                         FreeB[ind_b] -= flow_to_push
#                         FreeA[ind_a] -= flow_to_push
#                         f -= flow_to_push
#                         yFA[ind_b, ind_a] = yA[ind_a] - 1
#                         a_pushed[ind_a] = True
#                     else:## not sure
#                         ind_b_relabel_candi = np.argwhere(F[:, ind_a] > 0 and yFA[:, ind_a] == yA[ind_a])
#                         ind_b_relabel_candi_pt = 0
#                         ind_b_relabel_candi_size = ind_b_relabel_candi.size
#                         while FreeB[ind_b] > 0 and ind_b_relabel_candi_pt < ind_b_relabel_candi_size:
#                             ind_b_relabel = ind_b_relabel_candi[ind_b_relabel_candi_pt]
#                             flow_to_push = min(FreeB[ind_b], F(ind_b_relabel, ind_a))
#                             F[ind_b, ind_a] += flow_to_push
#                             F[ind_b_relabel, ind_a] -= flow_to_push
#                             if F[ind_b_relabel, ind_a] == 0:
#                                 yFB[ind_b_relabel, ind_a] = 0 ## not sure here
#                                 FreeB[ind_b] += flow_to_push
#                                 FreeB[ind_b_relabel] -= flow_to_push
#                                 yFA[ind_b, ind_a] = yA[ind_a] - 1
#                                 a_pushed[ind_a] = True
#             if FreeB[ind_b] > 0:
#                 yB[ind_b] += 1
#                 S[ind_b, :] -= 1
        
#         ind_a_free_set = np.argwhere(FreeA > 0)
#         for ind_a_free in ind_a_free_set:
#             yA[ind_a_free] = np.max(yFA[:,ind_a_free])
#             if not a_pushed[ind_a_free]:
#                 S[:,ind_a_free] += 1
        
#         iteration += 1
        
    
#     for ind_b in range(n):
#         for ind_a in range(n):
#             if FreeB[ind_b] > 0 and FreeA[ind_a] > 0:
#                 flow_to_push = min(FreeB[ind_b], FreeA[ind_a])
#                 F[ind_b, ind_a] += flow_to_push
#                 FreeB[ind_b] -= flow_to_push
#                 FreeA[ind_a] -= flow_to_push
    
#     total_cost = F.multiply(C).sum()
#     return F, yA, yB, total_cost, iteration

def transport_new(DA, SB, C, delta):
    n = C.shape[0]
    F = csc_matrix(C.shape, dtype=int)
    yFA = csc_matrix(C.shape, dtype=int)
    # yFB = csc_matrix(C.shape, dtype=int)
    yA = np.zeros(n, dtype=int)
    yB = np.ones(n, dtype=int)
    C_scaled = ((4*C)//delta).astype(int)
    S = ((4*C)//delta).astype(int) #forward slack between yA and yB 
    PushableA = np.zeros((n,2), dtype=int) # better name? store number of copies with smaller/larger dual weights 
    #PushableA[:,0] -> avaliable mass of A
    #PushableA[:,1] -> unavaliable mass of A 

    max_C = np.max(C)
    alpha = 6 * n * max_C / delta

    FreeA = np.ceil(DA * alpha).astype(int)
    PushableA[:,0] = FreeA
    FreeB = (SB * alpha).astype(int)
    f = np.sum(FreeB) #flow remaining to push
    iteration = 0

    while f > n:
        FreeB_gpu = FreeB
        # ind_b_free = np.squeeze(np.argwhere(FreeB_gpu>0),0)
        ind_b_free = np.squeeze(np.argwhere(FreeB_gpu>0),1)
        # flow_validation_a(F, PushableA, FreeA)
        # flow_validation_b(f, FreeB)
        # slack_validation(yB, yA, S, C_scaled)
        # feasibilty_validation(yFA, yB, yA, F, C_scaled)
        ind_a_zero_slack = np.argwhere(S[ind_b_free,:]==0) 
        ind_b_free_cpu = ind_b_free
        ind_a_zero_slack_cpu = ind_a_zero_slack
        edges_full_released = ([],[])
        edges_part_released = ([],[])
        edges_pushed = ([],[])
        flow_pushed = ([])
        flow_released_ind_a = np.zeros(n, dtype=int)
        flow_partial_released = ([])
        ind_b_not_exhausted = ([])
        ind_a_exhausted = ([])

        cur_S_zero_pt = 0
        for ind_b in ind_b_free_cpu:
            b_exhausted = False
            while cur_S_zero_pt < len(ind_a_zero_slack_cpu) and ind_b == ind_b_free_cpu[ind_a_zero_slack_cpu[cur_S_zero_pt,0]]:
                ind_a = ind_a_zero_slack_cpu[cur_S_zero_pt,1]
                a_exhausted = ind_a in ind_a_exhausted
                cur_S_zero_pt += 1

                if not b_exhausted and not a_exhausted:
                    flow_to_push = min(FreeB[ind_b], PushableA[ind_a,0])
                    # push
                    FreeB[ind_b] -= flow_to_push
                    flow_pushed.append(flow_to_push)
                    # relabel
                    if FreeA[ind_a] > 0:
                        # It's guaranteed only push to either free or occuplied demands(A), need proof in paper
                        f -= flow_to_push
                        FreeA[ind_a] -= flow_to_push
                    else:
                        flow_released_ind_a[ind_a] += flow_to_push
                    # maintain variables
                    if flow_to_push == PushableA[ind_a,0]:
                        # yA[ind_a] -= 1
                        ind_a_exhausted.append(ind_a)
                    if FreeB[ind_b] == 0:
                        b_exhausted = True
                    PushableA[ind_a,1] += flow_to_push
                    PushableA[ind_a,0] -= flow_to_push
                    edges_pushed[0].append(ind_b)
                    edges_pushed[1].append(ind_a)
            if not b_exhausted:
                ind_b_not_exhausted.append(ind_b)
        
        # yFA[edges_self_released] = yA[edges_self_released[1]] - 1

        FreeB_release = np.zeros(n, dtype=int)
        # log relabel edges
        for ind_a in range(n):
            if ind_a in ind_a_exhausted and yA[ind_a] < 0:
                # List relabel candidates of B, select edges to release
                ind_b_relabel = np.argwhere(yFA[:,ind_a] == yA[ind_a])
                # log full released edges for slack & flow update
                if len(ind_b_relabel) > 0:
                    release_edge_b = ind_b_relabel[:,0].tolist()
                    release_edge_a = len(ind_b_relabel)*[ind_a]
                    edges_full_released[0].extend(release_edge_b)
                    edges_full_released[1].extend(release_edge_a)
                    FreeB_release[ind_b_relabel[:,0].tolist()] += np.squeeze(np.transpose(np.array(F[(release_edge_b, release_edge_a)])),1)
                # full_release_validate(ind_a, ind_b_relabel, F, flow_released_ind_a)
            elif PushableA[ind_a,0]>0 and yA[ind_a] < 0 and flow_released_ind_a[ind_a]>0:
                # List relabel candidates of B, select ind_b to full/partial release
                ind_b_relabel = np.argwhere(yFA[:,ind_a] == yA[ind_a])
                ind_b_full_released, ind_b_part_released = partial_release_a(F, ind_a, ind_b_relabel[:,0], flow_released_ind_a[ind_a])
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
                part_release_validate(ind_a, ind_b_full_released, ind_b_part_released, F, flow_released_ind_a, part_flow_ind_a)

        edges_full_released_gpu = edges_full_released
        edges_part_released_gpu = edges_part_released
        edges_pushed_gpu = edges_pushed
        ind_b_not_exhausted_gpu = ind_b_not_exhausted
        ind_a_exhausted_gpu = ind_a_exhausted
        # update variables
        PushableA[ind_a_exhausted, 0] = PushableA[ind_a_exhausted, 1]
        PushableA[ind_a_exhausted, 1] = 0

        # release mass in B and flow
        FreeB += FreeB_release
        if len(edges_full_released_gpu[0]) > 0:
            # FreeB[edges_full_released_gpu[0]] += np.squeeze(np.transpose(np.array(F[edges_full_released_gpu])))
            F[edges_full_released_gpu] = 0
            # S[edges_full_released_gpu] += 1
            yFA[edges_full_released_gpu] = 0 # set to default value when no flow
            # yFA[edges_full_released_gpu] += 1 
        if len(edges_part_released_gpu[0]) > 0:
            # FreeB[edges_part_released_gpu[0]] += np.array(flow_partial_released, dtype = int)
            F[edges_part_released_gpu] -= np.array(flow_partial_released, dtype = int)

        # push flow
        F[edges_pushed_gpu] += np.array(flow_pushed, dtype = int) # not efficient
        yFA[edges_pushed_gpu] = yA[edges_pushed_gpu[1]] - 1 # update dual weight of a on pushed edges 
        yA[ind_a_exhausted_gpu] -= 1
        

        # update slack
        # S[edges_pushed_gpu] += 1 # No need to update the slack of pushed edge unless corresponding vertex A exhausted
        # S[edges_self_released] += 1
        S[:,ind_a_exhausted_gpu] += 1
        # S[:,ind_a_exhausted_gpu] = C_scaled[:,ind_a_exhausted_gpu] + 1 - yA[np.newaxis, ind_a_exhausted_gpu] - yB[:,np.newaxis]
        S[ind_b_not_exhausted_gpu, :] -= 1
        yB[ind_b_not_exhausted_gpu] += 1

        # self release
        S_yFA = C_scaled + 1 - yFA - yB[:,np.newaxis]
        ind_self_release = np.nonzero((S_yFA==0) & (F!=0).toarray())
        # ind_self_release = ind_self_release[np.nonzero(yFA[ind_self_release] == yA[ind_self_release[1]])] try this!!!
        if len(ind_self_release[0]) > 0:
            flow_release = np.squeeze(np.array(F[ind_self_release]))
            np.add.at(PushableA[:,1], ind_self_release[1], flow_release)
            np.subtract.at(PushableA[:,0], ind_self_release[1], flow_release)
            # Need to consider the opreatioins in places that are indexed more than once  
            # PushableA[ind_self_release[1],1] += flow_release
            # PushableA[ind_self_release[1],0] -= flow_release
            ind_a_exhausted = np.nonzero(PushableA[:,0]==0)[0]
            PushableA[ind_a_exhausted, 0] = PushableA[ind_a_exhausted, 1]
            PushableA[ind_a_exhausted, 1] = 0
            yFA[ind_self_release] = yA[ind_self_release[1]] - 1
            yA[ind_a_exhausted] -= 1
            # S[ind_self_release] += 1
            S[:,ind_a_exhausted] += 1
            # S[:,ind_a_exhausted] = C_scaled[:,ind_a_exhausted] + 1 - yA[np.newaxis, ind_a_exhausted] - yB[:,np.newaxis]
        iteration += 1

    for ind_b in range(n):
        for ind_a in range(n):
            if FreeB[ind_b] > 0 and FreeA[ind_a] > 0:
                flow_to_push = min(FreeB[ind_b], FreeA[ind_a])
                F[ind_b, ind_a] += flow_to_push
                FreeB[ind_b] -= flow_to_push
                FreeA[ind_a] -= flow_to_push   
    total_cost = F.multiply(C).sum()
    return F, yA, yB, total_cost, iteration
    
def partial_release_a(F, ind_a, ind_b_relabel, flow_to_push):
    cumsum_flow = np.cumsum(F[ind_b_relabel, ind_a].toarray())
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

def flow_validation_a(F, PushableA, FreeA):
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

def slack_validation(yB, yA, S, C):
    # Only need to check S with yA and yB
    if (S != C + 1 - yB[:,np.newaxis] - yA[np.newaxis, :]).any():
        print("slack not valid")

# def pushableA_validation(PushableA, yA, yFA, F):


# def yA_validation(yFA, yA):
#     ind
#     yA_max = np