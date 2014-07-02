# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
from cython.parallel import prange
from libc.math cimport exp
cimport cython
cimport numpy as np
cimport openmp
from libcpp cimport bool
from libc.stdio cimport printf
np.import_array()

cdef extern from "stdlib.h":
    void srand(unsigned int seed) 

ctypedef size_t NodeId
ctypedef size_t EdgeId
ctypedef float Real

cdef extern from "QPBO.h":
    cdef cppclass QPBO[REAL]:
        QPBO(unsigned int node_num_max, unsigned int edge_num_max)
        NodeId AddNode(unsigned int num)
        void AddUnaryTerm(NodeId i, REAL E0, REAL E1)
        EdgeId AddPairwiseTerm(NodeId i, NodeId j, REAL E00, REAL E01, REAL E10, REAL E11)

        int GetLabel(NodeId i)
        void Solve()
        void ComputeWeakPersistencies()


cdef extern from "LSA.h":
    cdef cppclass LSA[REAL]:
        LSA(unsigned int node_num_max, unsigned int edge_num_max)
        size_t AddNode(unsigned int num)
        void AddUnaryTerm(size_t i, REAL E0, REAL E1)
        size_t AddPairwiseTerm(size_t i, size_t j, REAL E00, REAL E01, REAL E10, REAL E11)

        int GetLabel(size_t i)
        void Solve()
        void InitalizeBy(int* initial_solution)
        void InitializeByUnary()

cdef extern from "Fast_PD.h":
    cdef cppclass CV_Fast_PD:
        CV_Fast_PD(int numpoints, int numlabels, Real* lcosts,
                   int numpairs, int *pairs,
                   Real* dist, int max_iters,
                   Real* wcostss)
        void run()
        int GetLabel(int n)
        
cdef void add_pairwise_constraint(float[:,::1] pairwise,
                            float cost,
                            float LARGE_CONSTANT = 1e7) nogil: 
    cdef int n_labels = pairwise.shape[0]
    cdef int i,j
    for i in range(n_labels):
        for j in range(n_labels):
            if i == j:
               pairwise[i,j] = 0
            else:
               pairwise[i,j] = cost

    pairwise[3,4] = pairwise[3,4] = LARGE_CONSTANT

def get_binary_mask_fast(int[:, :, ::1] current_labeling,
                         int[:,:,:,::1] masks):
    cdef int n_labels, w, h,d
    cdef int l,x,y,z
    n_labels = masks.shape[0]
    d = masks.shape[1]
    h = masks.shape[2]
    w = masks.shape[3]    
    for z in range(d):
            for y in range(h):
                for x in range(w):
                    l = current_labeling[z,y,x]
                    masks[l, z, y, x] = 1

@cython.boundscheck(False)
@cython.wraparound(False)
def get_feature(float[:,:,::1] padded_vol, float[:, ::1] features, int rad, int n_threads = 8):
    cdef int d,h,w,size,local_index
    d = padded_vol.shape[0]
    h = padded_vol.shape[1]
    w = padded_vol.shape[2]    
    size = w * h * d    
#    cdef float[:, ::1] features = np.empty((size, (rad*2+1)**3),dtype=np.float32)

    cdef unsigned int index = 0
    cdef float value
    cdef int x,y,z,xx,yy,zz
    cdef int patch_size = 2*rad + 1
#    for z in prange(rad, d, nogil=True, num_threads=n_threads):
    for z in range(rad, d-rad):
        print z
        for y in range(rad, h-rad):
            for x in range(rad, w-rad):
                local_index = 0
                for zz in range(-rad, rad+1):
                    for yy in range(-rad, rad+1):
                        for xx in range(-rad, rad+1):
                            features[index, local_index] = padded_vol[z + zz, y + yy, x + xx]
                            local_index += 1

                index += 1

    return features

@cython.boundscheck(False)
@cython.wraparound(False)
def get_edge_cost(short[:,:,::1] vol_data,
                  int w, int h, int d,
                  double beta, double pair_coeff,
                  unsigned int[:,::1] pair_index,
                  float[:] pair_costs,
                  int n_threads = 8):

    cdef int x,y,z
    cdef double value,value_n,cost
    cdef unsigned int x_index, y_index, z_index, index, index_n

    for z in prange(d, nogil=True, num_threads=n_threads):
        printf("%d\n", z)
        for y in range(h):
            for x in range(w):
    
                index = x + y * w + z * h * w
                value = <double>vol_data[z,y,x]
    
                if x != w-1:
                    index_n = index + 1
                    value_n = <double>vol_data[z, y, x+1]
                    cost = pair_coeff * exp(-(value-value_n)**2 * beta)
                    x_index = x + y * (w-1) + z * (w-1)*h
                    pair_index[x_index,0] = index
                    pair_index[x_index,1] = index_n                    
                    pair_costs[x_index] = cost
                    
    
                if y != h-1:    
                    index_n = index + w
                    value_n = <double>vol_data[z, y+1, x]
                    cost = pair_coeff * exp(-(value-value_n)**2 * beta)
                    y_index = (w-1)*h*d + y + x * (h-1) + z * (h-1)*w                
                    pair_index[y_index,0] = index
                    pair_index[y_index,1] = index_n                    
                    pair_costs[y_index] = cost
                    
                if z != d-1:
                    index_n = index + w*h
                    value_n = <double>vol_data[z+1, y, x]
                    cost = pair_coeff * exp(-(value-value_n)**2 * beta)
                    z_index = (w - 1) * h * d + (h-1)*w*d+ x + y * w + z * w * h
                    pair_index[z_index,0] = index
                    pair_index[z_index,1] = index_n
                    pair_costs[z_index] = cost                    
    
                
def fusion_move(int[:] current,
                int[:] proposal,
                float[:,::1] unary_cost,
                float[:,:,::1] pair_costs,
                unsigned int[:, ::1] pair_index):

    cdef unsigned int n_node = current.shape[0]
    cdef unsigned int n_edge = pair_costs.shape[0]    
    cdef unsigned int i,l
    cdef int[:] labeling = np.zeros(n_node, np.int32)
    cdef float e00, e01, e10, e11
    cdef float energy = 0
    cdef QPBO[float]* q = new QPBO[float](n_node, n_edge)
    cdef unsigned int n_sup = 0    
    q.AddNode(n_node)
    
    for i in range(n_node):
        q.AddUnaryTerm(i, unary_cost[i, current[i]], unary_cost[i,proposal[i]])
        
    for i in range(n_edge):
        e00 = pair_costs[i, current[pair_index[i,0]], current[pair_index[i,1]]]
        e01 = pair_costs[i, current[pair_index[i,0]], proposal[pair_index[i,1]]]
        e10 = pair_costs[i, proposal[pair_index[i,0]], current[pair_index[i,1]]]
        e11 = pair_costs[i, proposal[pair_index[i,0]], proposal[pair_index[i,1]]]
        q.AddPairwiseTerm(pair_index[i,0], pair_index[i,1], e00, e01, e10, e11)
        if e00 + e11 > e10 + e01:
            n_sup += 1

    q.Solve()
#    q.ComputeWeakPersistencies()
    
    for i in range(n_node):
        l = q.GetLabel(i)
        if l == 1:
            labeling[i] = proposal[i]
        else:
            labeling[i] = current[i]

    del q
    for i in range(n_node):
        energy += unary_cost[i, labeling[i]]
        
    for i in range(n_edge):
        energy += pair_costs[i, labeling[pair_index[i,0]], labeling[pair_index[i,1]]]
    
    return labeling, energy, n_sup

def fastpd(float[:,::1] unary_cost,
           int[:, ::1] pair_index           
           float[:,::1] D,
           float[:] pair_costs,
           int max_iter = 10
           ):

    cdef int n_node = current.shape[0]
    cdef int n_edge = pair_costs.shape[0]
    cdef int n_labels = unary_cost.shape[1]
    cdef int i
    cdef int[:] labeling = np.zeros(n_node, np.int32)
    CV_Fast_PD fastpd = CV_Fast_PD(n_node, n_labels, unary_cost, n_edge, pair_index, D, max_iter, pair_costs)
    fastpd.run()

    for i in range(n_node):
        labeling[i] = fastpd.GetLabel(i)

    return labeling
    
#     cdef float e00, e01, e10, e11
#     cdef float energy = 0
#     cdef QPBO[float]* q = new QPBO[float](n_node, n_edge)
#     cdef unsigned int n_sup = 0    
#     q.AddNode(n_node)
    
#     for i in range(n_node):
#         q.AddUnaryTerm(i, unary_cost[i, current[i]], unary_cost[i,proposal[i]])
        
#     for i in range(n_edge):
#         e00 = pair_costs[i, current[pair_index[i,0]], current[pair_index[i,1]]]
#         e01 = pair_costs[i, current[pair_index[i,0]], proposal[pair_index[i,1]]]
#         e10 = pair_costs[i, proposal[pair_index[i,0]], current[pair_index[i,1]]]
#         e11 = pair_costs[i, proposal[pair_index[i,0]], proposal[pair_index[i,1]]]
#         q.AddPairwiseTerm(pair_index[i,0], pair_index[i,1], e00, e01, e10, e11)
#         if e00 + e11 > e10 + e01:
#             n_sup += 1

#     q.Solve()
# #    q.ComputeWeakPersistencies()
    
#     for i in range(n_node):
#         l = q.GetLabel(i)
#         if l == 1:
#             labeling[i] = proposal[i]
#         else:
#             labeling[i] = current[i]

#     del q
#     for i in range(n_node):
#         energy += unary_cost[i, labeling[i]]
        
#     for i in range(n_edge):
#         energy += pair_costs[i, labeling[pair_index[i,0]], labeling[pair_index[i,1]]]
    
#     return labeling, energy, n_sup

def fusion_move_lsa(int[:] current,
                int[:] proposal,
                float[:,::1] unary_cost,
                float[:,:,::1] pair_costs,
                unsigned int[:, ::1] pair_index):

    cdef unsigned int n_node = current.shape[0]
    cdef unsigned int n_edge = pair_costs.shape[0]    
    cdef unsigned int i,l
    cdef int[:] labeling = np.zeros(n_node, np.int32)
    cdef float e00, e01, e10, e11
    cdef float energy = 0

    cdef LSA[float]* q = new LSA[float](n_node, n_edge)
    q.AddNode(n_node)
    
    for i in range(n_node):
        q.AddUnaryTerm(i, unary_cost[i, current[i]], unary_cost[i,proposal[i]])
        
    for i in range(n_edge):
        e00 = pair_costs[i, current[pair_index[i,0]], current[pair_index[i,1]]]
        e01 = pair_costs[i, current[pair_index[i,0]], proposal[pair_index[i,1]]]
        e10 = pair_costs[i, proposal[pair_index[i,0]], current[pair_index[i,1]]]
        e11 = pair_costs[i, proposal[pair_index[i,0]], proposal[pair_index[i,1]]]
        q.AddPairwiseTerm(pair_index[i,0], pair_index[i,1], e00, e01, e10, e11)

    q.Solve()
#    q.ComputeWeakPersistencies()
    
    for i in range(n_node):
        l = q.GetLabel(i)
        if l == 1:
            labeling[i] = proposal[i]
        else:
            labeling[i] = current[i]

    del q
    for i in range(n_node):
        energy += unary_cost[i, labeling[i]]
        
    for i in range(n_edge):
        energy += pair_costs[i, labeling[pair_index[i,0]], labeling[pair_index[i,1]]]
    
    return labeling, energy

    
                         
    

