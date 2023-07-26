import numpy as np
from functools import partial

err = np.array([0.12,0.105,0.23])

def formate_energy(co, h, oh ,eU):
    rxns = []

    rxns.append(h + 0.178 - 0 + eU)             # * -> H*
    rxns.append(1.1757*oh - h - 0.916 - 0)      # CO2 + H* -> HCOO* 
    rxns.append(0.32 - 1.1757*oh + 0.738 + eU)  # HCOO* -> HCOOH
    
    return max(rxns)
    
def h2_energy(co, h, oh ,eU):
    rxns = []
    
    rxns.append(h + 0.178 - 0 + eU)             # CO2 -> H*
    rxns.append(0 - h - 0.178 + eU)             # H* -> H2
    
    return max(rxns)
    
def c1_energy(co, h, oh ,eU):
    rxns = []
    
    rxns.append(0.6014*co + 0.3438*oh + 0.5289 - 0 + eU)        # CO2 -> COOH*
    rxns.append(0.3986*co - 0.3438*oh - 0.309 + eU)             # COOH* -> CO*
    if co + 0.32 - 0 <= 0:                                      # If CO* -> CO(g) is not favorable 
        rxns.append(0.5628*co + 1.6066 + eU)                    # CO* -> COH*
    else:                                                       # If CO* -> CO(g) is favorable 
        rxns.append(-co - 0.32)                                 # CO* -> CO
        rxns.append(1.5628*co + 1.9266 + eU)                    # CO -> COH*
    rxns.append(-0.507*co - 0.712 + eU)                         # CH* -> CH2*
    
    return max(rxns)    

def co_energy(co, h, oh, eU):
    rxns = []
    
    rxns.append(0.6014*co + 0.3438*oh + 0.5289 - 0 + eU)        # CO2 -> COOH*
    rxns.append(0.3986*co - 0.3438*oh - 0.2089 + eU)            # COOH* -> CO* 
    rxns.append(- co - 0.32)                                    # CO* -> CO(g)
    
    return max(rxns)

def activity(co, h, oh, eU):

    coord = np.array([co,h,oh])
    prob_arr = calculate_volume(coord,err,eU,res=10)
    
    formate = formate_energy(co, h, oh, eU)
    h2 = h2_energy(co, h, oh, eU)
    co_g = co_energy(co, h, oh, eU)
    c1 = c1_energy(co, h, oh, eU)
    
    active_arr = np.array([formate,co_g,c1,h2])
    active_arr = np.multiply(active_arr,prob_arr)
    
    return active_arr
       
       
def get_activity(co_arr,h_arr,oh_arr,eU):
    length = len(co_arr)
    eU_arr = np.repeat([eU],length)
    g_max_lst = list(map(activity, co_arr, h_arr, oh_arr, eU_arr))
    
    g_max_by_prod = [[],[],[],[]]
    products = ['formate','co','c1','h2']
    for g_max in g_max_lst:
        for i in range(4):
            g_max_by_prod[i].append(g_max[i])
    return g_max_by_prod

def weight(g):
    return np.exp(-g)  
    
def boltzman_product(g_max_by_prod,tot):
    arr = np.array(g_max_by_prod)
    prod_arr = np.zeros(4)
    for i in range(4):
        ar = arr[i]
        mask = ar <0
        masked_ar = ar[mask]
    
        vfunc = np.vectorize(weight)
        if len(masked_ar) != 0:
            weighted_prod = vfunc(masked_ar)
        else:
            weighted_prod = [0]        
        prod_arr[i] = np.sum(weighted_prod)/tot
    
    return prod_arr
    
def calculate_volume(coord,err,eU,res=100):
    volume = res*res*res
    bc1 = [0.601, 1, -0.832, 1.445+eU] # B.C 1 : 0.601*x + y - 0.832*z + 1.445 + eU = 0
    bc2 = [0, 1, 0, 0.178 + eU] # B.C 2 : y + 0.178 + eU = 0
    bc3 = [0, 0, -1, 0.22 + eU] # B.C 3 : -z + 0.22 + eU = 0
    if coord[0] <= -0.32:
        bc4 = [0.563, 0, 0, 0.857 + eU] # B.C 4i : 0.563*x + 0.857 + eU = 0 at x <= -0.32
    else:
        bc4 = [1.563, 0, 0, 1.222 + eU] # B.C 4ii : 1.563*x + 1.222 + eU = 0 at x > -0.32
    bc5 = [0.507, 1, 0, 0.89] # B.C 5 : 0.507*x + y + 0.89 
    bc6 = [0, -1, 0, -0.178 + eU ] # B.C 6 : -y - 0.178 + eU
    
    
    x_inter, y_inter, z_inter = np.array([coord - err, coord + err]).T
    x_ = np.linspace(x_inter[0],x_inter[1],res)
    y_ = np.linspace(y_inter[0],y_inter[1],res)
    z_ = np.linspace(z_inter[0],z_inter[1],res)
    
    x, y, z = np.meshgrid(x_,y_,z_,indexing='ij')
    grid = np.stack([x.ravel(),y.ravel(),z.ravel()],axis=1)
    
    mask1 = partial(masking,grid=grid,plane=bc1)
    mask2 = partial(masking,grid=grid,plane=bc2)
    mask3 = partial(masking,grid=grid,plane=bc3)
    mask4 = partial(masking,grid=grid,plane=bc4)
    mask5 = partial(masking,grid=grid,plane=bc5)
    mask6 = partial(masking,grid=grid,plane=bc6)

    formate = len(grid[mask1(operator='>') & mask2(operator='<') & mask3(operator='<')])/volume
    h2 = len(grid[mask1(operator='<') & mask6(operator='<') & mask3(operator='<') & mask5(operator='<')])/volume
    co = len(grid[mask1(operator='<') & mask3(operator='<') & mask4(operator='>') & mask5(operator='>')])/volume
    c1 = len(grid[mask1(operator='<') & mask3(operator='<') & mask4(operator='<') & mask5(operator='>')])/volume
    
    return np.array([formate, co, c1 ,h2]) 
    
def masking(grid,plane,operator):
    if operator == '>':
        mask = np.matmul(grid,plane[:3])+plane[3] >= 0
    elif operator == '<':
        mask = np.matmul(grid,plane[:3])+plane[3] <= 0
    else:
        raise ValueError
    return mask