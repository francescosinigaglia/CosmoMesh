import numpy as np
import numba as nb
from numba import njit, prange, jit, typed, types

# **********************************************
# **********************************************
# This script reads particle position and performs CIC interpolation on a grid
# **********************************************
# **********************************************
# Input parameters

# Input files
posx_filename = '...'
posy_filename = '...'
posz_filename = '...'

# Output DM field name
out_filename = '...'

# General
lbox = 1000.
ngrid = 256 

compute_delta = True

# **********************************************
@njit(parallel=False, cache=True, fastmath=True)
def get_ngp(posx, posy, posz, lbox, ngrid):

    lcell = lbox/ngrid

    delta = np.zeros((ngrid,ngrid,ngrid))

    for ii in prange(len(posx)):
        xx = posx[ii]
        yy = posy[ii]
        zz = posz[ii]
        indxc = int(xx/lcell)
        indyc = int(yy/lcell)
        indzc = int(zz/lcell)

        if indxc >= ngrid:
            indxc -= ngrid
        if indyc >= ngrid:
            indyc -= ngrid
        if indzc >= ngrid:
            indzc -= ngrid

        delta[indxc,indyc,indzc] += 1


    if compute_delta==True:
        delta = delta/np.mean(delta) - 1.

    return delta


# **********************************************
# **********************************************
# **********************************************
# Compute useful quantities

# Input data
# DM particles positions
posx = np.fromfile(open(posx_filename, 'r'), dtype=np.float32)
posy = np.fromfile(open(posy_filename, 'r'), dtype=np.float32)
posz = np.fromfile(open(posz_filename, 'r'), dtype=np.float32)

delta = get_ngp(posx, posy, posz, lbox, ngrid)

delta.astype('float32').tofile(out_filename)
 
