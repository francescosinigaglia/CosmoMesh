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
  
# **********************************************
@njit(parallel=True, cache=True, fastmath=True)
def get_cic(posx, posy, posz, lbox, ngrid):

    lcell = lbox/ngrid

    delta = np.zeros((ngrid,ngrid,ngrid))

    for ii in prange(len(posx)):
        xx = posx[ii]
        yy = posy[ii]
        zz = posz[ii]
        indxc = int(xx/lcell)
        indyc = int(yy/lcell)
        indzc = int(zz/lcell)

        wxc = xx/lcell - indxc
        wyc = yy/lcell - indyc
        wzc = zz/lcell - indzc

        if wxc <=0.5:
            indxl = indxc - 1
            if indxl<0:
                indxl += ngrid
            wxc += 0.5
            wxl = 1 - wxc
        elif wxc >0.5:
            indxl = indxc + 1
            if indxl>=ngrid:
                indxl -= ngrid
            wxl = 1 - wxc

        if wyc <=0.5:
            indyl = indyc - 1
            if indyl<0:
                indyl += ngrid
            wyc += 0.5
            wyl = 1 - wyc
        elif wyc >0.5:
            indyl = indyc + 1
            if indyl>=ngrid:
                indyl -= ngrid
            wyl = 1 - wyc

        if wzc <=0.5:
            indzl = indzc - 1
            if indzl<0:
                indzl += ngrid
            wzc += 0.5
            wzl = 1 - wzc
        elif wzc >0.5:
            indzl = indzc + 1
            if indzl>=0:
                indzl -= ngrid
            wzl = 1 - wzc

        #print(indxc,indyc,indzc)

        delta[indxc,indyc,indzc] += wxc*wyc*wzc
        delta[indxl,indyc,indzc] += wxl*wyc*wzc
        delta[indxc,indyl,indzc] += wxc*wyl*wzc
        delta[indxc,indyc,indzl] += wxc*wyc*wzl
        delta[indxl,indyl,indzc] += wxl*wyl*wzc
        delta[indxc,indyl,indzl] += wxc*wyl*wzl
        delta[indxl,indyc,indzl] += wxl*wyc*wzl
        delta[indxl,indyl,indzl] += wxl*wyl*wzl

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

delta = get_cic(posx, posy, posz, lbox, ngrid)

delta.astype('float32').tofile(out_filename)
 