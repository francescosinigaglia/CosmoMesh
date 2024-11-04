import numpy as np
import numba as nb
#import emcee
from numba import njit, prange, jit, typed, types
from numba.typed import List
#import corner
#from scipy.spatial import KDTree
import matplotlib.pyplot as plt
#from numba_kdtree import KDTree

import time

# **********************************************
# **********************************************
# **********************************************
# Input parameters

mode = 'random_test' # 'random_test', 'particle_test', 'true'

# Input files
dm_filename =  'run/deltaBOXOM0.315OL0.685G360V2000.0_ALPTrs6.000z0.500.dat'#'DensityDM.z2_0.V500G256.zspace.losz.dat'
tweb_filename = 'run/TwebDelta_OM0.315OL0.685G360V2000.0z0.500.dat'
counts_filename = '/srv/astro/projects/cosmo3d/desi/SecondGenMocks/AbacusHOD/LRG/z0.500/AbacusSummit_base_c000_ph000/total_ncounts.npy'

# General
lbox = 2000
ngrid = 360 

redshift = 0.5

# Parameters related to subgrid physics
collfrac = 0.01 # in fraction, i.e 0<collfrac<1
satmaxdist = 3. # in Mpc/h

# Parameters related to MCMC optimization 
fit = False
numfreepars = 2
cwtype = 4
numit = 1000
thinning = 15
nthreads = 1 # specific to emcee - be careful not to clash with the general numba parallelization

# Bias parameters
apar1 = 100.
alphapar1 = 1.5
apar2 = 100.
alphapar2 = 1.5
apar3 = 100.
alphapar3 = 1.5
apar4 = 100.
alphapar4 = 1.5

# Velocity model
norm = 1.
exp = 1.

# Set random seeds
np.random.seed(123456)

# **********************************************
# **********************************************
# **********************************************
def fftr2c(arr):
    arr = np.fft.rfftn(arr, norm='ortho')

    return arr

def fftc2r(arr):
    arr = np.fft.irfftn(arr, norm='ortho')

    return arr

# **********************************************
def measure_spectrum(signal):

    nbin = round(ngrid/2)
    
    fsignal = np.fft.fftn(signal) #np.fft.fftn(signal)

    kmax = np.pi * ngrid / lbox #np.sqrt(k_squared(L,nc,nc/2,nc/2,nc/2))
    dk = kmax/nbin  # Bin width

    nmode = np.zeros((nbin))
    kmode = np.zeros((nbin))
    power = np.zeros((nbin))

    kmode, power, nmode = get_power(fsignal, nbin, kmax, dk, kmode, power, nmode)
    
    return kmode[1:], power[1:]

# **********************************************                                                                                    
def cross_spectrum(signal1, signal2):

    nbin = round(ngrid/2)

    fsignal1 = np.fft.fftn(signal1) #np.fft.fftn(signal)                                             
    fsignal2 = np.fft.fftn(signal2) #np.fft.fftn(signal)                                             

    kmax = np.pi * ngrid / lbox #np.sqrt(k_squared(L,nc,nc/2,nc/2,nc/2))            
    dk = kmax/nbin  # Bin width                                                                                                    

    nmode = np.zeros((nbin))
    kmode = np.zeros((nbin))
    power = np.zeros((nbin))

    kmode, power, nmode = get_cross_power(fsignal1, fsignal2, nbin, kmax, dk, kmode, power, nmode)

    return kmode[1:], power[1:]

# **********************************************                                                                                        
def compute_cross_correlation_coefficient(cross, power1,power2):
    ck = cross/(np.sqrt(power1*power2))
    return ck

# **********************************************                                                                                         
@njit(parallel=False, cache=True, fastmath=True)
def get_power(fsignal, Nbin, kmax, dk, kmode, power, nmode):
    
    for i in prange(ngrid):
        for j in range(ngrid):
            for k in range(ngrid):
                ktot = np.sqrt(k_squared_nohermite(lbox,ngrid,i,j,k))
                if ktot <= kmax:
                    nbin = int(ktot/dk-0.5)
                    akl = fsignal.real[i,j,k]
                    bkl = fsignal.imag[i,j,k]
                    kmode[nbin]+=ktot
                    power[nbin]+=(akl*akl+bkl*bkl)
                    nmode[nbin]+=1

    for m in prange(Nbin):
        if(nmode[m]>0):
            kmode[m]/=nmode[m]
            power[m]/=nmode[m]

    power = power / (ngrid/2)**3

    return kmode, power, nmode

# **********************************************                                                                            
@njit(parallel=False, cache=True, fastmath=True)
def get_cross_power(fsignal1, fsignal2, Nbin, kmax, dk, kmode, power, nmode):

    for i in prange(ngrid):
        for j in range(ngrid):
            for k in range(ngrid):
                ktot = np.sqrt(k_squared_nohermite(lbox,ngrid,i,j,k))
                if ktot <= kmax:
                    nbin = int(ktot/dk-0.5)
                    akl1 = fsignal1.real[i,j,k]
                    bkl1 = fsignal1.imag[i,j,k]
                    akl2 = fsignal2.real[i,j,k]
                    bkl2 = fsignal2.imag[i,j,k]
                    kmode[nbin]+=ktot
                    power[nbin]+=(akl1*akl2+bkl1*bkl2)
                    nmode[nbin]+=1

    for m in prange(Nbin):
        if(nmode[m]>0):
            kmode[m]/=nmode[m]
            power[m]/=nmode[m]

    power = power / (ngrid/2)**3

    return kmode, power, nmode
  
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
@njit(parallel=False, cache=True, fastmath=True)
def trilininterp(xx, yy, zz, arrin, lbox, ngrid):

    lcell = lbox/ngrid

    #xx = posx
    #yy = posy
    #zz = posz
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

    wtot = wxc*wyc*wzc + wxl*wyc*wzc + wxc*wyl*wzc + wxc*wyc*wzl + wxl*wyl*wzc + wxl*wyc*wzl + wxc*wyl*wzl + wxl*wyl*wzl

    out = 0.

    out += arrin[indxc,indyc,indzc] * wxc*wyc*wzc
    out += arrin[indxl,indyc,indzc] * wxl*wyc*wzc
    out += arrin[indxc,indyl,indzc] * wxc*wyl*wzc
    out += arrin[indxc,indyc,indzl] * wxc*wyc*wzl
    out += arrin[indxl,indyl,indzc] * wxl*wyl*wzc
    out += arrin[indxc,indyl,indzl] * wxc*wyl*wzl
    out += arrin[indxl,indyc,indzl] * wxl*wyc*wzl
    out += arrin[indxl,indyl,indzl] * wxl*wyl*wzl

    #print(wtot)
    #out /= wtot

    return wtot
      
# **********************************************
@njit(parallel=True, cache=True, fastmath=True)
def divide_by_k2(delta,ngrid, lbox):
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(round(ngrid/2.)): 
                k2 = k_squared(lbox,ngrid,ii,jj,kk) 
                if k2>0.:
                    delta[ii,jj,kk] /= -k_squared(lbox,ngrid,ii,jj,kk) 

    return delta
    

# **********************************************
#@njit(parallel=True, cache=True)
def poisson_solver(delta, ngrid, lbox, fastmath=True):

    delta = fftr2c(delta)

    delta = divide_by_k2(delta, ngrid, lbox)
    #for ii in range(delta.shape[0]):
    #    for jj in range(delta.shape[1]):
    #        for kk in range(delta.shape[2]):               
    #            delta[ii,jj,kk] /= k_squared(lbox,ngrid,ii,jj,kk) 
    delta[0,0,0] = 0.

    delta = fftc2r(delta)

    return delta

# **********************************************
@njit(cache=True, fastmath=True)
def k_squared(lbox,ngrid,ii,jj,kk):
    
      kfac = 2.0*np.pi/lbox

      if ii <= ngrid/2:
        kx = kfac*ii
      else:
        kx = -kfac*(ngrid-ii)
      
      if jj <= ngrid/2:
        ky = kfac*jj
      else:
        ky = -kfac*(ngrid-jj)
      
      #if kk <= nc/2:
      kz = kfac*kk
      #else:
      #  kz = -kfac*np.float64(nc-k)
      
      k2 = kx**2+ky**2+kz**2

      return k2

@njit(cache=True, fastmath=True)
def k_squared_nohermite(lbox,ngrid,ii,jj,kk):

      kfac = 2.0*np.pi/lbox

      if ii <= ngrid/2:
        kx = kfac*ii
      else:
        kx = -kfac*(ngrid-ii)

      if jj <= ngrid/2:
        ky = kfac*jj
      else:
        ky = -kfac*(ngrid-jj)

      if kk <= ngrid/2:
          kz = kfac*kk
      else:
          kz = -kfac*(ngrid-kk)                                                                                                           

      k2 = kx**2+ky**2+kz**2

      return k2

# **********************************************
@njit(parallel=True, cache=True, fastmath=True)
def gradfindiff(lbox,ngrid,arr,dim):

    fac = ngrid/(2*lbox)

    outarr = arr.copy()

    for xx in prange(ngrid):
        for yy in range(ngrid):
            for zz in range(ngrid):

                xdummy = np.array([xx,xx,xx,xx])
                ydummy = np.array([yy,yy,yy,yy])
                zdummy = np.array([zz,zz,zz,zz])
                xxr = xdummy[0]
                xxrr = xdummy[1]
                xxl = xdummy[2]
                xxll = xdummy[3]
                yyr = ydummy[0]
                yyrr = ydummy[1]
                yyl = ydummy[2]
                yyll = ydummy[3]
                zzr = zdummy[0]
                zzrr = zdummy[1]
                zzl = zdummy[2]
                zzll = zdummy[3]

                # Periodic BCs
                if dim == 1:
                    xxl = xx - 1
                    xxll = xx - 2
                    xxr = xx + 1
                    xxrr = xx + 2
                    
                    if xxl<0:
                        xxl += ngrid
                    if xxl>=ngrid:
                        xxl -= ngrid
                    
                    if xxll<0:
                        xxll += ngrid
                    if xxll>=ngrid:
                        xxll -= ngrid
                    
                    if xxr<0:
                        xxr += ngrid
                    if xxr>=ngrid:
                        xxr -= ngrid

                    if xxrr<0:
                        xxrr += ngrid
                    if xxrr>=ngrid:
                        xxrr -= ngrid


                elif dim == 2:
                    
                    yyl = yy - 1
                    yyll = yy - 2
                    yyr = yy + 1
                    yyrr = yy + 2
                    
                    if yyl<0:
                        yyl += ngrid
                    if yyl>=ngrid:
                        yyl -= ngrid
                    
                    if yyll<0:
                        yyll += ngrid
                    if yyll>=ngrid:
                        yyll -= ngrid
                    
                    if yyr<0:
                        yyr += ngrid
                    if yyr>=ngrid:
                        yyr -= ngrid

                    if yyrr<0:
                        yyrr += ngrid
                    if yyrr>=ngrid:
                        yyrr -= ngrid


                elif dim == 3:
                    
                    zzl = zz - 1
                    zzll = zz - 2
                    zzr = zz + 1
                    zzrr = zz + 2
                    
                    if zzl<0:
                        zzl += ngrid
                    if zzl>=ngrid:
                        zzl -= ngrid
                    
                    if zzll<0:
                        zzll += ngrid
                    if zzll>=ngrid:
                        zzll -= ngrid
                    
                    if zzr<0:
                        zzr += ngrid
                    if zzr>=ngrid:
                        zzr -= ngrid

                    if zzrr<0:
                        zzrr += ngrid
                    if zzrr>=ngrid:
                        zzrr -= ngrid

                outarr[xx,yy,zz] = -fac*((4.0/3.0)*(arr[xxl,yyl,zzl]-arr[xxr,yyr,zzr])-(1.0/6.0)*(arr[xxll,yyll,zzll]-arr[xxrr,yyrr,zzrr]))

    return outarr

# **********************************************
@njit(parallel=True, cache=True, fastmath=True)
def get_tidal_invariants(arr, ngrid, lbox):

    # Get gradients exploiting simmetry of the tensor, i.e. gradxy=gradyx

    # X DIRECTION
    # 1st deriv
    grad = gradfindiff(lbox,ngrid,arr,1)
    #2nd derivs
    gradxx = gradfindiff(lbox,ngrid,grad,1)
    gradxy = gradfindiff(lbox,ngrid,grad,2)
    gradxz = gradfindiff(lbox,ngrid,grad,3)

    # Y DIRECTION
    # 1st deriv
    grad = gradfindiff(lbox,ngrid,arr,2)
    #2nd derivs
    gradyy = gradfindiff(lbox,ngrid,grad,2)
    gradyz = gradfindiff(lbox,ngrid,grad,3)

    # Y DIRECTION
    # 1st deriv
    grad = gradfindiff(lbox,ngrid,arr,3)
    #2nd derivs
    gradzz = gradfindiff(lbox,ngrid,grad,3)

    #del arr, grad

    lambda1 = np.zeros_like((gradxx))
    lambda2 = np.zeros_like((gradxx))
    lambda3 = np.zeros_like((gradxx))
    tweb = np.zeros_like((gradxx))

    # Compute eigenvalues    
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):
                mat = np.array([[gradxx[ii,jj,kk],gradxy[ii,jj,kk],gradxz[ii,jj,kk]],[gradxy[ii,jj,kk],gradyy[ii,jj,kk],gradyz[ii,jj,kk]],[gradxz[ii,jj,kk],gradyz[ii,jj,kk],gradzz[ii,jj,kk]]])
                eigs = np.linalg.eigvals(mat)
                eigs = np.flip(np.sort(eigs))
                lambda1[ii,jj,kk] = eigs[0]
                lambda2[ii,jj,kk] = eigs[1]
                lambda3[ii,jj,kk] = eigs[2]
                if eigs[0]>=0 and eigs[1]>=0 and eigs[2]>=0:
                    tweb[ii,jj,kk] = 1
                elif eigs[0]>=0 and eigs[1]>=0 and eigs[2]<0:
                    tweb[ii,jj,kk] = 2
                elif eigs[0]>=0 and eigs[1]<0 and eigs[2]<0:
                    tweb[ii,jj,kk] = 3
                elif eigs[0]<0 and eigs[1]<0 and eigs[2]<0:
                    tweb[ii,jj,kk] = 4

    # Now compute invariants
    #del gradxx, gradxy, gradxz, gradyy,gradyz,gradzz
    
    #I1 = lambda1 + lambda2 + lambda3
    #I2 = lambda1 * lambda2 + lambda1 * lambda3 + lambda2 * lambda3
    #I3 = lambda2 * lambda2 * lambda3

    #del lambda1, lambda2, lambda3

    return tweb

# **********************************************
@njit(parallel=True, cache=True, fastmath=True)
def get_overdens(delta):

    mean_delta = np.mean(delta)
    nn = ngrid * ngrid * ngrid
    for ii in prange(nn):
        delta[ii] = delta[ii]/mean_delta - 1.

    return delta

# **********************************************
ARRAY_TYPE = nb.types.Array(dtype=nb.types.float32, ndim=1, layout="C")
@njit(parallel=True, cache=True, fastmath=True)
def collapse_galaxies_subgrid(posx, posy, posz, delta, tweb, counts, lbox, ngrid, collfrac, satmaxdist, vx, vy, vz, norm, exp):

    lcell = lbox/ngrid

    posx = posx.astype(nb.float32)
    posy = posy.astype(nb.float32)
    posz = posz.astype(nb.float32)

    vx = vx.astype(nb.float32)
    vy = vy.astype(nb.float32)
    vz = vz.astype(nb.float32)

    delta = delta.astype(nb.float32)
    tweb = tweb.astype(nb.int32)
    counts = counts.astype(nb.int32)

    lbox = nb.float32(lbox)
    ngrid = nb.int32(ngrid)
    collfrac = nb.float32(collfrac)
    satmaxdist = nb.float32(satmaxdist)

    # First: identify parent haloes per cell 
    num_dm_part = np.zeros((ngrid,ngrid,ngrid), dtype=nb.int32)

    # Initialize arrays of parent haloes coordinates
    posx_parent = nb.typed.List.empty_list(ARRAY_TYPE)
    posy_parent = nb.typed.List.empty_list(ARRAY_TYPE)
    posz_parent = nb.typed.List.empty_list(ARRAY_TYPE)

    # Initialize lists of DM particles indices in cells
    ind_dm = List() 

    # Initialize lists of galaxies coordinates
    posx_new = nb.typed.List.empty_list(ARRAY_TYPE)
    posy_new = nb.typed.List.empty_list(ARRAY_TYPE)
    posz_new = nb.typed.List.empty_list(ARRAY_TYPE)

    # Initialize lists of galaxies velocities
    vx_new = nb.typed.List.empty_list(ARRAY_TYPE)
    vy_new = nb.typed.List.empty_list(ARRAY_TYPE)
    vz_new = nb.typed.List.empty_list(ARRAY_TYPE)

    # Prepare the lists. This loop can be parallelized
    for ii in range(ngrid*ngrid*ngrid):
        
        ind_dm.append(typed.List.empty_list(types.int32))
        
        posx_parent.append(nb.float32([0]))
        posy_parent.append(nb.float32([0]))
        posz_parent.append(nb.float32([0]))

    print("Finished initializing lists")

    # First loop: it stores the indices of DM particles belonging to each cell.
    # Don't parallelize the loop to avoid mis-ordering particles
    print(ngrid**3, len(posx))
    assert len(posx) == ngrid**3
    for aa in range(len(posx)):

        indx = nb.int32(posx[aa] / lcell) 
        if indx >= ngrid: indx -= ngrid
        indy = nb.int32(posy[aa] / lcell) 
        if indy >= ngrid: indy -= ngrid
        indz = nb.int32(posz[aa] / lcell) 
        if indz >= ngrid: indz -= ngrid

        indxyz = nb.int32(indz + ngrid*(indy + ngrid*indx))
        #print(indx, indy, indz, indxyz, aa)
        ind_dm[indxyz].append(nb.int32(aa))
        num_dm_part[indx, indy, indz] += 1
    print("Finished listing DM particles @ each Grid pos.")
    # Second loop, it does what follows:
    # 1) find existing DM particles in each cell
    # 2) assign parent haloes positions based on existing DM particles, and randomly-samples new ones if needed
    # This loop can be parallelized (parallelize the outer loop)
    
    parent_halo_counter = 0
    not_enough_counter = 0
    for ii in range(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):
                
                xmin = ii * lcell
                xmax = (ii+1) * lcell
                ymin = jj * lcell
                ymax = (jj+1) * lcell
                zmin = kk * lcell
                zmax = (kk+1) * lcell

                ind3d = nb.int32(kk + ngrid*(jj + ngrid*ii))
                #if ind3d % 1e4:
                #    print(ind3d, "/", ngrid**3)
                
                # Get indices of DM particles in this cell
                indd = ind_dm[ind3d]
                indd = np.asarray(indd)
                
                num_dm_part[ii,jj,kk] = nb.int32(len(indd))

                # Fetch DM particle positions in the cell
                posxdummy = posx[indd]
                posydummy = posy[indd]
                poszdummy = posz[indd]
                
                # Get parent haloes number counts in the cell
                parent_counts = get_parents_counts(delta[ii,jj,kk],tweb[ii,jj,kk], counts[ii,jj,kk], num_dm_part[ii,jj,kk])
                
                # Assign parent haloes positions
                if parent_counts<= len(posxdummy): # if number parent haloes <= number of DM particles
                
                    # Initialize arrays of parent haloes coordinates
                    posx_parent_tmp = np.empty((parent_counts), dtype=nb.float32)#[np.float64(x) for x in range(0)]
                    posy_parent_tmp = np.empty((parent_counts), dtype=nb.float32)#[np.float64(x) for x in range(0)]
                    posz_parent_tmp = np.empty((parent_counts), dtype=nb.float32)#[np.float64(x) for x in range(0)] 
                    
                    
                    shuffler = np.random.permutation(len(posxdummy))
                    
                    posxdummy = posxdummy[shuffler]
                    posydummy = posydummy[shuffler]
                    poszdummy = poszdummy[shuffler]

                    for pp in range(parent_counts):

                        posx_parent_tmp[pp] = posxdummy[pp]
                        posy_parent_tmp[pp] = posydummy[pp]
                        posz_parent_tmp[pp] = poszdummy[pp]

                
                else: # if number of parent haloes > number DM particles, rarely or never happens, randomly-sample the missing parent haloes.
                    
                    nrandstemp = parent_counts-len(posxdummy)

                    #posx_parent_tmp = np.empty((parent_counts+nrandstemp), dtype=nb.float32)
                    #posy_parent_tmp = np.empty((parent_counts+nrandstemp), dtype=nb.float32)
                    #posz_parent_tmp = np.empty((parent_counts+nrandstemp), dtype=nb.float32)
                    posx_parent_tmp = np.empty((parent_counts), dtype=nb.float32)
                    posy_parent_tmp = np.empty((parent_counts), dtype=nb.float32)
                    posz_parent_tmp = np.empty((parent_counts), dtype=nb.float32)

                    #for pp in range(parent_counts):
                    for pp in range(len(posxdummy)):
                        posx_parent_tmp[pp] = posxdummy[pp]
                        posy_parent_tmp[pp] = posydummy[pp]
                        posz_parent_tmp[pp] = poszdummy[pp]


                    for pp in range(nrandstemp):

                        #qq = pp + parent_counts
                        qq = pp + len(posxdummy)
                        posx_parent_tmp[qq] = nb.float32(np.random.uniform(xmin,xmax))
                        posy_parent_tmp[qq] = nb.float32(np.random.uniform(ymin,ymax))
                        posz_parent_tmp[qq] = nb.float32(np.random.uniform(zmin,zmax))
                    not_enough_counter += len(posx_parent_tmp)
                posx_parent[ind3d] = posx_parent_tmp
                posy_parent[ind3d] = posy_parent_tmp
                posz_parent[ind3d] = posz_parent_tmp
                parent_halo_counter += len(posx_parent_tmp)
    print("Finished assigning positions of parent halos")
    print("There are", parent_halo_counter, "parent halos")
    print(not_enough_counter, "parent halos had to be randomly generated")
    # We have the positions for parent haloes
    # Now let's assign the random positions for centrals
    # We need another loop over the whole mesh
    # This loops does what follows:
    # 1) splits number counts in cell into centrals and satellites given the central/satellite fraction
    # 2) starts assigning centrals and completes the missing with random sample (if needed)
    # 3) collapses centrals towards parent haloes centres
    # 4) around each central, sample satellites (at the moment, random uniform)  
    # This loop can be parallelized (parallelize the outer loop)
    number_centrals = 0
    number_random_centrals = 0
    for ii in range(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):
                
                if counts[ii,jj,kk] == 0: continue
                # First, re-compute the 3D index, to be used later 
                ind3d = nb.int32(kk + ngrid*(jj + ngrid*ii))
                
                if ind3d % 1e5 == 0:
                    print("Iteration", ind3d, "/", ngrid**3, "pct done", 100 * ind3d / ngrid**3, "%")
                
                # Split into centrals and satellites
                centfrac = get_central_fraction(delta[ii,jj,kk], tweb[ii,jj,kk], counts[ii,jj,kk])
                centcounts = nb.int32(np.around(counts[ii,jj,kk] * centfrac))
                satcounts = nb.int32(int(counts[ii,jj,kk] - centcounts)) 
                # For convention, if the counts of centrals is smaller than the number of DM particles, set it to the num of particles (can be changed)
                if centcounts < num_dm_part[ii,jj,kk]:
                    centcounts = nb.int32(num_dm_part[ii,jj,kk])
                    satcounts = max(0, nb.int32(counts[ii,jj,kk] - centcounts))
                
              
                # Compute number of randoms
                nrands = nb.int32(counts[ii,jj,kk] - num_dm_part[ii,jj,kk])

                # Get indices of DM particles in the considered cell
                indd = np.asarray(ind_dm[ind3d])

                # Compute number of randoms per cell
                if nrands <=0:
                    posxdummy = posx[indd]
                    posydummy = posy[indd]
                    poszdummy = posz[indd]
                
                elif nrands > 0:

                    # Sample the missing centrals with random catalogs
                    posxdummytmp = posx[indd]
                    posydummytmp = posy[indd]
                    poszdummytmp = posz[indd]

                    lentmp = len(posxdummytmp) 

                    posxdummy = np.empty((lentmp+nrands), dtype=nb.float32)
                    posydummy = np.empty((lentmp+nrands), dtype=nb.float32)
                    poszdummy = np.empty((lentmp+nrands), dtype=nb.float32)

                    for dd in range(lentmp):
                    
                        posxdummy[dd] = posxdummytmp[dd]
                        posydummy[dd] = posydummytmp[dd]
                        poszdummy[dd] = poszdummytmp[dd]

                    xmin = ii * lcell
                    xmax = (ii+1) * lcell
                    ymin = jj * lcell
                    ymax = (jj+1) * lcell
                    zmin = kk * lcell
                    zmax = (kk+1) * lcell

                    for dd in range(nrands):
                        ee = dd + lentmp
                        
                        posxdummy[ee] = nb.float32(np.random.uniform(xmin,xmax))
                        posydummy[ee] = nb.float32(np.random.uniform(ymin,ymax))
                        poszdummy[ee] = nb.float32(np.random.uniform(zmin,zmax))
                    number_random_centrals += nrands
                number_centrals += len(posxdummy)
                


                # Split the satellites around the centrals
                satprob = np.empty((len(posxdummy)), dtype=nb.float32)
                satsum = np.int32(0)

                for ww in range(len(satprob)-1):
                    if satcounts > 0:
                        numrand = nb.int32(np.random.randint(0, satcounts))
                    else: numrand = 0
                    satprob[ww] = numrand
                    satsum += numrand


                satproblast = nb.int32(satcounts - satsum)
                if satproblast<0:
                    satproblast = nb.int32(0)

                satprob[-1] = satproblast 
                
                # Now loop over centrals:
                # 1) find nearest parent halo
                # 2) collapse the central towards it
                # 3) sample the satellites around each central

                for ll in range(len(posxdummy)):
                    
                    indmin = nb.int32(0)
                    indsubmin = nb.int32(0)
                    dmin = nb.float32(1e6)
                    # Compute distances only with close parent haloes
                    for ltmp in range(-1,2):
                        for mtmp in range(-1,2):
                            for ntmp in range(-1,2):

                                itmp = nb.int32(ii + ltmp)
                                jtmp = nb.int32(jj + mtmp)
                                ktmp = nb.int32(kk + ntmp)

                                # In order to compute distances correctly with BCs,
                                # compute an auxiiary distance so as to refer the 
                                # distance computation to the same "center of mass"
                                dx = nb.float32(0.)
                                dy = nb.float32(0.)
                                dz = nb.float32(0.)

                                # Periodic BCs for indices
                                if itmp<nb.int32(0):
                                    itmp += ngrid
                                    dx = -lbox
                                elif itmp>=ngrid:
                                    itmp -= ngrid
                                    dx = +lbox

                                if jtmp<nb.int32(0):
                                    jtmp += ngrid
                                    dy = -lbox
                                elif jtmp>=ngrid:
                                    jtmp -= ngrid
                                    dy = +lbox

                                if ktmp<nb.int32(0):
                                    ktmp += ngrid
                                    dz = -lbox
                                elif ktmp>=ngrid:
                                    ktmp -= ngrid
                                    dz = +lbox

                                ind3dtmp = nb.int32(ktmp + ngrid*(jtmp + ngrid*itmp))

                                lenpar_tmp = nb.int32(len(posx_parent[ind3dtmp]))

                                # Compute distance
                                for wtmp in range(lenpar_tmp):

                                    xpar = posx_parent[ind3dtmp][wtmp] + dx
                                    ypar = posy_parent[ind3dtmp][wtmp] + dy
                                    zpar = posz_parent[ind3dtmp][wtmp] + dz

                                    dtmp = nb.float32(np.sqrt((posxdummy[ll] - posx_parent[ind3dtmp][wtmp])**2 + (posydummy[ll] - posy_parent[ind3dtmp][wtmp])**2 + (poszdummy[ll] - posz_parent[ind3dtmp][wtmp])**2))
                                    # Update distance
                                    if dtmp<dmin:
                                        dmin = nb.float32(dtmp)
                                        indmin = nb.int32(ind3dtmp)
                                        indsubmin = nb.int32(wtmp)

                    xpar = posx_parent[indmin][indsubmin]
                    ypar = posy_parent[indmin][indsubmin]
                    zpar = posz_parent[indmin][indsubmin]

                    # Subtract the parent halo position to be in its reference frame
                    xtmp = posxdummy[ll] - xpar
                    ytmp = posydummy[ll] - ypar
                    ztmp = poszdummy[ll] - zpar
                    
                    
                    # So, we need to take this into account we considering the coordinates of the parent halo
                    if abs(ztmp) > dmin: 
                        if ztmp <= nb.float32(0):
                            ztmp += lbox
                        else:
                            ztmp -= lbox

                    if abs(xtmp) > dmin: 
                        if xtmp <= nb.float32(0):
                            xtmp += lbox
                        else:
                            xtmp -= lbox
                     
                    # Move the central towards the closest parent halo (the latter is taken as the centre of coordinates (0,0,0) )
                    # The central position may coincide with parent halo position (dist=0), take care of this
                    if dmin > nb.float32(0):
                        phi = nb.float32(np.arccos(ztmp/dmin)) 
                        # Check if phi is != pi/2, polar coordinates have a singularity...
                        if (dmin*np.sin(phi)) != nb.float32(0):
                            theta = nb.float32(np.arccos(xtmp /(dmin*np.sin(phi))))
                        else:
                            theta = nb.float32(np.pi/4) # Arbitrary value...

                        # Define the new distance by multiplying by the collapse fraction 
                        dmin *= collfrac

                        posx_cent_tmp = nb.float32(dmin * np.cos(theta) * np.sin(phi))
                        posy_cent_tmp = nb.float32(dmin * np.sin(theta) * np.sin(phi))
                        posz_cent_tmp = nb.float32(dmin * np.cos(phi))

                        # Now re-sum the coordinates of the parent halo
                        posx_cent_tmp += xpar
                        posy_cent_tmp += ypar
                        posz_cent_tmp += zpar

                        # Periodic BCs for centrals
                        if posx_cent_tmp >= lbox:
                            posx_cent_tmp -= lbox
                        elif posx_cent_tmp < nb.float32(0):
                            posx_cent_tmp += lbox

                        if posy_cent_tmp >= lbox:
                            posy_cent_tmp -= lbox
                        elif posy_cent_tmp < nb.float32(0):
                            posy_cent_tmp += lbox

                        if posz_cent_tmp >= lbox:
                            posz_cent_tmp -= lbox
                        elif posz_cent_tmp < nb.float32(0):
                            posz_cent_tmp += lbox

                        # Append 
                        posx_new.append(nb.float32(posx_cent_tmp))
                        posy_new.append(nb.float32(posy_cent_tmp))
                        posz_new.append(nb.float32(posz_cent_tmp))

                        xtmp = posx_cent_tmp
                        ytmp = posy_cent_tmp
                        ztmp = posz_cent_tmp

                    else:
                        #Append              
                        posx_new.append(nb.float32(xtmp))
                        posy_new.append(nb.float32(ytmp))
                        posz_new.append(nb.float32(ztmp))

                    # Now model velocities: v = v_coh + v_disp
                    #xtmp = xtmp.astype(nb.float32)
                    #ytmp = ytmp.astype(nb.float32)
                    #ztmp = ztmp.astype(nb.float32)

                    vx_coh = trilininterp(xtmp, ytmp, ztmp, vx, lbox, ngrid)
                    vy_coh = trilininterp(xtmp, ytmp, ztmp, vy, lbox, ngrid)
                    vz_coh = trilininterp(xtmp, ytmp, ztmp, vz, lbox, ngrid)
                    
                    vx_disp, vy_disp, vz_disp = sample_velocity_dispersion(delta,tweb, norm, exp)

                    vx_comp = nb.float32(vx_coh) + nb.float32(vx_disp)
                    vy_comp = nb.float32(vy_coh) + nb.float32(vy_disp)
                    vz_comp = nb.float32(vz_coh) + nb.float32(vz_disp)

                    vx_new.append(nb.float32(vx_comp))
                    vy_new.append(nb.float32(vy_comp))
                    vz_new.append(nb.float32(vz_comp))
                    
                    # Now sample satellites around centrals

                    for zz in range(satprob[ll]):
                    
                        # This creates Gaussian distributed points in the sphere using cartesian coordinates
                        #posx_sat_tmp = nb.float32(nb.float32(np.random.normal(0,satmaxdist)) + posx_cent_tmp)
                        #posy_sat_tmp = nb.float32(nb.float32(np.random.normal(0,satmaxdist)) + posy_cent_tmp)
                        #posz_sat_tmp = nb.float32(nb.float32(np.random.normal(0,satmaxdist)) + posy_cent_tmp)

                        # This creates a custom radial distribution in the sphere using polar coordinates
                        # Sample cos(phi) instead of phi to avoid streak around the poles
                        dnew = nb.float32(abs(np.random.normal(1e-3, satmaxdist)))#nb.float32(np.random.uniform(1e-3, satmaxdist))
                        cosphinew = nb.float32(np.random.uniform(-1, 1))
                        phinew = nb.float32(np.arccos(cosphinew))
                        thetanew = nb.float32(np.random.uniform(0, 2*np.pi))

                        posx_sat_tmp = nb.float32(dnew * np.cos(thetanew) * np.sin(phinew) + posx_cent_tmp)
                        posy_sat_tmp = nb.float32(dnew * np.sin(thetanew) * np.sin(phinew) + posy_cent_tmp)
                        posz_sat_tmp = nb.float32(dnew * np.cos(phinew) + posy_cent_tmp)

                    
                        # Periodic BCs for satellites
                        if posx_sat_tmp >= lbox:
                            posx_sat_tmp -= lbox
                        elif posx_sat_tmp < 0:
                            posx_sat_tmp += lbox
                            
                        if posy_sat_tmp >= lbox:
                            posy_sat_tmp -= lbox
                        elif posy_sat_tmp < 0:
                            posy_sat_tmp += lbox
                            
                        if posz_sat_tmp >= lbox:
                            posz_sat_tmp -= lbox
                        elif posz_sat_tmp < 0:
                            posz_sat_tmp += lbox
                    
                        # Append satellites to total position arrays
                        posx_new.append(nb.float32(posx_sat_tmp))
                        posy_new.append(nb.float32(posy_sat_tmp))
                        posz_new.append(nb.float32(posz_sat_tmp))

                        # For the time being, satellites inherit the same velocity as the central
                        # One may want to model a Maxwellian velocity rotation component as well 
                        vx_new.append(nb.float32(vx_comp))
                        vy_new.append(nb.float32(vy_comp))
                        vz_new.append(nb.float32(vz_comp))
                
    print("There are", number_centrals, "central galaxies")
    print("of which", number_random_centrals, "had to be randomly generated")
    # Convert lists to arrays
    posx_new = np.asarray(posx_new)
    posy_new = np.asarray(posy_new)
    posz_new = np.asarray(posz_new)               

    return posx_new, posy_new, posz_new

# **********************************************
# SUBGRID PHYSICS
@njit(parallel=False, cache=True, fastmath=True)
def get_parents_counts(deltacell,twebcell, countscell, num_part_dm):
    
    parents_counts = np.int32(np.round(0.3*num_part_dm))

    return parents_counts

@njit(parallel=False, cache=True, fastmath=True)
def get_central_fraction(deltacell,twebcell, countscell):

    central_fraction = np.float32(0.3)

    return central_fraction

@njit(parallel=False, cache=True, fastmath=True)
def sample_velocity_dispersion(deltacell,twebcell, norm, exp):

    sigma = 1.#norm*(1+deltacell)**exp
    vxdisp = nb.float32(np.random.normal(0,sigma))
    vydisp = nb.float32(np.random.normal(0,sigma))
    vzdisp = nb.float32(np.random.normal(0,sigma))

    return nb.float32(vxdisp), nb.float32(vydisp), nb.float32(vzdisp)

"""
@njit(parallel=False, cache=True, fastmath=True)
def project_vector(vx,vy, vz, phi, theta):

    vxnew = vx

    return vxdisp, vydisp, vzdisp
"""
    
# **********************************************
# **********************************************
@njit(parallel=True, cache=True, fastmath=True)
def log_likelihood(theta, pkref, kkref, lbox, ngrid):

    # Assign y to a global variable (instead of passing qso as argument) - allows to save a lot of time! 
    y = counts_ref
    
    # Define model parameters: set the same parameters everywhere, here, in the prior, and in the main body 
    if cwtype>0:
        aa, alpha = theta
    else:
        aa1, alpha1, aa2, alpha2, aa3, alpha3, aa4, alpha4 = theta
    
    # Prepare a dummy model array, to be filled later
    model = np.zeros((ngrid,ngrid,ngrid))

    # Compute the model
    # Optionally, restrict the fit only to a given cosmic web type

    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                if cwtype >0:
                    if tweb[ii,jj,kk]==cwtype:
                        model[ii,jj,kk] = apply_bias_mcmc(delta[ii,jj,kk], aa, alpha)
                else:
                    if tweb[ii,jj,kk]==1:
                        model[ii,jj,kk] = apply_bias_mcmc(delta[ii,jj,kk], aa1, alpha1)
                    if tweb[ii,jj,kk]==2:
                        model[ii,jj,kk] = apply_bias_mcmc(delta[ii,jj,kk], aa2, alpha2)
                    if tweb[ii,jj,kk]==3:
                        model[ii,jj,kk] = apply_bias_mcmc(delta[ii,jj,kk], aa3, alpha3)
                    if tweb[ii,jj,kk]==4:
                        model[ii,jj,kk] = apply_bias_mcmc(delta[ii,jj,kk], aa4, alpha4)

    
    # In what follows, posx, posy and posz are global variables
    posx_new, posy_new, posz_new = collapse_galaxies_subgrid(posx, posy, posz, delta, tweb, model, lbox, ngrid, collfrac, satmaxdist, norm, exp)
    
    # Get new counts field after collapse
    model = get_cic(posx_new, posy_new, posz_new, lbox, ngrid)
    
    # Restrict to one CW type if requested
    if cwtype > 0:
        y = y[np.where(tweb==cwtype)]
        model = model[np.where(tweb==cwtype)]

    # Compute PDFs for fitting
    bins = np.linspace(np.amin(y), np.amax(y), num=30)
    pdf_model, binedg = np.histogram(model, bins=bins)
    pdf_truth, binedg =	np.histogram(y, bins=bins)

    return -0.5 * np.sum((pdf_model-pdf_truth)**2)
    
# **********************************************
@njit(parallel=False, cache=True, fastmath=True)
def log_prior(theta):
    
    # Define model parameters: set the same parameters everywhere, here, in the likelihood, and in the main body
    if cwtype>0:
        aa, alpha = theta
    else:
        aa1, alpha1, aa2, alpha2, aa3, alpha3, aa4, alpha4 = theta

    # Set uniform priors
    if cwtype>0:
        if 0. < aa < 2 and 0.5<alpha<3.: #and 0. < b < 10.:
            return 0.0
        return -np.inf
    
    else: 
        if 0. < aa1 < 2 and 0.5<alpha1<3. and 0. < aa2 < 2 and 0.5<alpha2<3. and 0. < aa3 < 2 and 0.5<alpha3<3. and 0. < aa4 < 2 and 0.5<alpha4<3.: #and 0. < b < 10.:
            return 0.0
        return -np.inf
    #return 0.0

@njit(parallel=False, cache=True, fastmath=True)
def log_probability(theta, pkref, kkref, lbox, ngrid):

    lp = log_prior(theta)
    ll = log_likelihood(theta, pkref, kkref, ngrid)
    if not np.isfinite(lp) or np.isnan(ll)==True:
        return -np.inf 
    return lp + log_likelihood(theta, pkref, kkref, lbox, ngrid)

# **********************************************
@njit(parallel=True, cache=True, fastmath=True)
def apply_calibrated_bias(delta,tweb):
    
    counts = np.zeros((ngrid,ngrid,ngrid))

    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                if tweb[ii,jj,kk] == 1:
                    aa = apar1
                    alpha = alphapar1
                elif tweb[ii,jj,kk] == 2:
                    aa = apar2
                    alpha = alphapar2
                elif tweb[ii,jj,kk] == 3:
                    aa = apar3
                    alpha = alphapar3
                elif tweb[ii,jj,kk] == 4:
                    aa = apar4
                    alpha = alphapar4

                counts[ii,jj,kk] = np.int32(np.around(aa * (1 + delta[ii,jj,kk])**alpha))

    return counts

# **********************************************
  
@njit(parallel=False, cache=True, fastmath=True)
def apply_bias_mcmc(deltacell, aa, alpha):
    
    countscell = aa * (1 + deltacell)**alpha
        
    return countscell
if __name__ == '__main__':
    # **********************************************
    # **********************************************
    # **********************************************
    # Compute useful quantities

    lcell = lbox/ngrid

    aa = 1./(1.+redshift)
    HH = 100.

    # Input data
    # Dark matter field
    delta = np.fromfile(open(dm_filename, 'r'), dtype=np.float32)
    delta = get_overdens(delta)
    delta = np.reshape(delta, (ngrid,ngrid,ngrid))

    # Real-space tweb
    tweb = np.fromfile(open(tweb_filename, 'r'), dtype=np.float32)
    tweb = np.reshape(tweb, (ngrid,ngrid,ngrid))

    # Reference galaxy number counts field
    #counts_ref = np.fromfile(open(counts_filename, 'r'), dtype=np.float32)
    counts_ref = np.load(counts_filename)
    print(counts_ref.sum())

    axes = ['x', 'y', 'z']
    part_saved = []
    for i in range(3):
        filename = f"run/PSI{axes[i]}z0.500.dat"
        displ = np.fromfile(open(filename, 'r'), dtype=np.float32)
        part_saved.append(np.fromfile(open(f"run/pos{axes[i]}.dat", 'r'), dtype=np.float32))
        part_saved[i] += displ + lbox
        part_saved[i] %= lbox
        
    posx, posy, posz = part_saved

    # Get delta and T-web classification, they will be casted as 3D arrays
    vxfield = delta.copy()
    vyfield = delta.copy()
    vzfield = delta.copy()



    if fit == True:

        kkref, pkref = measure_spectrum(counts_ref)

        # Enter MCMC
        """
        print('Entering MCMC ...')
        pars = [0.15, 2.03, 0.37, 0.25]

        pos = pars + 1e-4 * np.random.randn(32, numfreepars)
        nwalkers, ndim = pos.shape

        
        with Pool(processes=nthreads) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(pkref, kkref, lbox, ngrid), pool=pool)
            sampler.run_mcmc(pos, numit, progress=True)
        
        # Get autocorrelation time of the chain
        autocorr = sampler.get_autocorr_time(tol=0)
        print('Autocorrelation lenght: ', autocorr)

        # Process the samples
        flat_samples = sampler.get_chain(discard=int(3*autocorr), thin=thinning, flat=True)

        # Save samples
        np.save('samples.npy', flat_samples)
        """

    else:

        #counts_new = apply_calibrated_bias(delta, tweb)
        counts_new = counts_ref
        print(counts_ref.sum())
        
        
        ti = time.time()
        print("Launching collapse galaxies", flush=True)
        
        posx_new, posy_new, posz_new = collapse_galaxies_subgrid(posx, posy, posz, delta, tweb, counts_new, lbox, ngrid, collfrac, satmaxdist, vxfield, vyfield, vzfield, norm, exp)

        tf = time.time()
        print('Elapsed %3.5f secs for collapse function ...' %(tf-ti), flush=True)
        #np.save("run/make_catalog_out.npy", np.c_[posx_new, posy_new, posz_new])
        import proplot as pplt
        fig, ax = pplt.subplots(nrows=1, ncols=2, share=0)
        posx_new = (np.concatenate(posx_new) + lbox) % lbox
        posy_new = (np.concatenate(posy_new) + lbox) % lbox
        posz_new = (np.concatenate(posz_new) + lbox) % lbox
        posx_new = (posx_new + lbox) % lbox
        posy_new = (posy_new + lbox) % lbox
        posz_new = (posz_new + lbox) % lbox
        
        populated_mesh = get_cic(posx_new, posy_new, posz_new, lbox, ngrid)
        k, pk_ref = measure_spectrum(populated_mesh / populated_mesh.mean() - 1)
        ax[0].loglog(k, pk_ref)
        k, pk_ref = measure_spectrum(counts_ref / counts_ref.mean() - 1)
        ax[0].loglog(k, pk_ref)
        fig.savefig("plots/make_catalogs.png", dpi=300)
        