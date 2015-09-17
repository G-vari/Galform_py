import numpy as np; import time; from utilities_interpolation import *; from input_parameters import *

print "Tabulating cooling quantities"
cooling_t0 = time.time()

def Z_to_Z_FeH(Z):
    ''' input is metallicity Z, NOT in units of Z_sun!'''
    Z_FeH = Z * 56.39 / (1. - 4.474*Z)
    return Z_FeH

def Zdot_to_Z_FeH_dot(Zdot,Z):
    '''Compute dZ_FeH/dt, given Z and dZ/dt in normal (not solar) units'''
    Z_FeH_dot = Zdot * 56.39 * ( 1. - 2. * 4.474*Z)
    return Z_FeH_dot

def X_from_Z_FeH(Z_FeH):
    ''' calculate hydrogen mass fraction of a gas of metallicity Z_FeH'''
    Z_Metals_Solar=0.0189
    Z_Metals  =5.36e-10
    X_Hydrogen_Solar=0.707
    X_Hydrogen=0.778
    Z_FeH_Max=3.16227766

    dXdZ_FeH=(X_Hydrogen_Solar-X_Hydrogen)/(Z_to_Z_FeH(Z_Metals_Solar)-Z_to_Z_FeH(Z_Metals))

    X_Hydrogen_of_Z_Metals=X_Hydrogen+dXdZ_FeH*np.minimum(Z_FeH,np.zeros_like(Z_FeH)+Z_FeH_Max)

    return X_Hydrogen_of_Z_Metals

def Load_Sutherland_Cooling_Grid():
    '''Read in Sutherland & Doptia cooling tables'''

    path = "./Data/"
    file_list = ["mzero.cie",
                 "m-30.cie",
                 "m-20.cie",
                 "m-15.cie",
                 "m-10.cie",
                 "m-05.cie",
                 "msol.cie",
                 "m+05.cie"]

    Z_FeH_grid = np.array([0.0, 0.001, 0.01, 0.0316228, 0.1, 0.316228, 1.0, 3.16228]) # These metallicities are in units of 10^(Fe/H) instead of usual convention
    nZ = len(Z_FeH_grid)

    logT_grid = np.loadtxt(path+file_list[0],unpack=True,usecols=[0])

    T_grid = 10.**logT_grid

    logLam_grid = np.zeros(( len(Z_FeH_grid), len(logT_grid) ))
    ne_grid = np.zeros_like(logLam_grid); nt_grid = np.zeros_like(logLam_grid)

    for n in range(nZ):
        ne_grid[n], nt_grid[n], logLam_grid[n] = np.loadtxt(path+file_list[n],unpack=True,usecols=[1,3,4])

    return Z_FeH_grid, logT_grid, logLam_grid, ne_grid, nt_grid

Z_FeH_grid, logT_grid, logLam_grid, ne_grid, nt_grid = Load_Sutherland_Cooling_Grid()

def Gradient_NonUniformGrid(x,y,z):
    '''Compute partial derivatives at each point on a arbitrary grid using central difference method
    This routine should produce the same output as numpy.gradient() for a regular grid
    x and y are the grid coordinates, z is the function evaluated at each point on the grid'''

    # First compute intervals around each grid point, extrapolating from closest pair of grid points at the grid edges

    dx = x[1:] - x[0:-1]
    dx = np.hstack((dx[0],dx))
    dx = np.hstack((dx,dx[-1]))
    dx = 0.5*(dx[1:] + dx[0:-1])

    dy = y[1:] - y[0:-1]
    dy = np.hstack((dy[0],dy))
    dy = np.hstack((dy,dy[-1]))
    dy = 0.5*(dy[1:] + dy[0:-1])
    
    # Do the same but in 2d for the function intervals

    dzx = z[1:] - z[0:-1]
    dzx = np.vstack((dzx[0],dzx))
    dzx = np.vstack((dzx,dzx[-1]))
    dzx = 0.5*(dzx[1:] + dzx[0:-1])

    dzy = z[:,1:] - z[:,0:-1]
    dzy = np.hstack((dzy[:,0][np.newaxis, :].T,dzy))
    dzy = np.hstack((dzy,dzy[:,-1][np.newaxis, :].T))
    dzy = 0.5*(dzy[:,1:] + dzy[:,0:-1])

    # Create a grid of the x,y coordinates

    dxx, dyy = np.meshgrid(dx,dy,indexing="ij")

    # Compute the partial derivatives using central difference method (as in numpy.gradient() )

    dzdx = dzx / dxx
    dzdy = dzy / dyy

    return dzdx, dzdy


dlogLam_dZFeH_grid, dlogLam_dlogT_grid = Gradient_NonUniformGrid(Z_FeH_grid, logT_grid, logLam_grid)




def Calculate_Cooling_Function(T,Zhot, logLam_grid, logT_grid, Z_FeH_grid, ne_grid, nt_grid):
    ''' Use interpolation to get the cooling function, Lambda(T,Z) in J s^-1 m^3
    Cooling functions are read in from tables originally described in
    Sutherland & Dopita (1993). The cooling function is normalised to unit density
    of hydrogen in cm^3.

    Interpolation scheme is slightly convoluted because it is designed to match Cooling_Time_Standard in cooling.F90
    In that function, interpolation is done in terms of the cooling time, tc_normed, for unit density of hydrogen ions.
    tc_normed \propto T (ne+nt) / Lambda_cool
    1st) Interpolate log(tc_normed) in log(T) --> Interpolate log(Lambda_cool^-1), log(ne+nt) in log(T)
    2nd) Interpolate tc_normed^-1 in Z_FeH --> Interpolate Lambda_cool, (ne+nt)^-1 in Z_FeH

    DO NOT feed any NaNs into this routine, it will crash!'''

    
    nZ = len(Z_FeH_grid)
    n_T = len(logT_grid)
    T_grid = 10.**logT_grid

    if T > T_grid.max or T < T_grid.min:
        if T > T_grid.max():
            #print "Temperature out of bounds (too high) from Sutherland & Dopita grid, setting to endpoint value"
            T = T_grid.max()

        if T < T_grid.min():
            #print "Temperature out of bounds (too low) from Sutherland & Dopita grid, setting to endpoint value"
            T = T_grid.min()

    # Run interpolation in log(Lambda), log(T) and Z to get log(Lambda) for desired Z,T

    Z_FeH = Z_to_Z_FeH(Zhot)
    
    if Z_FeH > Z_FeH_grid.max():
        Z_FeH = Z_FeH_grid.max()

    logT_desired = np.log10(T)

    # 1st perform interpolation of log(Lambda^-1) and log(ne+nt) in log(T)

    loginvLam_grid = - logLam_grid
    loginvLam_broadcast = np.swapaxes(loginvLam_grid,0,1)
    loginvLam_interpT = Linear_Interpolation(logT_grid,loginvLam_broadcast,[logT_desired])
    loginvLam_broadcast = np.swapaxes(loginvLam_interpT,0,1)

    lognepnt_grid = np.log10(ne_grid+nt_grid)

    lognepnt_broadcast = np.swapaxes(lognepnt_grid,0,1)
    lognepnt_interpT = Linear_Interpolation(logT_grid,lognepnt_broadcast,[logT_desired])
    lognepnt_broadcast = np.swapaxes(lognepnt_interpT,0,1)

    # 2nd, perform interpolation of Lambda and (ne+nt)^-1 in Z_FeH

    ind_Z = np.argmin(abs(np.reshape([Z_FeH],(len([Z_FeH]),1))-Z_FeH_grid),axis=1)
    ind_Z_1 = np.copy(ind_Z)
    ind_Z_2 = np.copy(ind_Z_1) +1

    gt = [Z_FeH] > Z_FeH_grid[ind_Z_1]
    ind_Z_2[gt == False] -= 1
    ind_Z_1[gt == False] -= 1

    # Set metallicities outside of the Sutherland & Dopita grid to their endpoint values
    ok = (ind_Z_2 < len(Z_FeH_grid)) & (ind_Z_1 >= 0)
    too_low = ind_Z_1 < 0
    ind_Z_1[too_low] = 0
    too_high = ind_Z_2 >= len(Z_FeH_grid)
    ind_Z_2[too_high] = len(Z_FeH_grid)-1

    Z_1 = Z_FeH_grid[ind_Z_1]; Z_2 = Z_FeH_grid[ind_Z_2]

    fraction = np.zeros_like(Z_1)
    fraction[ok] = (Z_2[ok]-[Z_FeH])/(Z_2[ok]-Z_1[ok])

    Lam_out = np.zeros_like([Z_FeH])
    Lam_1 = np.choose(ind_Z_1,10**(-loginvLam_broadcast))
    Lam_2 = np.choose(ind_Z_2,10**(-loginvLam_broadcast))
    Lam_out = Lam_1 * fraction + Lam_2 * (1-fraction)
    #logLam_out[ok==False] = np.diagonal(logLam_broadcast[ind_Z[ok==False],:])

    invnepnt_out = np.zeros_like([Z_FeH])
    invnepnt_1 = np.choose(ind_Z_1, 1./(10**lognepnt_broadcast))
    invnepnt_2 = np.choose(ind_Z_2, 1./(10**lognepnt_broadcast))
    invnepnt_out = invnepnt_1 * fraction + invnepnt_2 * (1-fraction)
    nepnt_out = 1./invnepnt_out # number density per (unit hydrogen ions per cm^3)

    Lambda_cool = Lam_out # erg s-1 cm^3

    Lambda_cool *= 10**-7 # J s^-1 cm^3 = kg m^2 cm^3 s^-3
    Lambda_cool *= 10**-6 #J s^-1 m^3 = kg s^-3 m^5

    return Lambda_cool[0], nepnt_out[0]

def Calculate_Cooling_Function_Partial_Derivatives(T,Zhot, Lam, dlogLam_dZFeH_grid, dlogLam_dlogT_grid, logT_grid, Z_FeH_grid):
    ''' Use interpolation to get the partial derivates of the cooling function, Lambda(T,Z) in J s^-1 m^3 over K (for w.r.t T)
    Cooling functions are read in from tables originally described in
    Sutherland & Dopita (1993). The cooling function is normalised to unit density
    of hydrogen in cm^3.

    For x = dlogT and x = dZ_FeH:
    Do interpolation of dlog(L^-1)/dx in log T, then interpolation of dL/dx in Z_FeH

    Do not input NaNs into this routine'''

    
    nZ = len(Z_FeH_grid)
    n_T = len(logT_grid)
    T_grid = 10.**logT_grid

    if T > T_grid.max or T < T_grid.min:
        if T > T_grid.max():
            T = T_grid.max()

        if T < T_grid.min():
            T = T_grid.min()

    # Run interpolation in log(Lambda), log(T) and Z to get log(Lambda) for desired Z,T

    Z_FeH = Z_to_Z_FeH(Zhot)
    
    if Z_FeH > Z_FeH_grid.max():
        Z_FeH = Z_FeH_grid.max()

    logT_desired = np.log10(T)

    # 1st perform interpolation of dlog(Lambda^-1)/dx in log(T)

    loginvLam_grid_Z = - dlogLam_dZFeH_grid
    loginvLam_broadcast_Z = np.swapaxes(loginvLam_grid_Z,0,1)
    loginvLam_interpT_Z = Linear_Interpolation(logT_grid,loginvLam_broadcast_Z,[logT_desired])
    loginvLam_broadcast_Z = np.swapaxes(loginvLam_interpT_Z,0,1)

    loginvLam_grid_T = - dlogLam_dlogT_grid
    loginvLam_broadcast_T = np.swapaxes(loginvLam_grid_T,0,1)
    loginvLam_interpT_T = Linear_Interpolation(logT_grid,loginvLam_broadcast_T,[logT_desired])
    loginvLam_broadcast_T = np.swapaxes(loginvLam_interpT_T,0,1)

    # Convert Lambda back to cgs units for consistency with dlog(Lambda)/dx
    Lam_cgs = Lam * 10**13 # erg s-1 cm^3

    Lam_broadcast_T = -loginvLam_broadcast_T * Lam_cgs / T
    Lam_broadcast_Z = -loginvLam_broadcast_Z * Lam_cgs * np.log(10)

    # Convert to SI units
    Lam_broadcast_T *= 10**(-13) # kg s^-3 m^5 K^-1
    Lam_broadcast_Z *= 10**(-13) # kg s^-3 m^5

    # 2nd, perform interpolation of dLambda/dx in Z_FeH

    ind_Z = np.argmin(abs(np.reshape([Z_FeH],(len([Z_FeH]),1))-Z_FeH_grid),axis=1)
    ind_Z_1 = np.copy(ind_Z)
    ind_Z_2 = np.copy(ind_Z_1) +1

    gt = [Z_FeH] > Z_FeH_grid[ind_Z_1]
    ind_Z_2[gt == False] -= 1
    ind_Z_1[gt == False] -= 1

    # Set metallicities outside of the Sutherland & Dopita grid to their endpoint values
    ok = (ind_Z_2 < len(Z_FeH_grid)) & (ind_Z_1 >= 0)
    too_low = ind_Z_1 < 0
    ind_Z_1[too_low] = 0
    too_high = ind_Z_2 >= len(Z_FeH_grid)
    ind_Z_2[too_high] = len(Z_FeH_grid)-1

    Z_1 = Z_FeH_grid[ind_Z_1]; Z_2 = Z_FeH_grid[ind_Z_2]

    fraction = np.zeros_like(Z_1)
    fraction[ok] = (Z_2[ok]-[Z_FeH])/(Z_2[ok]-Z_1[ok])

    Lam_out_Z = np.zeros_like([Z_FeH])
    Lam_1_Z = np.choose(ind_Z_1,Lam_broadcast_Z)
    Lam_2_Z = np.choose(ind_Z_2,Lam_broadcast_Z)
    Lam_out_Z = Lam_1_Z * fraction + Lam_2_Z * (1-fraction)
    
    dLambda_cool_dZ_FeH = Lam_out_Z #J s^-1 m^3 = kg s^-3 m^5

    Lam_out_T = np.zeros_like([Z_FeH])
    Lam_1_T = np.choose(ind_Z_1,Lam_broadcast_T)
    Lam_2_T = np.choose(ind_Z_2,Lam_broadcast_T)
    Lam_out_T = Lam_1_T * fraction + Lam_2_T * (1-fraction)

    dLambda_cool_dT = Lam_out_T #J s^-1 m^3 K^-1 = kg s^-3 m^5 K^-1

    return dLambda_cool_dZ_FeH[0], dLambda_cool_dT[0]

def Calculate_Cooling_Time(M_hot, R_h, V_h, Zhot, logLam_grid, logT_grid, Z_FeH_grid):
    ''' Calculate the characteristic timescale for a halo of hot gas to cool in Gyr,
    assuming a constant density, isothermal density profile. The cooling function
    is taken from Sutherland & Doptita (1993).
    Inputs:
       M_h = Mass of hot gas inside the halo in Msun
       R_h = Virial radius in kpc
       V_h = Circular velocity at the virial radius in km s^-1
       Zhot = Metallicity of the hot gas, in standard units (not in units of Z_sun)
    Ouput:
       t_cool = cooling timescale in Gyr'''

    # constants
    kb = 1.3806488 * 10**-23 # m^2 kg s^-2 K^-1
    M_Atomic=1.66053873e-27 # kg
    Atomic_Mass_Hydrogen=1.00794 # in units of M_Atomic
    Atomic_Mass_Helium=4.002602 # in units of M_Atomic
    fH = 0.778 # primordial hyrdrogen abundance
    fHe = 0.222 # primordial helium abundance
    Msun = 1.98855 * 10**30 #kg
    pc = 3.08567758 * 10**16 # m
    kpc = pc * 10**3
    Gyr = 365.25 * 24 * 3600 * 10**9 # s
    kms = 10**3 # ms^-1

    # Convert metallicity to units of Fe/H
    Z_FeH = Z_to_Z_FeH(Zhot)

    # Calculate hydrogen mass fraction
    X = X_from_Z_FeH(Z_FeH)

    # Calculate mean molecular weight in atomic mass units of a primordial gas
    mu_Primordial=1.0/(2.0*fH/Atomic_Mass_Hydrogen+3.0*fHe/Atomic_Mass_Helium) # Mean atomic weight

    # For an isothermal profile, the temperature of the gas is approximated by the Virial temperature  
    T = 0.5 * mu_Primordial * M_Atomic / kb * (V_h*kms)**2 # Kelvin

    # Get the cooling function

    Lambda_cool, nepnt = Calculate_Cooling_Function(T,Zhot, logLam_grid, logT_grid, Z_FeH_grid, ne_grid, nt_grid) # J s^-1 m^3

    mu_actual = 1./(nepnt) * 1/X

    # Calculate the cooling luminosity per unit mass
    L_cool = 3/(4*np.pi) * X**2 * Lambda_cool * M_hot * Msun / (R_h*kpc)**3 / (Atomic_Mass_Hydrogen * M_Atomic)**2 # J s^-1 kg^-1
    
    # Calculate the internal energy per unit mass
    U = 1.5 * kb * T / (mu_actual * M_Atomic) # m^2 s^-2 = J kg^-1

    # Calculate the cooling timescale
    t_cool = U/L_cool/Gyr
    
    return t_cool

def Calculate_Cooling_Radius_LGalaxies(M_hot, R_h, V_h, Zhot, tdyn, logLam_grid, logT_grid, Z_FeH_grid):
    ''' Calculate the cooling radius following the cooling model from LGalaxies
    Assumed hot gas profile is isothermal (I think).
    The cooling function is taken from Sutherland & Doptita (1993).
    
    Inputs:
       M_hot = Mass of hot gas inside the halo in Msun
       R_h = Virial radius in kpc
       V_h = Circular velocity at the virial radius in km s^-1
       Zhot = Metallicity of the hot gas, in standard units (not in units of Z_sun)
       tdyn = halo dynamical timescale in Gyr
    Ouput:
       r_cool = cooling radius in kpc'''

    # if R_h = 0, set rcool = 0
    if R_h == 0:
        r_cool = 0.0
        return r_cool

    # constants
    kb = 1.3806488 * 10**-23 # m^2 kg s^-2 K^-1
    M_Atomic=1.66053873e-27 # kg
    Atomic_Mass_Hydrogen=1.00794 # in units of M_Atomic
    Atomic_Mass_Helium=4.002602 # in units of M_Atomic
    fH = 0.778 # primordial hyrdrogen abundance
    fHe = 0.222 # primordial helium abundance
    Msun = 1.98855 * 10**30 #kg
    pc = 3.08567758 * 10**16 # m
    kpc = pc * 10**3
    Gyr = 365.25 * 24 * 3600 * 10**9 # s
    kms = 10**3 # ms^-1


    # Convert metallicity to units of Fe/H
    Z_FeH = Z_to_Z_FeH(Zhot)

    # Calculate hydrogen mass fraction
    X = X_from_Z_FeH(Z_FeH)

    # Calculate mean molecular weight in atomic mass units of a primordial gas
    mu_Primordial=1.0/(2.0*fH/Atomic_Mass_Hydrogen+3.0*fHe/Atomic_Mass_Helium) # Mean atomic weight

    # For an isothermal profile, the temperature of the gas is approximated by the Virial temperature  
    T = 0.5 * mu_Primordial * M_Atomic / kb * (V_h*kms)**2 # Kelvin

    # Get the cooling function
    Lambda_cool, nepnt = Calculate_Cooling_Function(T,Zhot, logLam_grid, logT_grid, Z_FeH_grid, ne_grid, nt_grid) # J s^-1 m^3

    # Recompute effective particle weight 
    mu_actual = 1./(nepnt) * 1/X # dimensionless

    r_cool = (tdyn * M_hot * X**2 * Lambda_cool / (6.*np.pi* mu_actual * M_Atomic *  kb * T *R_h*kpc))**0.5 # (Gyr Msun J s^-1 m^3 kg^-1 J^-1 m^-1)**0.5 = (Gyr s^-1 Msun kg^-1 m^2)
    r_cool *= (Gyr * Msun)**0.5 # m
    r_cool *= 1./kpc # kpc

    return r_cool

def Calculate_Cooling_Radius(M_hot, R_h, V_h, Zhot, tform, logLam_grid, logT_grid, Z_FeH_grid):
    ''' Calculate the radius at which characteristic cooling timescale for hot gas is equal to
    time elapsed since the formation of the halo.
    Assumed hot gas profile is density \propto 1/(r^2 +r_core^2) where r_core = 0.1 R_h.
    The cooling function is taken from Sutherland & Doptita (1993).
    
    Inputs:
       M_hot = Mass of hot gas inside the halo in Msun
       R_h = Virial radius in kpc
       V_h = Circular velocity at the virial radius in km s^-1
       Zhot = Metallicity of the hot gas, in standard units (not in units of Z_sun)
       tform = time elapsed since halo formation event in Gyr
    Ouput:
       r_cool = cooling radius in kpc'''

    # if tform = 0 or R_h = 0, no gas has had time to cool and r_cool = 0
    if tform == 0 or R_h == 0:
        r_cool = 0.0
        return r_cool

    # constants
    kb = 1.3806488 * 10**-23 # m^2 kg s^-2 K^-1
    M_Atomic=1.66053873e-27 # kg
    Atomic_Mass_Hydrogen=1.00794 # in units of M_Atomic
    Atomic_Mass_Helium=4.002602 # in units of M_Atomic
    fH = 0.778 # primordial hyrdrogen abundance
    fHe = 0.222 # primordial helium abundance
    Msun = 1.98855 * 10**30 #kg
    pc = 3.08567758 * 10**16 # m
    kpc = pc * 10**3
    Gyr = 365.25 * 24 * 3600 * 10**9 # s
    kms = 10**3 # ms^-1

    # Set core radius
    r_core = R_h * 0.1

    # Convert metallicity to units of Fe/H
    Z_FeH = Z_to_Z_FeH(Zhot)

    # Calculate hydrogen mass fraction
    X = X_from_Z_FeH(Z_FeH)

    # Calculate mean molecular weight in atomic mass units of a primordial gas
    mu_Primordial=1.0/(2.0*fH/Atomic_Mass_Hydrogen+3.0*fHe/Atomic_Mass_Helium) # Mean atomic weight

    # For an isothermal profile, the temperature of the gas is approximated by the Virial temperature  
    T = 0.5 * mu_Primordial * M_Atomic / kb * (V_h*kms)**2 # Kelvin

    # Calculate density profile normalisation
    norm = M_hot*Msun / (4*np.pi*(R_h*kpc  - r_core*kpc*np.arctan(R_h/r_core))) # kg m^-1

    # Get the cooling function
    Lambda_cool, nepnt = Calculate_Cooling_Function(T,Zhot, logLam_grid, logT_grid, Z_FeH_grid, ne_grid, nt_grid) # J s^-1 m^3

    # Recompute effective particle weight 
    mu_actual = 1./(nepnt) * 1/X # dimensionless

    # Calculate cooling factor
    F_cool = 1.5  / (mu_actual * M_Atomic) *  kb * T * (Atomic_Mass_Hydrogen * M_Atomic)**2 / (X**2 * Lambda_cool) / Gyr # kg m^-3 Gyr
    
    if norm*tform/F_cool >= (r_core*kpc)**2:
        r_cool = (norm*tform/F_cool - (r_core*kpc)**2)**0.5 / kpc
    else:
        r_cool = 0.

    return r_cool

def Calculate_Cooling_Radius_Rate(rhalo,rhalo_dot,rcore,rcore_dot,r_cool,mhot,mhot_dot,vhalo,vhalo_dot,Zhot,Zhot_dot):
    ''' Calculate the rate at which the cooling radius (see Calculate_Cooling_Radius()) propagates outwards in a
    halo where the hot gas follows a Beta profile.
    Assumed hot gas profile is density \propto 1/(r^2 +r_core^2) where r_core = 0.1 R_h.
    The cooling function is taken from Sutherland & Doptita (1993).
    
    Inputs:    
       rhalo       = Halo virial radius in kpc
       rhalo_dot   = Rate of change of the halo virial radius in kpc Gyr^-1
       rcore       = core radius for the hot gas profile in kpc
       rcore_dot   = Rate of change of the core radius in kpc Gyr^-1
       r_cool      = cooling radius in kpc
       Lambda_cool = Cooling function in J s^-1 m^3
       nept        = (ne+nt) - used to compute effective particle weight, (assumed to be constant in time)
       mhot        = Hot gas mass in Msun
       mhot_dot    = Rate of change of hot gas mass in Msun Gyr^-1
       vhalo       = Halo circular velocity in kms^-1
       vhalo_dot   = Rate of change of halo circular velocity in kms^-1 Gyr^-1
       Zhot        = Hot gas metallicity (not in solar or FeH units)
       Zhot_dot    = Rate of change of hot gas metallicity

    Ouput:
       r_cool_dot = rate of change of the cooling radius in kpc Gyr^-1'''

    # Convert metallicity to units of Fe/H
    Z_FeH = Z_to_Z_FeH(Zhot)
    Z_FeH_dot = Zdot_to_Z_FeH_dot(Zhot_dot,Zhot)

    # For an isothermal profile, the temperature of the gas is approximated by the Virial temperature  
    T = 0.5 * Constant.mu_Primordial * Constant.M_Atomic / Constant.kb * (vhalo*Constant.kms)**2 # Kelvin

    # Get the cooling function
    Lambda_cool, nepnt = Calculate_Cooling_Function(T,Zhot, logLam_grid, logT_grid, Z_FeH_grid, ne_grid, nt_grid) # J s^-1 m^3

    # Compute rate of change of temperature
    dT_dt = Constant.mu_Primordial * Constant.M_Atomic / Constant.kb * (vhalo*Constant.kms) * (vhalo_dot*Constant.kms) # Kelvin Gyr^-1

    dLambda_dZFeH, dLambda_dT = Calculate_Cooling_Function_Partial_Derivatives(T,Zhot, Lambda_cool, dlogLam_dZFeH_grid, dlogLam_dlogT_grid, logT_grid, Z_FeH_grid)

    # Compute time derivative of the cooling function
    dLambda_dt = dLambda_dT * dT_dt + dLambda_dZFeH * Z_FeH_dot # kg s^-3 m^5 Gyr^-1  

    # Compute F'(r_H,r_core) = r_H - r_core * arctan(r_H/r_core)
    Fprime = rhalo - rcore * np.arctan(rhalo/rcore)

    # Compute dF'/dt = drhalo/dt - drcore/dt * arctan(r_H/r_core)
    Fprime_dot = rhalo_dot - rcore_dot * np.arctan(rhalo/rcore)

    # Compute F'' = r_core**2 + r_cool**2
    Fprime2 = rcore**2 + r_cool**2

    # Compute dF''/dt
    dLambda_m1_dt = - Lambda_cool**(-2.) * dLambda_dt # kg^-1 s^3 m^-5 Gyr^-1
    mhot_m1_dot = - mhot**(-2.) * mhot_dot # Msun^-1 Gyr^-1
    dvhalo_2_dt = 2. * vhalo * vhalo_dot # km^2 s^-2 Gyr^-1

    conversion_factor = Constant.Msun / Constant.km**2 / Constant.kpc**3 * Constant.Gyr

    # Calculate hydrogen mass fraction
    X = X_from_Z_FeH(Z_FeH)
    # Recompute effective particle weight 
    mu_actual = 1./(nepnt) * 1/X # dimensionless

    term1 = conversion_factor * X**2 * mu_actual / (3. * np.pi * (Constant.Atomic_Mass_Hydrogen * Constant.M_Atomic)**2 * Constant.mu_Primordial) * Lambda_cool * mhot / (vhalo**2 * Fprime * Fprime2) # Gyr^-1

    Fprime2_dot = Fprime2 * (term1 - dvhalo_2_dt/vhalo**2 - Fprime_dot/Fprime - dLambda_m1_dt*Lambda_cool - mhot_m1_dot * mhot)

    r_cool_dot = 1 / (2.*r_cool) * ( Fprime2_dot - 2. * rcore * rcore_dot ) # kpc Gyr^-1

    return r_cool_dot

def Tabulate_pIp():
    '''Tabulate p I(p) as a function of p.
    This is used for numerically inverting the freefall time equation to calculate the freefall radius'''
    NP = 100; NZ = 1000
    dz = 1./NZ

    p = 10.**(-2 + 4*(np.arange(NP)-1)  /(NP-1.))
    lgp_tmp = np.log10(p)

    # Trapezium integration

    # z=1 term
    vel2 = 1. - np.log(1.+p)/p
    Sum = 0.5 * 2. / (vel2)**0.5
    
    #z = 1-dz to z = dz terms
    z = 1. - (np.arange(NZ-1.)+1)*dz
    x = 1. - z**2
    ones_like_x = np.ones_like(x)
    vel2 = np.log(1. + np.outer(p,x)) / np.outer(p,x) - np.log(1.+np.outer(p,ones_like_x)) / np.outer(p,np.ones_like(x))
    Sum += np.sum(2.*z/(vel2)**0.5,axis=1)

    #z = 0 terms
    Sum += 1./(np.log(1.+p)/p - 1./(1.+p))**0.5

    lgpip_tmp = np.log10(p * Sum * dz)

    pipmin=10.0**(lgpip_tmp[0])
    pipmax=10.0**(lgpip_tmp[NP-1])

    pmin=10.0**(lgp_tmp[0])
    pmax=10.0**(lgp_tmp[NP-1])

    # For rapid access the tabulated values of pip need to be uniformly spaced in logpip rather than in logp.
    scl=float(NP-1)/(lgpip_tmp[NP-1]-lgpip_tmp[0])
    
    lgpip_tab = np.zeros(NP); lgp_tab = np.zeros(NP)

    for n in range(NP):
        lgpip_tab[n]=lgpip_tmp[0]+float(n)/scl # Set points in pI(p), equally spaced.
        # Linearly interpolate in temporary table to get corresponding value of p.
        lgp_tab[n]=Linear_Interpolation(lgpip_tmp,lgp_tmp,[lgpip_tab[n]])

    return lgp_tab, lgpip_tab, pipmin, pipmax, pmin, pmax

# Set up the grid for inverting the freefall time equation
lgp_tab, lgpip_tab, pipmin, pipmax, pmin, pmax = Tabulate_pIp()

def Calculate_p_of_pIp(pIp, lgp_tab, lgpip_tab, pmin, pmax, pipmin, pipmax):
    """ For a given value of p I(p), calculate p by interpolating the grid created using Tabulate_pIp()
    This is used for numerically inverting the freefall time equation to calculate the freefall radius"""

    if pIp < pipmin:
        p_of_pip = ((pIp/pipmin)**2)*pmin # Asymptotic expression.

    elif pIp > pipmax:
        p_of_pip = ((pIp/pipmax)**0.75)*pmax # Asymptotic expression.

    else:
        lgpIp = np.log10(pIp)
        lgp = Linear_Interpolation(lgpip_tab, lgp_tab, [lgpIp])[0]
        p_of_pip = 10.**lgp

    return p_of_pip

def Calculate_Ip(p, lgp_tab, lgpip_tab, pmin, pmax, pipmin, pipmax):
    ''' For a given value of p, calculate I(p) by interpolating the grid created using Tabulate_pIp()
    This is used to compute the rate of change of the free fallradius'''

    if p < pmin:
        Ip = pipmin/p * (p/pmin)**0.5

    elif p > pmax:
        Ip = pipmax/p * (p/pmax)**(4/3.)

    else:
        lgp = np.log10(p)
        lgpIp = Linear_Interpolation(lgp_tab, lgpip_tab, [lgp])[0]
        Ip = 10.**(lgpIp-lgp)
        
    return Ip

def Calculate_dfdp_2z(p,z):
    '''Compute 2 z df(p,x)/dp at fixed x for 0 < z < 1
    
    g(p,x) = ln(1+px)/px - ln(1+p)/p
    
    f(p,x) = f(p,x)**-0.5

    Note, variable transformation x = 1-z**2'''
    
    x = 1.-z**2

    # 2 z df/dp Limit for z --> 0

    if z.min() < 0.0 or z.max() > 1.0:
        print "Error, 2z df/dp is only defined for 0 < z < 1"
        quit()

    g = lambda x, p: np.log(1.+p*x) / (p*x) - np.log(1.+p)/p
    dgdp = lambda x, p: x/(1.+p*x)/(p*x) - np.log(1.+p*x)/(p**2*x) + np.log(1.+p)/p**2 - 1./(p*(1.+p))

    dfdp_2z = np.zeros((len(z),len(p)))

    # z->0 limit
    lo = z == 0.0
    dfdp_2z[lo] = 2 * -0.5 * (1/(p*(1.+p)) - np.log(1.+p)/p**2 + 1/(1.+p)**2) * (np.log(1.+p)/p - 1/(1.+p))**-1.5

    # z->1 limit
    hi = z == 1.

    dfdp_2z[hi] = 2 * -0.5 * (-1/(1.+p)/p + np.log(1.+p)/p**2) * (1.-np.log(1.+p)/p)**-1.5

    # Normal case
    normal = (lo==False)&(hi==False)
    p_pass = np.outer(p,np.ones_like(x[normal]))

    dfdp_2z[normal] = np.swapaxes(2. * z[normal] * -0.5 * g(x[normal],p_pass)**-1.5 * dgdp(x[normal],p_pass),0,1)

    return dfdp_2z

def Tabulate_dIdp():
    '''Tabulate log10( dI(p)/dp ) as a function of log10(p)

    dI/dp is obtained via int_0^1 2 z df(p,z)/dp dz'''
    
    NP = 100; NZ = 1000
    dz = 1./NZ

    p = 10.**(-2 + 4*(np.arange(NP)-1)  /(NP-1.))

    #p = 10.**(np.arange(

    z = 1. - (np.arange(NZ+1.))*dz

    dfdp_2z = Calculate_dfdp_2z(p,z)

    # Trapezium integration
    
    dIdp = np.sum(0.5*(dfdp_2z[0:-1]+dfdp_2z[1:]) * dz,axis=0)

    dIdp_tab = dIdp; p_tab = p

    pmax = p[-1]  
    pmin = p[0]

    dIdpmax = dIdp[-1] # dI(p)/dp at p=pmax
    dIdpmin = dIdp[0] # dI(p)/dp at p=pmin

    return p_tab, dIdp_tab, dIdpmax, dIdpmin

p_tab, dIdp_tab, dIdpmax, dIdpmin = Tabulate_dIdp()

def Calculate_dIdp(p, p_tab, dIdp_tab, pmin, pmax, pipmin, pipmax):
    ''' For a given value of p, calculate I(p) by interpolating the grid created using Tabulate_pIp()
    This is used to compute the rate of change of the free fallradius'''

    #got here, need to handle the limits properly. Should maybe be able to get these analytically from Ip limits (i.e. just do basic calculus on those limts).

    if p < pmin:
        dIdp = -0.5 * pipmin/p**2 * (p/pmin)**0.5

    elif p > pmax:
        dIdp = 1/3. * pipmax/p**2 * (p/pmax)**(4/3.)

    else:
        dIdp = Linear_Interpolation(p_tab, dIdp_tab, [p])[0]
        
    return dIdp

def Calculate_Freefall_Radius_Over_Rvir(tform, tdyn, anfw):
    ''' Calculate the freefall radius for an NFW halo that "formed" at time, tform
    Freefall radius is defined as the radius within which gas has had sufficient time to radially freefall under gravity to the halo centre after the halo formation event.
    inputs: tform = time since halo formation event
            tdyn  = halo dynamical time (in same units as tform)
            anfw  = inverse of NFW halo concentration
    output: r_freefall_o_rvir = freefall radius in units of the virial radius'''

    if tdyn == 0.0: # No gas can freefall if halo has not appeared yet!
        return 0.0

    f_anfw= np.log(1.0+1.0/anfw)-1.0/(1.0+anfw)

    t_tdyn=tform/tdyn # Elapsed time in units of the halo dynamical time, defined as  tdyn = r_vir/Vc(r_vir)

    pIp = t_tdyn * ( 2. / (anfw * f_anfw) ) **0.5 / anfw

    p = Calculate_p_of_pIp(pIp, lgp_tab, lgpip_tab, pmin, pmax, pipmin, pipmax)

    r_freefall_o_rvir = anfw * p

    return r_freefall_o_rvir

def Calculate_Freefall_Radius_Rate(anfw, anfw_dot, rff, rhalo, rhalo_dot, mhalo, mhalo_dot):
    '''Calculate the rate with which the freefall radius (see Calculate_Freefall_Radius_Over_Rvir) is propagating outwards into a NFW halo in kpc Gyr^-1
    inputs: 
            anfw      = inverse of NFW halo concentration
            anfw_dot  = Rate of change of inverse of NFW halo concentration in Gyr^-1
            rff       = freefall radius in kpc
            rhalo     = Halo virial radius in kpc
            rhalo_dot = Rate of change of halo virial radius in kpc Gyr^-1
            mhalo     = Halo mass in Msun
            mhalo_dot = Rate of change of halo mass in Msun Gyr^-1
            
    output: 
            rff_dot = rate of change of the freefall radius in kpc Gyr^-1'''

    # Compute dc/dt where c is the nfw halo concentration
    c = 1./anfw
    c_dot = -c**2 * anfw_dot

    r_s = anfw * rhalo # Nfw scale radius in kpc
    r_s_dot = 1/c * (rhalo_dot - r_s * c_dot)

    p = rff / r_s

    F_o_c = np.log(1.+c) - c / (1.+c)
    F_o_c_dot = c_dot * c/(1.+c)**2

    # Compute I(p)
    Ip = Calculate_Ip(p, lgp_tab, lgpip_tab, pmin, pmax, pipmin, pipmax)
    dIdp = Calculate_dIdp(p, p_tab, dIdp_tab, pmin, pmax, pipmin, pipmax)
    
    dmhalo_m0p5_dt = -0.5 * mhalo**(-1.5) * mhalo_dot
    dr_s_0p5_dt = 0.5 * r_s**(-0.5) * r_s_dot
    dF_o_c_0p5_dt = 0.5 * F_o_c**(-0.5) * F_o_c_dot

    conversion_factor = Constant.Msun**0.5 / Constant.kpc**1.5 * Constant.Gyr

    term1 = (2 * Constant.G)**0.5 * r_s**-0.5 * rff**-1 * mhalo**0.5 * Ip**-1 * F_o_c**-0.5 * conversion_factor

    # Calculate dr_ff/dt (1/r_ff  + 1/r_s dI/dp / I)
    term2 = term1 - dmhalo_m0p5_dt * mhalo**0.5 - dr_s_0p5_dt / r_s**0.5 - dF_o_c_0p5_dt / F_o_c**0.5 + dIdp/Ip * rff/r_s**2 * r_s_dot
    rff_dot = term2 / (1./rff + 1./r_s * dIdp/Ip) # kpc Gyr^-1

    return rff_dot

def Mass_Shell_Arctan_Function(x):
    '''Calculates 1 - x arctan(1/x)'''
    return 1. - x * np.arctan(1./x)

def Calculate_Mass_Fraction_Within_r(r, rcore, rhalo):
    """Calculate the fraction of hot gas enclosed within a radius r"""
    if r == 0.0:
        fraction = 0.0
    else:
        fraction = r / rhalo * Mass_Shell_Arctan_Function(rcore/r) / Mass_Shell_Arctan_Function(rcore/rhalo)

    return fraction

def Calculate_Mass_Shell(rcore, rinfallj, rinfalljm1, rhalo, Mhot, outfractions=False):
    ''' Calculate the mass of hot gas enclosed within a shell bounded by radii rinfallj and rinfalljm1.
    inputs:  rcore= core radius of hot gas profile in kpc
          rinfallj= infall radius at current timestep in kpc
        rinfalljm1= infall radius at previous timestep in kpc
             rhalo= halo radius in kpc
              Mhot= mass of hot gas in halo'''
    
    
    if rinfallj == 0.:
        fractionj = 0.
    else:
        fractionj = Calculate_Mass_Fraction_Within_r(rinfallj, rcore, rhalo)

    if rinfalljm1 == 0.:
        fractionjm1 = 0.
    else:
        fractionjm1 = Calculate_Mass_Fraction_Within_r(rinfalljm1, rcore, rhalo)

    print fractionjm1

    dMcool = Mhot * (fractionj - fractionjm1)

    if outfractions:
        return dMcool, fractionj, fractionjm1
    else:
        return dMcool

def Calculate_Hot_Gas_Density_Beta_Profile(r,r_core,r_halo,mhot):
    """Calculate the hot gas density (in Msun kpc^-3) of a beta profle at a radius r
    inputs:  r = radius to evaluate hot gas density (kpc)
         r_core = core radius of the beta profile
         r_halo = halo virial radius (kpc)
         mhot   = hot gas mass in the beta profile (Msun)
    output: rho_gas = hot gas density (Msun kpc^-3)"""

    term1 = mhot / (4.*np.pi * (r**2+r_core**2))
    term2 = (r_halo - r_core * np.arctan(r_halo/r_core))**-1

    rho_gas = term1 * term2

    return rho_gas

def Calculate_Hot_Gas_Density_Beta_Profile_Truncated(r,r_inner,r_core,r_halo,mhot):
    """Calculate the hot gas density at a radius r in (Msun kpc^-3) of a beta profile truncated below r_inner and above r_halo
    inputs:  r = radius to evaluate hot gas density (kpc)
         r_inner = inner truncation radius
         r_core = core radius of the beta profile
         r_halo = halo virial radius (kpc)
         mhot   = hot gas mass in the beta profile (Msun)
    output: rho_gas = hot gas density (Msun kpc^-3)"""

    if r_inner == r_halo:
        return np.inf

    # Calculate density profile normalisation
    rho_0 = mhot / (4.*np.pi*r_core**2 * (r_halo-r_inner-r_core*(np.arctan(r_halo/r_core)-np.arctan(r_inner/r_core))))

    rho_gas = rho_0 / (1.+(r/r_core)**2)

    #print r_inner, r_core, r_halo, mhot, rho_0, np.arctan(r_halo/r_core), np.arctan(r_inner/r_core)

    return rho_gas


def Calculate_Mass_Infall_Rate(r_infall, r_infall_dot, M_notional, rhalo, rcore, mhot):
    """Calculate the mass infall rate for a beta hot gas profile
    inputs: r_infall = infall_radius (kpc)
            r_infall_dot = rate of change of the infall radius (kpc Gyr^-1)
            M_notional = mass in the hot gas notional profile
            rhalo = halo virial radius (kpc)
            rcore = beta profile core radius (kpc)
    outputs: minfall_dot = hot gas infall rate (Msun Gyr^-1)"""

    #rho_gas = Calculate_Hot_Gas_Density_Beta_Profile(r_infall,rcore,rhalo,M_notional)
    
    rho_gas = Calculate_Hot_Gas_Density_Beta_Profile_Truncated(r_infall, r_infall, rcore, rhalo, mhot)

    if rho_gas == np.inf:
        return np.inf

    minfall_dot = 4. * np.pi * r_infall**2 * rho_gas * r_infall_dot # Msun Gyr^-1
    
    if np.isnan(minfall_dot):
        print "hmm", r_infall, rho_gas, r_infall_dot
        exit()

    return minfall_dot

def Calculate_Hot_Gas_Rotation_Velocity_Truncated_Profile(r_inner,r_core,r_halo,mhot,Jhot):
    """Calculate the rotation velocity of the hot gas in a truncated beta profile (truncated below r_inner and above r_halo)
    inputs:  r = radius to evaluate hot gas density (kpc)
         r_inner = inner truncation radius
         r_core = core radius of the beta profile
         r_halo = halo virial radius (kpc)
         mhot   = hot gas mass in the beta profile (Msun)
         Jhot   = angular momentum in the hot gas beta profile (Msun kpc kms^-1)
    output: vrot_hot = hot gas density (Msun kpc^-3)"""

    if r_inner == r_halo:
        return np.inf

    # Calculate density profile normalisation
    rho_0 = mhot / (4.*np.pi*r_core**2 * (r_halo-r_inner-r_core*(np.arctan(r_halo/r_core)-np.arctan(r_inner/r_core)))) # Msun kpc^-3

    # Calculate hot gas rotation profile, this can be derived by integrating dJ/dr between r_inner and r_halo

    # First calculate the ratio of Jhot/vrot_hot
    Jhot_o_vrot = 2. *np.pi *rho_0 *r_core**2 * (r_halo**2 - r_inner**2 - r_core**2 * (np.log(r_core**2 + r_halo**2) -np.log(r_core**2 +r_inner**2))) # Msun kpc
    
    vrot_hot = Jhot / Jhot_o_vrot # kms^-1

    return vrot_hot

def Calculate_J_Infall_Rate(r_infall, Mdot_infall, rhalo, rcore, Jhot, mhot):
    """Compute the infalling angular momentum rate for a truncated beta profile"""

    # This will occur when r_infall >= r_halo. The code will deal with this at a later stage
    if Mdot_infall == np.inf:
        return np.inf

    # Calculate hot gas rotation velocity (normalised such that Jhot is the total angular momentum in a truncated beta profile)
    vrot_hot = Calculate_Hot_Gas_Rotation_Velocity_Truncated_Profile(r_infall,rcore,rhalo,mhot,Jhot)

    Jinfall_dot = Mdot_infall * vrot_hot * r_infall # Msun kpc kms^-1 Gyr^-1
    
    if np.isnan(Jinfall_dot):
        print "hmm", r_infall, rho_gas_J, r_infall_dot
        exit()

    return Jinfall_dot

def Tabulate_mCDC(lxmin = -4.6):
    #nxtab = 10000 - galform value. This will be way too slow for my purposes though
    nxtab = 100
    lxmax = 7.0

    lx = lxmin + (lxmax-lxmin) * np.arange(nxtab) / (nxtab-1.)
    xCDC = np.exp(lx)
    mCDC = xCDC - np.arctan(xCDC)

    logxCDC = np.log10(xCDC); logmCDC = np.log10(mCDC)
    
    return logxCDC, logmCDC

logxCDC, logmCDC = Tabulate_mCDC()

def Calculate_r_Interior_M(mfrac,rcore,rhalo,rmax=None):
    ''' Calculate the maximum of rmax and the radius that encloses a given fraction of the total halo hot gas mass in kpc
    inputs: mfrac = fraction of halo mass that is enclosed in r_enclose
            rcore = core radius of hot gas profile in kpc
            rhalo = halo radius in kpc
            rmax  = upper bound on allowed value of r_enclose in kpc
    outputs: r_enclose = radius enclosing mfrac

    mfrac = mcooled/(mhot+mcooled)'''

    global logxCDC, logmCDC

    if mfrac <= 0.0:
        r_enclose = 0.0
    elif mfrac >= 1.0:
        r_enclose = 1.0 # 1kpc?? doesn't make sense?
    else:            
        c = rcore/rhalo

        uprime = mfrac * (1./c - np.arctan(1./c))
        

        if uprime < 0.0:
            invert_CDC = 0.0
        else:
            # Retabulate if we need to go outside of the original tabulated range
            lxmin_temp = -4.6
            while logmCDC.min() > np.log10([uprime]):
                lxmin_temp += -1.0
                logxCDC, logmCDC = Tabulate_mCDC(lxmin_temp)

            invert_CDC = 10**Linear_Interpolation(logmCDC,logxCDC,np.log10([uprime]))[0]
            if np.log10(uprime) < logmCDC[0]:
                invert_CDC = (3. * uprime)**(1./3.)
            elif np.log10(uprime) > logmCDC[-1]:
                invert_CDC = 0.25*np.pi + 0.5*uprime + 0.25 * np.sqrt(4. * uprime**2 + 4*np.pi * uprime + np.pi**2 - 16.0)

        r_enclose = rcore * invert_CDC

        # If called to calculate previous cooling radius, this ensures that nothing infalls if the infall radius shrinks
        if r_enclose > rmax and rmax is not None:
            r_enclose = rmax

    return r_enclose


def Read_NFW_Data():
    '''Read in tabulated halo angular momentum information. See Appendix A in Cole 2000 to understand where this comes from.
    Outputs:
            a_tab = NFW scale length / virial radius
            vrot_norm_tab = normalisation term such that Vrot = vrot_norm * Vhalo * lambda_spin
            J_o_Vrot_tab = Total halo angular momentum for a flat velocity profile / Vrot, units are in Mhalo Rhalo'''
    a_tab, vrot_norm_tab, J_o_Vrot_tab = np.loadtxt(Paths.nfw,unpack=True, usecols=(0,1,2))
    return a_tab, vrot_norm_tab, J_o_Vrot_tab

a_tab, vrot_norm_tab, J_o_Vrot_tab = Read_NFW_Data()

def Calculate_V_Rotation_Normalisation(a):
    vrot_norm = Linear_Interpolation(a_tab,vrot_norm_tab,[a])[0]
    return vrot_norm

def Calculate_J_o_Vrot(a):
    J_o_Vrot = Linear_Interpolation(a_tab,J_o_Vrot_tab,[a])[0]
    return J_o_Vrot

def Calculate_j_Interior(r,rcore,rhalo):
    ''' Calculate the angular momentum of hot gas interior to a radius, r, assuming a constant density core profile
    and a flat rotation velocity profile for both the dark matter and the hot gas
    inputs: r = radius of interest
            rcore = core radius for halo hot gas profile
            rhalo = halo virial radius
    outputs: j_intr = angular momentum interior to r: units = M_halo V_rot r_halo'''

    if r == 0.:
        return 0.0

    u = r/rhalo
    u_c2 = (r / rcore)**2
    c = rcore/rhalo
    ci = rhalo/rcore

    if u_c2 <= 0.1:
        j_intr = np.pi/8. * u**2 * (u_c2 * (0.5+u_c2*(-1./3. + u_c2*(0.25-u_c2*0.2)))) / Mass_Shell_Arctan_Function(c) #Taylor expansion in (r/rcore)^2.
    elif u_c2 > 0.999:
        j_intr = np.pi/8. * u**2 / Mass_Shell_Arctan_Function(c) * (1.-c**2*np.log(1.+ci**2) -(2./(1.+c**2) -2.*np.log(1.+ci**2)) * c**2 * (u-1.) -(3.*np.log(1.+ci**2) -3./(1.+c**2) -2./(1.+c**2)**2) *c**2 * (u-1.)**2)
    else:
        j_intr = np.pi/8. * u**2 * (1. - np.log(1. + u_c2)/u_c2) / Mass_Shell_Arctan_Function(c)

    return j_intr

    
def Calculate_Infalling_Angular_Momentum(rinfall, rinfall_prev, minfall, fm_infall, rhalo, vhalo, strc, rcore, spin, Jhot=None, Mhot=None):
    '''Calculate the angular momentum of a shell of infalling gas
       Inputs:
              rinfall = outer radius of infalling shell / kpc
              rinfall_prev = inner radius of infalling shell / kpc
              minfall = mass of infalling shell
              fm_infall = fraction of mass in gas profile that is infalling
              rhalo = virial radius / kpc
              vhalo = halo circular velocity at virial radius / kms^-1
              strc = dimensionless inverse NFW halo concentration
              rcore = hot gas core radius / kpc
              spin = dimenionless halo spin parameter
              Jhot = total angular momentum in the notional hot gas profile / Msun kms^-1 kpc, only used if angular_momentum_conservation=True
              Mhot = total mass in the notional hot gas profile / Msun, only used if angular_momentum_conservation=True'''

    if fm_infall == 0.0:
        return 0.0

    if Parameter.angular_momentum_conservation and Mhot == 0.0:
        return 0.0

    v_rot_norm = Calculate_V_Rotation_Normalisation(strc) # dimensionless

    # Calculate infalling angular momentum assuming a constant velocity profile for the hot gas
    if Parameter.gas_rotation_profile == "flat":
        fj_infall = Calculate_j_Interior(rinfall,rcore,rhalo) - Calculate_j_Interior(rinfall_prev,rcore,rhalo)
        
        if not Parameter.angular_momentum_conservation:
            J_infalls = minfall * spin * vhalo * v_rot_norm * rhalo * fj_infall / fm_infall # Msun kpc kms^-1
            print "warning, this doesn't include the Simon White fix to make j_hot = j_dm"
        else:
            J_infalls = Jhot * fj_infall # Msun kpc kms^-1

    # Calculate infalling angular momentum assuming a hot gas velocity profile \propto 1/r such that specific angular momentum is constant with radius
    # Note this is not a choice that is implemented in GALFORM
    elif Parameter.gas_rotation_profile == "const_j":
        
        if not Parameter.angular_momentum_conservation:
            J_o_Vrot = Calculate_J_o_Vrot([strc])[0] # Mhalo * Rhalo

            # Calculate constant rotation velocity for dark matter halo
            Vrot = v_rot_norm * vhalo * spin # kms^-1

            # Calculate total specific angular momentum of the halo
            j_halo = J_o_Vrot * Vrot * rhalo # kms^-1 kpc

            J_infalls = j_halo * minfall

        else:
            j_notional = Jhot / Mhot
            J_infalls = minfall * j_notional

    else:
        print "Error in Calculate_Infalling_Angular_Momentum: gas_rotation_profile = ", gas_rotation_profile, " is not implemented."
        quit()

    return J_infalls

def Calculate_Halo_Rotation_Velocity(vhalo,spin,strc):
    """Calculate halo rotatio velocity in kms^-1"""

    v_rot_norm = Calculate_V_Rotation_Normalisation(strc) # dimensionless

    if not Parameter.angular_momentum_conservation:
        print "Error, the non-conserving angular momentum scheme (i.e. the galform scheme) has not been implemented for the continous cooling model"
        quit()

    else:
        v_rot = v_rot_norm * vhalo * spin

    return v_rot

cooling_t1 = time.time()
print "Done, time taken = ", cooling_t1 - cooling_t0
