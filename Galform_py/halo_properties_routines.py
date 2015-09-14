import numpy as np; from input_parameters import *; from utilities_interpolation import *; from scipy.optimize import brentq; import time; from utilities_cosmology import *

print "Tabulating halo properties"
halo_prop_t0 = time.time()

def Halo_Spin():
    '''Generate a value of the dimenionless halo spin parameter.'''

    spin_max = 0.2 # maximum allowed spin parameter

    Lambda_Halo = Parameter.spin_med*np.exp(Parameter.spin_disp*np.random.randn())

    while Lambda_Halo > spin_max: # Limit the spin parameter to be less than spin_max.
        Lambda_Halo=Parameter.spin_med*np.exp(Parameter.spin_disp*np.random.randn())

    return Lambda_Halo

def Calculate_Halo_Virial_Radius(mhalo, vhalo):
    if vhalo == 0.0:
        return 0.0

    rhalo = Constant.G * mhalo * Constant.Msun / np.square(vhalo * Constant.kms) / Constant.kpc

    return rhalo

def Calculate_Halo_Virial_Radius_Rate(z,mhalo,mhalo_dot):
    '''Calculate rate of change the halo virial radius'''

    rhobar = Calculate_Mean_Halo_Density(z) # Msun kpc^-3
    if Parameter.propagate_vhalo:
        rhobar_dot = 0.0
    else:
        rhobar_dot = Calculate_Mean_Halo_Density_Rate(z)

    dmhalo_third_dt = 1/3. * mhalo**(-2/3.) * mhalo_dot # d(M_H^{1/3})/dt
    drhobar_mthird_dt = -1/3. * rhobar**(-4/3.) * rhobar_dot # d(rhobar^{-1/3}/dt
    
    rhalo_dot = (3/(4*np.pi))**(1/3.) * (dmhalo_third_dt * rhobar**(-1/3.) + drhobar_mthird_dt * mhalo**(1/3.)) # Msun^-1/3 Mpc Msun^1/3 Gyr^-1 = Mpc Gyr^-1
    rhalo_dot *= 1000. # kpc # Gyr^-1

    return rhalo_dot

def Calculate_Halo_Virial_Velocity(mhalo, a):
    ''' Compute the circular velocity at the virial radius assuming
    that the has an overdensity relative to the critical density
    given by the spherical collapse model, as calculated in Eke 96 (or97?)'''

    z = 1./a -1

    # Calculate halo virial velocity assuming the halo has mean density = 200 mean density at a given redshift
    # Vhalo = (G Mhalo H0)^1/3 * (1+z)^0.5 * 200^1/6
    vchalo_200_flat = (1+z)**0.5 * (Constant.G *mhalo * Constant.Msun * Parameter.h * Constant.H0100 * Constant.kms / Constant.Mpc)**(1/3.) * (100**(1/6.)) /Constant.kms # kms^-1 This is the virial velocity for Delta = 200 rho_crit and for omega_m = 1 universe

    halo_overdensity = Halo_Overdensity(a)

    vchalo = vchalo_200_flat * halo_overdensity**(1./6)
    
    return vchalo

def Calculate_Halo_Virial_Velocity_Rate(z,mhalo,mhalo_dot):
    """Calculate the rate of change of the halo circular velocity at the virial radius"""
    
    if Parameter.propagate_vhalo:
        return 0.0
    
    # Otherwise, use the spherical collapse model
    else:
        rhobar = Calculate_Mean_Halo_Density(z)
        rhobar_dot = Calculate_Mean_Halo_Density_Rate(z)

        dmhalo_third_dt = 1/3. * mhalo**(-2/3.) * mhalo_dot # d(M_H^{1/3})/dt
        drhobar_sixth_dt = 1/6. * rhobar**(-5/6.) * rhobar_dot # d(rhobar^{1/6}/dt

        vhalo_dot = (Constant.G**3 * 4 * np.pi /3.)**(1/6.) * (dmhalo_third_dt * rhobar**(1/6.) + drhobar_sixth_dt * mhalo**(1/3.)) # m^3/2 kg^-0.5 s^-1 Gyr^-1 Msun^1/3 Msun^1/6 Mpc^-0.5 = m^3/2 kg^-0.5 Msun^0.5 Gyr^-1 s^-1 Mpc^-0.5
        vhalo_dot *= Constant.Msun**0.5 / Constant.Mpc**0.5 / Constant.km # km s^-1 Gyr^-1

    return vhalo_dot

def NFW_Scale_Radius(mhalo, a):
    """Compute the scale length (strc) for an NFW halo of specified mass (mhalo in Msun) and identification expansion factor (a)
    in units of the virial radius, using the method proposed by Navarro, Frenk \& White (1996).
    
    Details of the equations to be solved and the algorithms used to solve them can be found in the Galform code dark_matter_profiles.NFW_scale_radius.F90"""

    if mhalo == 0.0:
        return 0.0

    # Define constants used in this calculation
    C=3000.0; F=0.01; RATIO2=0.2274682116

    fm=F*mhalo

    deltac=np.sqrt( 2.0 * RATIO2 * (sigma_Mass(fm)**2 - sigma_Mass(mhalo)**2) ) + Critical_Overdensity(a)

    a_form = a_of_deltac(deltac)

    delta_coll = C * Parameter.omm * (a/a_form)**3
    
    # As we have omm = Omega_m0 in the above rather than Omega(z) we now need to multiply by 3*Omega(z)/(Omega_0*Delta_vir) to get since the
    # function "Halo_Overdensity" is defined as Halo_Overdensity= Delta_vir*omega0/(200*omega(z)) this is achieved as follows:
    
    fa3_coll= delta_coll * 3.0 / ( Constant.Delta200 * Halo_Overdensity(a) )

    strc = a_of_fa3(fa3_coll, a_tab, fa3_tab)

    return strc



def Tabulate_rho_vir_o_rho_crit():
    amax = 1.1; alow = 0.1; amin = 0.05

    Omega_tab, rho_vir_o_rho_crit_tab, delta_crit_tab = np.loadtxt(Input_Data_Path.spherical_collapse,unpack=True)

    inv_delta_a= (199)/(amax-amin) # I should probably do this in the initialisation process for the code, not every time we want this quantity

    aflat = np.arange(0,200) / inv_delta_a + amin
    
    lambda_a = Parameter.oml / (Parameter.oml+(1.0-Parameter.omm-Parameter.oml)/aflat**2+Parameter.omm/aflat**3)
    omega_a = Parameter.omm*lambda_a/(Parameter.oml*aflat**3)
    
    den_flat = Linear_Interpolation(Omega_tab, rho_vir_o_rho_crit_tab, omega_a) * Parameter.omm / omega_a

    low = aflat < alow
    DeltaEdS=18.0*np.pi**2 # Density contrast at virial radius of a halo in an Einstein-de Sitter universe.
    den_flat[low] =  DeltaEdS * Parameter.omm # High redshift extrapolation

    return aflat, den_flat

aflat1_tab, den_flat_tab = Tabulate_rho_vir_o_rho_crit()

def Halo_Overdensity(a):
    '''The overdensity for a spherical top-hat perturbation after virialization
    in units of present critical density multiplied by $200(1+z)^3$.
    
    This routine assumes a flat Universe with $\Omega_0+\Lambda_0=1$ and works by tabulating numerical
    results and use the {\tt flat.data} file originally produced by
    Vince Eke.'''

    if a > aflat1_tab.max() or a < aflat1_tab.min():
        print "Error in Halo_Overdensity(): requested expansion factor a=", a, " is outside the tabulated range"
        quit()

    halo_overdensity = np.array(Linear_Interpolation(aflat1_tab,den_flat_tab,[a]))[0]

    halo_overdensity *= 1./Constant.Delta200 

    return halo_overdensity

def Tabulate_Delta_Flat():

    NSUM=100

    Omega_tab, rho_vir_o_rho_crit_tab, delta_crit_tab = np.loadtxt(Input_Data_Path.spherical_collapse,unpack=True)
    
    omflat = Omega_tab; density = rho_vir_o_rho_crit_tab; delflat  =delta_crit_tab
 
    # Evaluate constant required to normalize the linear growth factor.
    x0=(2.0*(1.0/Parameter.omm-1.0))**(1./3.)   
    dxp=x0/float(NSUM)
    xp = x0 * (np.arange(0,NSUM)+0.5) / NSUM
    Sum = np.sum( (xp/(xp**3 +2.0))**1.5) * dxp
    dlin0=Sum*np.sqrt(x0**3+2.0)/np.sqrt(x0**3)
    
    # Tabulate delta_c versus a for the specified values of omega0, lambda0. Spacing in a is linear in order to enable quick look up.
    amax = 1.1; amin = 0.05

    ntab = 200
    aflat_tab = np.zeros(ntab)
    delta_flat_tab = np.zeros(ntab)
    inv_delta_a= ntab/float(amax-amin)


    aflat_tab = np.arange(ntab)/inv_delta_a + amin
    lambda_a = lambda_a= Parameter.oml / (Parameter.oml+(1.0-Parameter.omm-Parameter.oml)/aflat_tab**2+Parameter.omm/aflat_tab**3)
    omega_a = Parameter.omm*lambda_a/(Parameter.oml*aflat_tab**3)

    x = x0 * aflat_tab
    dxp = x/NSUM
    xp = np.outer(x, (np.arange(0.,NSUM)+0.5) / NSUM)
    Sum = np.sum( (xp/(xp**3+2.0))**1.5, axis=1) * dxp
    dlin=(Sum*np.sqrt(x**3+2.0)/np.sqrt(x**3))/dlin0
    
    delta_flat_tab = Linear_Interpolation(Omega_tab,delta_crit_tab,omega_a)/dlin

    return aflat_tab, delta_flat_tab

aflat2_tab, delta_flat_tab = Tabulate_Delta_Flat()

def Critical_Overdensity(a):
   
    '''Calculates the critical linear overdensity, $\delta_{\rm c}$, for collapse at time $t$ for density field normalized at
    reference epoch $a=1$. N.B. the critical overdensity is the value extrapolated from the collapse epoch to $a=1$. For
    example, for $\Omega=1$, $\delta_{\rm c} = 1.686(1+z)$.'''
          
    critical_overdensity = Linear_Interpolation(aflat2_tab,delta_flat_tab,[a])[0]

    return critical_overdensity

def Tabulate_a_of_deltac():
    '''Compute a grid of expansion factor against density threshold'''

    a_first = 1.0; a_last = 0.06
    delta_first = Critical_Overdensity(a_first)
    delta_last = Critical_Overdensity(a_last)

    NT_min = 100
    NT_per_delta = 2.0
    NT = max(int(NT_per_delta*(delta_last-delta_first)),NT_min)

    a = 1.0
    d_delta = (delta_last-delta_first)/float(NT-1)
    inv_d_delta = 1./d_delta

    a_of_deltac_tab = np.zeros(NT)
    delta_tab = np.zeros(NT)

    Critical_Overdensity_brent = lambda a, delta : Critical_Overdensity(a) - delta

    for i in range(NT):
        delta = delta_first + d_delta * i
        a = brentq(Critical_Overdensity_brent,0.055,1.0,args=(delta))
        a_of_deltac_tab[i] = a
        delta_tab[i] = delta

    return delta_tab, a_of_deltac_tab

delta_tab, a_of_deltac_tab = Tabulate_a_of_deltac()

def a_of_deltac(deltac):
    '''Tabulated function that returns the expansion factor, a, at which the density threshold delta_c = deltac has the specified value.'''
    
    if deltac < delta_tab.min() or deltac > delta_tab.max():
        print "Error: requested deltac is outside of tabulated range"
        quit()

    a_of_deltac = Linear_Interpolation(delta_tab, a_of_deltac_tab, [deltac])[0]

    return a_of_deltac

def fa3_func(a):
    '''The function: {\tt func\_fa3(a)} $=a^{-3}/[\ln(1+1/a)-1/(1+a)]$.'''
    fa3 = 1.0/((a**3)*(np.log(1.0+1.0/a)-1.0/(1.0+a)))
    return fa3

def Tabulate_a_of_fa3(lnfa_last=6):
    lnfa_first = 0.0
    NT = 100

    fa3_root = lambda a, fa3: fa3_func(a) - fa3
    
    a_tab = np.zeros(NT); fa3_tab = np.zeros(NT)

    for n in range(NT):

        fa3_tab[n]=np.exp(lnfa_first+(lnfa_last-lnfa_first)*n/(NT-1.))
        a_tab[n] = brentq(fa3_root,0.001,3.0,args=(fa3_tab[n]))
    
    return a_tab, fa3_tab

a_tab, fa3_tab = Tabulate_a_of_fa3()

def a_of_fa3(fa3, a_tab, fa3_tab):
    """Tabulated function that returns the expansion factor, $a$, at which the function $f(a)/a^3$ has the specified value, where
    f(a) = {1 \over \ln(1+1/a)-1/(1+a)}."""

    if fa3 < fa3_tab.min():
        print "Error: requested fa3 is outside the tabulated range"
        quit()

    elif fa3 > fa3_tab.max():
        a_tab, fa3_tab = Tabulate_a_of_fa3(np.log(fa3)*2)

    a_of_fa3 = Linear_Interpolation(fa3_tab, a_tab, [fa3])[0]

    return a_of_fa3

def Tabulate_Spline_Interpolation_Sigma_Mass():
    
    # In principle you should now check if you have strayed outside of tabulated range of mhalo
    # This has not yet been implemented

    # Load tabulated spline fits of sigma as a function of m from input file
    m_sigma_Mass, s_sigma_Mass, s2_sigma_Mass, a_sigma_Mass, a2_sigma_Mass = np.loadtxt(Input_Data_Path.spline_interpolation_sigma_mass,unpack=True)

    return m_sigma_Mass, s_sigma_Mass, s2_sigma_Mass, a_sigma_Mass, a2_sigma_Mass

m_sigma_Mass_tab, s_sigma_Mass_tab, s2_sigma_Mass_tab, a_sigma_Mass_tab, a2_sigma_Mass_tab = Tabulate_Spline_Interpolation_Sigma_Mass()

def sigma_Mass_Norm():
    ''' Calculate normalisation term used in sigma_Mass'''
    m8=Constant.m8_crit*Parameter.omm # The mass within an 8Mpc/h sphere.
    sigma_m8_norm = Linear_Interpolation(m_sigma_Mass_tab, s_sigma_Mass_tab, [m8])[0]
    return sigma_m8_norm

sigma_m8_norm = sigma_Mass_Norm()

def sigma_Mass(mhalo):
    ''' Calculate sigma(M)
    Inputs:
    mhalo = halo mass in Msun
    Outputs:
    sigma_m = sigma(mhalo), normalised to sigma_8'''

    mhaloh = mhalo * Parameter.h # halo mass in Msun/h

    sigma_m = Linear_Interpolation(m_sigma_Mass_tab, s_sigma_Mass_tab, [mhaloh])[0]
    # For almost any system of interest, I think the higher order corrections from the spline fit are basicly negligigble compared to zero order - hence just use linear interpolation here.

    # Now need to normalise for desired sigma_8
    sigma_m_normed = sigma_m * Parameter.sigma8 / sigma_m8_norm

    return sigma_m_normed

def Calculate_Mean_Halo_Density(z):
    """Calculate mean halo density within the virial radius, following the spherical collapse model
    """

    a = 1./(1.+z)

    # This is a mess, I'm pretty sure I'm double/triple calculating stuff like Halo_Overdensity(a) - actually it's worse, I should only need to calculate this once per time step for the entire merger tree! Same with critical density
    # mmmm, on second thoughts, for the adaptive step version, this will need to be calculated at arbitrary times
    Delta_c = Halo_Overdensity(a) # Units: (present (z=0) critical density over critical density) multiplied by $200(1+z)^3$

    rho_crit = Critical_Density(z,Parameter.h,Parameter.omm,Parameter.oml) # Msun Mpc^-3  
    rho_crit0 = Critical_Density(0.0,Parameter.h,Parameter.omm,Parameter.oml) # Present day critical density in Msun Mpc^-3

    Delta_c   = Delta_c   * Constant.Delta200 * (1+z)**3 * rho_crit0 / rho_crit # Now dimensionless

    rhobar = Delta_c * rho_crit # Msun Mpc^-3

    return rhobar

def Calculate_Mean_Halo_Density_Rate(z):
    """Calculate the rate of change of the mean halo density, following the spherical collapse model"""
    
    a = 1./(1.+z)

    # This is a mess, I'm pretty sure I'm double/triple calculating stuff like Halo_Overdensity(a) - actually it's worse, I should only need to calculate this once per time step for the entire merger tree! Same with critical density
    Delta_c = Halo_Overdensity(a) # Units: (present (z=0) critical density over critical density) multiplied by $200(1+z)^3$
    dDelta_dt = Halo_Overdensity_Rate_Change(a) # Units: (present (z=0) critical density over critical density) multiplied by $200(1+z)^3$ per Gyr

    rho_crit = Critical_Density(z,Parameter.h,Parameter.omm,Parameter.oml) # Msun Mpc^-3  
    rho_crit_dot = Critical_Density_Rate(z,Parameter.h,Parameter.omm,Parameter.oml) # Msun Mpc^-3 Gyr^-1

    rho_crit0 = Critical_Density(0.0,Parameter.h,Parameter.omm,Parameter.oml) # Present day critical density in Msun Mpc^-3

    zdot = Redshift_Rate(z, Parameter.omm, Parameter.oml, Parameter.h)

    Delta_c   = Delta_c   * Constant.Delta200 * (1+z)**3 * rho_crit0 / rho_crit # Now dimensionless
    dDelta_dt = dDelta_dt * Constant.Delta200 * (1+z)**3 * rho_crit0 / rho_crit + 3 * (1+z)**(-1) * Delta_c * zdot - Delta_c / rho_crit * rho_crit_dot# Now in Gyr^-1

    rhobar_dot = Delta_c * rho_crit_dot + dDelta_dt * rho_crit # Msun Mpc^-3 Gyr^-1

    return rhobar_dot

halo_prop_t1 = time.time()
print "Done, time taken = ", halo_prop_t1 - halo_prop_t0
