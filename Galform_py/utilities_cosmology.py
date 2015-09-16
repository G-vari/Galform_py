import numpy as np

def t_Universe(a, omm0, h0):
    ''' Compute the age of the Universe at expansion factor a for
    cosmological parameters:
    omm0 = Omega matter at z=0
    h0 = Hubble constant at z=0 in units of 100 kms^-1 Mpc^-1
    
    This routine assumes a flat universe with non-zero lambda

    Outputs:
           t = age of the Universe in Gyr
           tlb = corresponding lookback time in Gyr'''


    Gyr = 3.15576e16 # s
    Mpc = 3.0856775807e22 # m
    Hubble_Time = 1./ (100 * 10**3 * Gyr / Mpc) # The Hubble time for H_0=100km/s/Mpc.
    t_Universe=Hubble_Time*(2.0/(3.0*h0*np.sqrt(1.0-omm0)))*np.arcsinh(np.sqrt((1.0/omm0-1.0)*a)*a)

    t0 = Hubble_Time*(2.0/(3.0*h0*np.sqrt(1.0-omm0)))*np.arcsinh(np.sqrt((1.0/omm0-1.0)*1.0)*1.0)

    tlb = t0 - t_Universe

    return t_Universe, tlb

def a_Universe(t, omm0, h0):
    '''Compute the expansion factor at an age, t for
    cosmological paramters:
    omm0 = Omega matter at z=0
    h0 = Hubble constant at z=0 in units of 100 kms^-1 Mpc^-1
    
    This routine assumes a flat universe with non-zero lambda

    Outputs:
           a = expansion factor'''

    Gyr = 3.15576e16 # s
    Mpc = 3.0856775807e22 # m
    Hubble_Time = 1./ (100 * 10**3 * Gyr / Mpc) # The Hubble time for H_0=100km/s/Mpc.

    y=1.5*t*h0*np.sqrt(1.0-omm0)/Hubble_Time
    sinhy=0.5*(np.exp(y)-np.exp(-y))
    a=(sinhy*(1.0-omm0)*omm0**2)**(2.0/3.0)/(omm0*(1.0-omm0))

    return a

def Redshift_Rate(z, omm0, oml0, h0):
    '''Compute dz/dt in Gyr^-1'''
    
    H = lambda z: H0 * (omm0*(1.0+z)**3.0 + oml0)**0.5

    H0 = h0 * 100. # kms^-1 Mpc^-1
    km = 1000.0 # m
    Gyr = 3.15576e16 # s
    Mpc = 3.0856775807e22 # m

    zdot = - H(z) * (1+z) # kms^-1 Mpc^-1

    zdot *= km / Mpc * Gyr

    return zdot

def Hubble_Parameter_Rate(z,h0,omm0):
    '''Compute the rate of change of the Hubble parameter, in kms^-1 Mpc^-1 Gyr^-1'''
    
    km = 1000.0 # m
    Gyr = 3.15576e16 # s
    Mpc = 3.0856775807e22 # m

    dH_dt = -3/2. * h0**2 * omm0 * (1+z)**3 * 100**2 * km * Gyr / Mpc # kms^-1 Mpc^-1 Gyr^-1

    return dH_dt

def Critical_Density(z,h,omm,oml):
    '''Compute the critical density of the Universe in Msun Mpc^-3'''

    H = lambda z: H0 * (omm*(1.0+z)**3.0 + oml)**0.5

    H0 = h * 100. # kms^-1 Mpc^-1

    G = 6.67259e-11 # m^3/kg/s^2
    Msun=1.9891e30 # The mass of the Sun in kg
    km = 1000.0 # m
    Mpc = 3.0856775807e22 # m

    rho_crit = 3 * H(z)**2 / (8 * np.pi * G) # km^2 s^-2 Mpc^-2 m^-3 kg s^2 = km^2 Mpc^-2 m^-3 kg

    rho_crit *= km**2 * Mpc / Msun # Msun Mpc^-3

    return rho_crit

def Critical_Density_Rate(z,h,omm,oml):
    '''Compute the rate of change of the critical density of the Universe, d(rho_crit(z))/dt'''

    G = 6.67259e-11 # m^3/kg/s^2
    Msun=1.9891e30 # The mass of the Sun in kg

    H = lambda z: H0 * (omm*(1.0+z)**3.0 + oml)**0.5

    H0 = h * 100 # kms^-1 Mpc^-1

    drho_dt = 6 * H(z) / (8 * np.pi * G) * Hubble_Parameter_Rate(z,h,omm) # km s^-1 Mpc^-1 m^-3 kg s^2 km s^-1 Mpc^-1 Gyr^-1 = km^2 m^-3 Mpc^-2 kg Gyr^-1

    # Convert to desired output units
    km = 1000.0 # m
    Mpc = 3.0856775807e22 # m
    drho_dt *= km**2 * Mpc / Msun # Msun Mpc^-3 Gyr^-1

    return drho_dt

