import numpy as np; import default_parameters

# Define your desired model parameters here. Refer to the Default_Parameters class (further down) to see the full range of model parameters and their meanings
Parameter = default_parameters.Default_Parameters()

Parameter.model_name = "test2"
Parameter.input_file = "merger_tree_test.hdf5"
Parameter.alphahot = 2.5


class Constant:
    '''Define constants (some physical, some specific to a given cosmology) to be used in other parts of the code'''

    G = 6.67259e-11 # m^3/kg/s^2
    cm = 0.01 #m
    km = 1000.0 #m
    kms = 1000.0 # ms^-1
    pc = 3.0856775807 * 10**16 # m
    kpc = pc * 10**3 # m
    Mpc = pc * 10**6 # m
    Msun=1.9891e30 # The mass of the Sun in kg
    Gyr = 3.15576*10**7 *10**9 #s
    kb = 1.3806488 * 10**-23 # m^2 kg s^-2 K^-1

    M_Atomic=1.66053873e-27 # kg
    Atomic_Mass_Hydrogen=1.00794 # in units of M_Atomic
    Atomic_Mass_Helium=4.002602 # in units of M_Atomic
    fH = 0.778 # primordial hyrdrogen abundance
    fHe = 0.222 # primordial helium abundance

    # Mean molecular weight in atomic mass units of a primordial gas
    mu_Primordial=1.0/(2.0*fH/Atomic_Mass_Hydrogen+3.0*fHe/Atomic_Mass_Helium) # Mean atomic weight

    fb = Parameter.omb / Parameter.omm # baryon fraction

    Z_primordial = 5.36*10**-10 # Mass fraction of metals in primordial plasma.
    
    H0100 = 100.0 # The Hubble constant in units of h km/s/Mpc.

    rho_crit = 3.0 * (Parameter.h * H0100 * kms / Mpc)**2 /8.0 /np.pi /G * (Mpc**3 / Msun)# Critical density of the Universe (3*H0^2/[8*PI*G]) in Msun/Mpc^3.

    m8_crit = rho_crit * Parameter.h**-2 * 4.0* np.pi * (8.0**3) /3.0 # Mass in a sphere of 8Mpc/h radius in a critical density Universe.

    Delta200 = 200.

    RDISK_HALF_SCALE=1.678346990 # The half-mass radius of an exponential disk in units of the disk scale-length.
    kstrc_1=0.6251543028 # Constant relating V_c(rdisk)^2 to GMdisk/rdisk in the disk plane.
    kstrc_2=1.191648617  # Constant relating disk angular momentum to r*V_c derived assuming a flat rotation curve


class Paths:
    '''Filenames of input data files (e.g. tabulated spherical collapse model data)'''

    # Path to the input merger tree hdf5 file
    input_path = "./input_data/" #merger_tree_test.hdf5"

    # Path to the output directory
    output_path = "./output_data/"

    # Path to input data directory (contains tabulated information e.g. spherical collapse model results)
    data_path = "./Data/"
    
    spherical_collapse = data_path + "Eke_Delta_collapse.data"
    spline_interpolation_sigma_mass = data_path + Parameter.PKfile + ".spline"
    nfw = data_path + "nfw.data"
