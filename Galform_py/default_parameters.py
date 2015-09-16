import numpy as np

class Default_Parameters:
    '''This class contains a set of default model parameters for the galform_py code.
    These default parameters are read in by input_parameters.py,
    where different models with variations of model parameters can be defined.'''

    # Default model name for output storage purposes
    model_name = "default"

    # Input merger trees file name
    input_file = "merger_tree_test.hdf5"

    n_linalg_variables = 16 # Array size for linear algebra calculations

    # Cosmological parameters
    h = 0.704; omm = 0.272; oml = 0.728; omb = 0.0455
    sigma8 = 0.810 # power spectrum amplitude

    PKfile = "pk_MillGas_norm.dat" # path to file used to calculate halo concentrations -note this code actually uses the .spline version of this file
    
    # path for merger tree data to be read by build_merger_tree.py
    input_merger_tree_path = "/gpfs/data/d72fqv/PythonSAM/input_data/merger_trees.hdf5"

    # Factor halo mass has to grow by to trigger a formation event
    fform = 2.0

    # Minimum formation mass, this should be chosen to be of order the minimum possible halo mass in the simulation, i.e. 20 x particle mass
    # Recommended value is half this value
    mform_min = 1.282732254*10**10 # Msun

    #If true, only update vhalo for halo formation events. 
    #If false, update vhalo on every substep 
    propagate_vhalo = True # 
    
    # Corresponding option for halo concentration, strc
    propagate_strc = True

    fcore = 0.1 # Core radius for hot gas profile in units of halo virial radius

    # Choose a cooling model
    #Cooling_Model = "toy" - infall timescale = age of universe
    Cooling_Model = "simple" # Lgalaxies cooling model (see Guo et al. 2011)
    #Cooling_Model = "icool3" # Adapted version of Galform's icool 3 model (Bower et al. 2006)

    gas_rotation_profile = "flat" # Assumed circular velocity profile for hot gas. "flat", "const_j" are the 2 options

    # If true, perform self consistent infalling angular momentum calculation. If False, use Galform icool3 scheme.
    angular_momentum_conservation = True
    # Note, code seems unstable with this set to False

    ############### Halo Angular momentum

    spin_med = 0.039		# median halo spin
    spin_disp = 0.53             # dispersion in ln(halo spin)
    print "setting spin_disp to 0"
    spin_disp = 0.0


    # Subhalo dynamical friction timescale (proxy until properly implemented)
    #tdfriction = 0.0
    #tdfriction = 200. # Gyr
    tdfriction = 3.

    
    ############### Choose whether to have decoupled gas and stellar disk sizes
    decoupled_disks = False

    ############### Choose whether to use the old Galform SF law where sfr = Mgas / tau_star and tau_star = 1/epsilon_Star * tdyn_disk * (vdisk/200.)**Parameter.alphastar
    # or to use the new sf law where sfr = Mmolecular * nu_sf and Mmolecular is calculated using Blitz & Rosolowsky (2006) empirical law.
    new_sf_law = False

    ####### Choose whether to use the angular momentum transfere scheme where specific angular momentum of gas forming stars (Jfr) = total cold gas specific angular momentum 
    # or new scheme where Jfr is calculated self-consistently from the assumed angular momentum profile of molecular star forming gas in the disk
    new_Jfr_scheme = False

    ######## Old SF law parameters
    epsilon_Star = 0.002857
    alphastar = -1.5

    ######## New SF law parameters

    #nu_sf is the inverse of the depletion timescale of molecular gas (H2+He) in units of Gyr-1
    #within observational errors (Bigiel et al. 2010), and for a Milky-way Xco, nu_sf=[0.2-1.7] Gyr-1, with a median
    #value of nu_sf=0.5. Here we use the median value of observations as the default value.
    nu_sf = 0.5 # Gyr^-1

    # Velocity dispersion of gas in galaxy disks. Consistent with observations from Leroy et al. (2008)
    sigma_gas = 10 # kms^-1

    # Ratio of the vertical scale height to the radial disk scale length for stars in galaxy disks
    f_star = 7.3 # Consistent with observations of local spiral galaxies (Kregel et al. 2002)

    #Po corresponds to the normalisation in the Pressure-H2/HI relation from Leroy et al. (2008) for their combined sample.
    Pext0 = 17000. # in units of kb cm^3 K^-1

    #beta_press: power-law index in the Pressure-H2/HI relation from Leroy et al. (2008) for their combined sample.               
    beta_press = 0.8

    ############ Choose whether to use old SN scheme where mass loading = angular momentum loading and are simply calculated as beta = (vdisk/vhot)**-alphahot
    new_sne_scheme = False # Note that to use the new SNe fb scheme, you need to also have new_sf_law = True

    ###### Choose whether to use the old angular momentum loading (Beta_J) = mass loading scheme 
    # or new scheme where Beta_J is calculated self-consistently from the assumed angular momentum profile of star forming gas in the disk
    new_betaJ_scheme = False # Note, this option only has an effect if new_sf_law = True

    #### Old SNe scheme parameters
    alphahot = 3.2
    vhot = 425 # kms-1
    
    #### New SNe scheme parameters

    
    # Normalisation factor in Claudia Lagos' mass loading parametrisation for disk annuli
    h_gas_norm = 5.05 * 10**-3 # kpc
    #print "hacking norm"
    #h_gas_norm = 15 * 10**-3 #kpc
    #h_gas_norm = 5.5 * 10**-3

    # Power law index in Claudia Lagos' mass loading parametrisation for disk annuli
    beta_index = 1.8

    # Gas reincorporation timescale parameter dM_res/dt = -M_res / tau_ret where tau_ret = t_dyn / alpha_reheat
    alpha_reheat = 1.26027

    R = 0.44 # recycled fraction
    p = 0.021 # yield

    
    # Set numerical tolerances for the ODE solver. Error < (rtol y) + atol
    rtol = 10**-3 # Relative error tolerance

    atol_mhalo = 10.**9 # Absolute error tolerance for dark matter halo mass
    atol_mbaryon = 10.**7 # Absolute error tolerance for baryonic mass
    atol_mmetal = atol_mbaryon *0.02 # Absolute error tolerance for metal mass
    atol_Jhalo = 10.**11 # Absolute error tolerance for halo angular momentum
    atol_Jbaryon = 10.**9 # Absolute error tolerance for baryon angular momentum
    
    #                      mhalo       mhot          mcold         mres          mstar         mZhot        mZcold       mZres        mZstar       Jhalo       Jhot          Jcold         Jres          Jstar         Mnotional     Jnotional
    atol_array = np.array([atol_mhalo, atol_mbaryon, atol_mbaryon, atol_mbaryon, atol_mbaryon, atol_mmetal, atol_mmetal, atol_mmetal, atol_mmetal, atol_Jhalo, atol_Jbaryon, atol_Jbaryon, atol_Jbaryon, atol_Jbaryon, atol_mbaryon, atol_Jbaryon])

    if len(atol_array) != n_linalg_variables:
        print "Error: absolute numerical tolerance array must be the same length as y array (number of reservorirs)"
        quit()

