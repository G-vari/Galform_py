import numpy as np; from input_parameters import *; from halo_properties_routines import *; from size_routines import *; from cooling_routines import *; from utilities_cosmology import *; import time; from non_linear_sf_fb_laws import *
import scipy.integrate; from reservoir_routines import *

def Initialise_y_total(halo):
    '''Initialise the total y array for this step (includes halo y and all subhalo y)'''
    
    # List of the halo and subhalo instances
    nodes = [halo] + halo.subhaloes

    y_total = np.zeros(Parameter.n_linalg_variables * len(nodes))

    for inode, node in enumerate(nodes):
        
        # Initialise y (mass/J/metals) array if this node has no progenitors
        if not hasattr(node, 'y'):
            node.y = np.zeros(Parameter.n_linalg_variables)

        ind1 = inode *Parameter.n_linalg_variables
        ind2 = inode *Parameter.n_linalg_variables + Parameter.n_linalg_variables

        y_total[ind1:ind2] = node.y

    return y_total

def Deconstruct_y_total(halo, y_total):
    '''Separate the total y array (which includes all halo and subhalo y) into each halo and subhalo'''
    
    # Now construct a list for the satellites
    y_nodes = []
    nodes = [halo] + halo.subhaloes

    for inode, node in enumerate(nodes):
        ind1 = inode *Parameter.n_linalg_variables
        ind2 = inode *Parameter.n_linalg_variables + Parameter.n_linalg_variables

        y_nodes.append(y_total[ind1:ind2])

    return y_nodes

def Cosmological_Gas_Accretion_Term(node):
    '''Compute the mass and angular momentum of material being accreted onto a halo for the first time'''

    # Set of coefficients that transforms the accretion_array into the correct quantities (i.e. metal mass and baryon mass)
    #                             halo, hot gas,              hot metal,                                  haloJ  hotJ                  Mnotional    Jnotional
    accretion_coeffs = np.array([ 1,    Constant.fb, 0, 0, 0, Constant.fb*Constant.Z_primordial, 0, 0, 0, 1,     Constant.fb, 0, 0, 0, Constant.fb, Constant.fb])
    
    accretion_term = node.hfr_array * accretion_coeffs

    return accretion_term

def Infall_Term(t, y, node, accretion_term, reincorporation_term):
    '''Compute the rates at which hot halo gas either cools or freefalls onto a galaxy disk'''

    #return np.zeros_like(y)

    mhot = Mhot(y) # hot gas mass

    if Jhot_f(y) ==0.0 or np.isnan(Jhot_f(y)) or Jhot_f(y)<0.0:
        print "yikes", Jhot_f(y), Mhot(y), node
        
        progen = node.progenitors[0]
        while len(progen.progenitors)>0:
            progen = progen.progenitors[0]
        descendant = progen.descendant
        while True:
            descendant = descendant.descendant
            
            print np.log10(Jhot_f(descendant.y)), np.log10(Mhot(descendant.y))
            
            if descendant == node:
                break

        quit()

    # Only do infall if there is a non-negligible amount of gas and angular momentum in the hot gas halo
    if mhot <  Parameter.atol_mbaryon or Jhot_f(y) < Parameter.atol_Jbaryon:
        return np.zeros_like(y)

    # Implement extremely simply cooling model for testing purposes
    if Parameter.Cooling_Model == "toy":
        if mhot > 0:
            Mdot_infall = Mhot(y) / t
            MZdot_infall = MZhot(y) / t
            Jdot_infall = Jhot_f(y) / t
        else:
            Mdot_infall = 0.0
            MZdot_infall = 0.0
            Jdot_infall = 0.0
    
    # Lgalaxies cooling model, see Guo et al. (2011)
    elif Parameter.Cooling_Model == "simple":
        
        Jhot = Jhot_f(y) # Angular momentum in the hot gas halo in Msun kpc kms^-1
        mZhot = MZhot(y) # metal mass in the hot gas halo in Msun
        Zhot = mZhot / mhot

        r_cool = Calculate_Cooling_Radius_LGalaxies(mhot, node.Substep.rchalo, node.Substep.vchalo, Zhot, node.Substep.T_dyn(), logLam_grid, logT_grid, Z_FeH_grid)

        # Calculate infall rate
        # No cooling
        if r_cool == 0.0:
            Mdot_infall = 0.0; Jdot_infall = 0.0; MZdot_infall = 0.0
        # Calculate effective cooling timescale
        else:
            # Radiative cooling limited regime
            if r_cool < node.rchalo:
                tau_infall = node.Substep.T_dyn() * node.Substep.rchalo / r_cool
            # Freefall limited regime
            else:
                tau_infall = node.Substep.T_dyn()

            Mdot_infall = mhot / tau_infall
            MZdot_infall = mZhot / tau_infall

            # Self-consistent scheme
            if Parameter.angular_momentum_conservation:
                Jdot_infall = Jhot / tau_infall
            # Alternative scheme where jhot always = jhalo when computing infall rates (where j is specific angular momentum)
            else:
                jhalo = Jhalo_f(y) / Mhalo(y)
                Jdot_infall = Mdot_infall * jhalo

                # Deal with rare cases where this scheme leads to numerical problems by reducing Jhot to < 0.
                # This happens when jhot starts to drop below jhalo
                jhot = Jhot_f(y) / Mhot(y) # Hot gas specific angular momentum
                if jhot < jhalo * 0.5:
                    print "warning, reduced Jdot_infall by", Jhot / tau_infall / Jdot_infall, " for node", node
                    Jdot_infall = Jhot / tau_infall

        #if Jhot_f(y)/Mhot(y) < 50:
        #    print node, Jhot_f(y)/Mhot(y), np.log10(Jhot_f(y)), jhot/jhalo, np.log10(Mhot(y)), t

    # Only have accretion if there is a non zero hot gas mass in this node
    elif Parameter.Cooling_Model == "icool3":
        
        a_substep = a_Universe(t, Parameter.omm, Parameter.h) # Calculate the expansion factor at this substep (i.e. at time, t)
        z_substep = 1./a_substep -1. # Redshift at this substep

        mZhot = MZhot(y) # mass in hot metals
        Zhot = mZhot / mhot # hot gas metallicity
        m_notional = Mnotional(y) # mass in the notional gas profile
        Jhot = Jhot_f(y)

        mhalo = node.Substep.mchalo # Halo mass in Msun
        rhalo = node.Substep.rchalo # Halo virial radius in kpc
        vhalo = node.Substep.vchalo # Halo circular velocity at the virial radius in kms^-1
        tform = node.Substep.tform # Time since the last formation event in Gyr
        strc = node.Substep.strc # Dimensionless inverse NFW halo concentration
        tdyn = node.Substep.T_dyn() # Halo dynamical timescale in Gyr
        rcore = node.Substep.Rcore() # Core radius for the hot gas halo in kpc
        spin = node.Substep.spin # Dimensionless halo spin parameter

        # Compute rate of change of various halo properties
        mhalo_dot = node.hfr # Halo mass accretion rate in Msun Gyr^-1
        rhalo_dot = Calculate_Halo_Virial_Radius_Rate(z_substep,mhalo,mhalo_dot) # Rate of change of halo virial radius in kpc Gyr^-1
        rcore_dot = Parameter.fcore * rhalo_dot # Rate of change of the hot gas core radius in kpc Gyr^-1
        vhalo_dot = Calculate_Halo_Virial_Velocity_Rate(z_substep,mhalo,mhalo_dot)
        strc_dot = 0.0 # Assume halo concentration constant over timestep for time being. This won't be the case for propagate_vhalo = False

        # Calculate the radius from which gas has had time to cool since t=tform
        r_cool = Calculate_Cooling_Radius(m_notional, rhalo, vhalo, Zhot, tform, logLam_grid, logT_grid, Z_FeH_grid)

        # Calculate the radius from which gas has had time to freefall since t=tform
        r_ff = Calculate_Freefall_Radius_Over_Rvir(tform, tdyn, strc) * rhalo

        # Infall radius is the minimum of these two and the node virial radius
        r_infall = min(r_cool, r_ff, rhalo)


        if r_infall == r_cool and r_cool > 0.0:

            # In tcool equation, I'm using the notional profile for mhot and mhot_dot but the actual hot gas profile for mZhot and mZhot_dot
            # At the moment, we just use Zhot_dot =0 as a guess when computing rcool_dot.
            # To do this properly would require some sort of iterative procedure, as Zhot_dot depends on rcool_dot depends on Zhot_dot

            Zhot_dot_guess = 0.0

            # For the notional profile, the mass rate is just the cosmological infall rate, which is constant across the timestep
            m_notional_dot = Mhot(accretion_term) # Msun Gyr^-1

            r_cool_dot = Calculate_Cooling_Radius_Rate(rhalo, rhalo_dot, rcore, rcore_dot, r_cool, m_notional, m_notional_dot, vhalo, vhalo_dot, Zhot, Zhot_dot_guess)
            r_infall_dot = r_cool_dot

        elif r_infall == r_ff and r_ff>0.0:

            strc_dot = 0.0 # Assume halo concentration constant over timestep for time being. This won't be the case for propagate_vhalo = False

            rff_dot = Calculate_Freefall_Radius_Rate(strc, strc_dot, r_ff, rhalo, rhalo_dot, mhalo, mhalo_dot)
            r_infall_dot = rff_dot

        else:
            # In this case r_infall == r_halo
            r_infall_dot = 0.0
            # Note this isn't physically very sensible! Pretty much icool=3 is a complete mess of logic.

        #Now we need to calculate dm_cool/dt, dJcool/dt and dMZ_cool/dt

        if r_infall > 0.0:

            # Calculate mass infall rate
            Mdot_infall = Calculate_Mass_Infall_Rate(r_infall, r_infall_dot, m_notional, rhalo, rcore, mhot)

            # Calculate angular momentum infall rate

            # In this case, assume hot gas always has same specific angular momentum as the dark matter halo (as in old Galform)
            if not Parameter.angular_momentum_conservation:
                Jhot_pseudo = Jhalo * mhot / Jhalo_f(y)
                Jdot_infall = Calculate_J_Infall_Rate(r_infall, Mdot_infall, rhalo, rcore, Jhot_pseudo, mhot)
            # In this case, use the correct hot gas mass
            else:
                Jdot_infall = Calculate_J_Infall_Rate(r_infall, Mdot_infall, rhalo, rcore, Jhot, mhot)

            # Mdot_infall will == infinity if r_infall > rhalo. In this case we set infall term such that it cancels accretion onto the hot gas halo and re-routes it to the cold gas disk instead
            # We also set existing hot gas profile to deplete over a short timescale
            # Note it should perhaps sit in the hot gas reservoir for a freefall time. But there is no easy way to implement this into the differential equations.
            if Mdot_infall == np.inf or Jdot_infall == np.inf:

                # Deplete remaining hot gas profile over a timestep down to the floor allowed by the absolute error tolerance, atol, used in the ODE integrator
                if mhot > 10 * Parameter.atol_mbaryon:
                    Mdot_depletion_term = mhot/node.level.dt # Msun Gyr^-1
                else:
                    Mdot_depletion_term = 0.0

                if Jhot > 10 * Parameter.atol_Jbaryon:
                    Jdot_depletion_term = Jhot/node.level.dt # Msun Gyr^-1
                else:
                    Jdot_depletion_term = 0.0

                if mZhot > 10 * Parameter.atol_mmetal:
                    MZdot_depletion_term = mZhot/node.level.dt # Msun Gyr^-1
                else:
                    MZdot_depletion_term = 0.0

                Mdot_infall =  Mhot(accretion_term) + Mhot(reincorporation_term) + Mdot_depletion_term
                MZdot_infall =  MZhot(accretion_term) + MZhot(reincorporation_term) + MZdot_depletion_term
                Jdot_infall = Jhot_f(accretion_term) + Jhot_f(reincorporation_term) + Jdot_depletion_term

            # Otherwise, calculate metal infall and angular momentum rates as normal
            else:
                # Calculate metal mass infall rate
                MZdot_infall = Mdot_infall * Zhot  # calculate metal infall rate

                if np.isnan(Mdot_infall):
                    print "Uh oh", r_infall, r_infall_dot, M_notional, node.rchalo, rcore, yj[1]+kn[1]*dt_rk
                    quit()

            #print np.log10(Mdot_infall), r_infall/rhalo, np.log10(mhot), t

            #if np.log10(Jhot) < 9.:
            #print "mdot_infall, Jdot_infall", np.log10(Mdot_infall), np.log10(Jdot_infall), "mhot, Jhot", np.log10(mhot), np.log10(Jhot_f(y)), "rinfall, rhalo=", r_infall, rhalo

        else:
            Mdot_infall = 0.0
            MZdot_infall = 0.0
            Jdot_infall = 0.0

    else:
        print "Error: Cooling_Model = ", Parameter.Cooling_Model, " has not been implemented"
        quit()

    infall_term = np.array([0.0, -Mdot_infall, Mdot_infall, 0.0, 0.0, -MZdot_infall, MZdot_infall, 0.0, 0.0, 0.0, -Jdot_infall, Jdot_infall, 0.0, 0.0, 0.0, 0.0]) # infall rates for different galaxy components

    return infall_term

def Calculate_Galaxy_Sizes(t, y, node):
    """Compute the size and circular velocity of different galaxy components.
    Note that the actual calculations are performed in an external module, size_routines.py"""

    Jcold_disk = Jcold_f(y) # Angular momentum of the cold gas disk (atomic+molecular) in Msun kpc kms^-1
    Jstar_disk = Jstar_f(y) # Angular momentum of the stellar disk
    mcold_disk = Mcold(y) # Mass in the cold gas disk (atomic + molecular) in Msun
    mstar_disk = Mstar(y) # Mass in the stellar disk

    mdisk = mcold_disk + mstar_disk
    Jdisk = Jcold_disk + Jstar_disk

    # Deal with rare cases (that come about because of "features" of the merger trees where the baronic specific angular momentum is much larger than the halo specific angular momentum
    jhalo = Jhalo_f(y) / Mhalo(y)
    if mdisk > 0.0:
        jdisk = Jdisk / mdisk
    else:
        jdisk = 0.0
    if jdisk/jhalo > 10:
        print "Warning: very high disk specific angular momentum, turning of size calculation for this galaxy"
        node.Substep.rdisks = [0.0, 0.0]
        node.Substep.vdisks = [0.0, 0.0]

    # Only compute sizes if there is a non-neglible amount of mass in the disk
    elif mdisk > Parameter.atol_mbaryon and Jdisk > Parameter.atol_Jbaryon:
        
        mhalo = Mhalo(y)      # Halo mass
        strc = node.Substep.strc # Dimensionless inverse NFW halo concentration
        rhalo = node.Substep.rchalo # Halo virial radius in kpc

        # In this case, we do the standard Galform calculation where the gas and stellar disks are considered as one entity
        if not Parameter.decoupled_disks:

            rdisk, vdisk = Calculate_Disk_Size_Dynamic_Halo(Jdisk, mdisk, strc, mhalo, rhalo) # Compute galaxy size and circular velocity
            rdisks = [rdisk, rdisk]; vdisks = [vdisk,vdisk] # Gas and stellar disk both have the same size and circular velocity
        
        else:
            Jdisks = [Jcold_disk , Jstar_disk]
            mdisks = [mcold_disk , mstar_disk]

            # Determine if the decoupled disk calculation should be performed.
            # First check if there is non-negligible mass/angular momentum in both disk components
            ok_decoupled = (mcold_disk > Parameter.atol_mbaryon) & (mstar_disk > Parameter.atol_mbaryon) & (Jstar_disk > Parameter.atol_Jbaryon) & (Jcold_disk > Parameter.atol_Jbaryon)
            
            # Then check that both disks have a specific angular momentum that is not too high compared to the dark matter halo. Only bother with this check if the first one passes
            if ok_decoupled:
                ok_decoupled = (Jcold_disk/mcold_disk / jhalo < 10) & (Jstar_disk/mstar_disk / jhalo < 10)
                if not ok_decoupled:
                    print "Warning: one of the disks had a very specific angular momentum. Coupling the two disks together for size calculation", Jcold_disk/mcold_disk / jhalo, Jstar_disk/mstar_disk / jhalo
            
            # If checks were passed, perform the normal decoupled disk size calculation
            if ok_decoupled:               
                rdisks, vdisks = Calculate_Decoupled_Disk_Sizes(Jdisks, mdisks, strc, mhalo, rhalo) # Sizes and circular velocities of [cold gas disk, stellar disk] in [kpc, kpc] and [kms^-1, kms^-1]

            # Otherwise, couple the two disks together
            else:
                rdisk, vdisk = Calculate_Disk_Size_Dynamic_Halo(Jdisk, mdisk, strc, mhalo, rhalo) # Compute galaxy size and circular velocity
                rdisks = [rdisk, rdisk]; vdisks = [vdisk,vdisk] # Gas and stellar disk both have the same size and circular velocity


        node.Substep.rdisks = rdisks
        node.Substep.vdisks = vdisks

    # Set rdisk and vdisk to 0 if there is no galaxy
    else:
        node.Substep.rdisks = [0.0, 0.0]
        node.Substep.vdisks = [0.0, 0.0]

def Star_Formation_Feedback_Term(t, y, node):
    '''Call either linear star formation law (sfr \propto mcold_disk, like old Galform) or non-linear SF law (Lagos et al. 2011)'''

    # Only have star formation and SNe feedback if there is mass in the cold gas disk
    mcold_disk = Mcold(y) # Total mass in the cold gas disk in Msun

    # Cold gas disk size in kpc
    rdisk_gas = node.Substep.rdisks[0] # Use gas disk radial scalelength. Note if Parameter.Decoupled_Disks = False, rdisk_gas is equal to rdisk_star
    vdisk_gas = node.Substep.vdisks[0] # Use gas disk circular velocity (kms^-1)

    # Only do star formation and feedback if there is a non-negligible amount of mass and angular momentum in the gas disk
    if mcold_disk > Parameter.atol_mbaryon and Jcold_f(y) > Parameter.atol_Jbaryon:

        # Don't do star formation/feedback if the disk has zero size
        if vdisk_gas == 0.0 or rdisk_gas == 0.0:
            print "Warning: disk has mcold = ", mcold_disk, "but rdisk = ", rdisk_gas, "and vdisk = ", vdisk_gas, ". Turning off SF and fb for this galaxy"
            sf_fb_term = np.zeros_like(y)
            return sf_fb_term

        # Call non linear SF law (as in Lagos et al. 2011)
        if Parameter.new_sf_law:           

            mstar_disk = Mstar(y) # Stellar mass of the disk in Msun
            Jcold_disk = Jcold_f(y) # Angular momentum of the gas disk in Msun kms^-1 kpc
            mZcold_disk = MZcold(y) # Mass in metals in the cold gas disk in Msun
            Zcold_disk = mZcold_disk / mcold_disk # Metallicity of the cold gas disk
            rdisks = node.Substep.rdisks # Radial scalelength of [cold gas, stellar] disks in kpc
            vdisks = node.Substep.vdisks # Circular velocity of [cold gas, stellar] disks in kms^-1

            sf_fb_term = nonlinear_sfr_fb_term(rdisks, vdisks, mcold_disk, mstar_disk, Zcold_disk, Jcold_disk) # Msun Gyr^-1
    
        # Use linear SF law (as in Cole et al. 2000)
        else:

            # Note again that if Parameter.Decoupled_Disks = False, rdisk_gas is equal to rdisk_star
            tdyn_disk = rdisk_gas * Constant.kpc / (vdisk_gas*Constant.kms) / Constant.Gyr # Disk dynamical timescale in Gyr
            
            # Compute star formation timescale
            tau_star = tdyn_disk /Parameter.epsilon_Star * (vdisk_gas/200.)**Parameter.alphastar

            # Compute dimenionless mass loading factor for SNe feedback
            beta = (Parameter.vhot/vdisk_gas)**Parameter.alphahot

            R = Parameter.R # Recycled fraction
            p = Parameter.p # yield

            # Matrix of coefficients for linear star formation and feedback
            #                  halo,hotgas, cold gas,             reservoir gas,stars,hot metal,cold metal,           reservoir metal, stellar metal, haloJ hotJ coldJ                 resJ        stellarJ mnotional Jnotional      
            coeffs = np.array([[ 0, 0,      0,                    0,            0,  0,          0,                    0,               0,             0,    0,   0,                    0,           0,      0,        0],       # 0 halo
                               [ 0, 0,      0,                    0,            0,  0,          0,                    0,               0,             0,    0,   0,                    0,           0,      0,        0],       # 1 hot gas
                               [ 0, 0,      -(1-R+beta)/tau_star, 0,            0,  0,          0,                    0,               0,             0,    0,   0,                    0,           0,      0,        0],       # 2 cold gas
                               [ 0, 0,      beta/tau_star,        0,            0,  0,          0,                    0,               0,             0,    0,   0,                    0,           0,      0,        0],       # 3 reservoir gas
                               [ 0, 0,      (1-R)/tau_star,       0,            0,  0,          0,                    0,               0,             0,    0,   0,                    0,           0,      0,        0],       # 4 stars    
                               [ 0, 0,      0,                    0,            0,  0,          0,                    0,               0,             0,    0,   0,                    0,           0,      0,        0],       # 5 hot metal
                               [ 0, 0,      p/tau_star,           0,            0,  0,          -(1-R+beta)/tau_star, 0,               0,             0,    0,   0,                    0,           0,      0,        0],       # 6 cold metal
                               [ 0, 0,      0,                    0,            0,  0,          beta/tau_star,        0,               0,             0,    0,   0,                    0,           0,      0,        0],       # 7 reservoir metal
                               [ 0, 0,      0,                    0,            0,  0,          (1-R)/tau_star,       0,               0,             0,    0,   0,                    0,           0,      0,        0],       # 8 stellar metal
                               [ 0, 0,      0,                    0,            0,  0,          0,                    0,               0,             0,    0,   0,                    0,           0,      0,        0],       # 9 halo angular momentum
                               [ 0, 0,      0,                    0,            0,  0,          0,                    0,               0,             0,    0,   0,                    0,           0,      0,        0],       # 10 hot gas angular momentum
                               [ 0, 0,      0,                    0,            0,  0,          0,                    0,               0,             0,    0,   -(1-R+beta)/tau_star, 0,           0,      0,        0],       # 11 cold gas angular momentum
                               [ 0, 0,      0,                    0,            0,  0,          0,                    0,               0,             0,    0,   beta/tau_star,        0,           0,      0,        0],       # 12 reservoir angular momentum
                               [ 0, 0,      0,                    0,            0,  0,          0,                    0,               0,             0,    0,   (1-R)/tau_star,       0,           0,      0,        0],       # 13 stellar angular momentum
                               [ 0, 0,      0,                    0,            0,  0,          0,                    0,               0,             0,    0,   0,                    0,           0,      0,        0],       # 14 Notional hot gas profile mass
                               [ 0, 0,      0,                    0,            0,  0,          0,                    0,               0,             0,    0,   0,                    0,           0,      0,        0]])      # 15 Notional hot gas profile J
            
            sf_fb_term = np.dot(coeffs,y) # Msun Gyr^-1

    else:
        sf_fb_term = np.zeros_like(y)
   
    return sf_fb_term

def Gas_Reincorporation_Term(t, y, node):
    '''Compute the rate with which mass,metals and angular momentum are reincorporated from the ejected gas reservoir into the hot gas halo'''

    tdyn = node.Substep.T_dyn() # Halo dynamical timescale in Gyr

    if tdyn <= 0.0:
        print "Error: Cannot calculate a return timescale for ejected gas when the halo dynamical time is = ", tdyn
        print node.y
        print node.mchalo, node.rchalo, node.vchalo
        quit()

    # Calculate the timescale over which gas in the ejected reservoir is reincorporated back into the hot gas halo reservoir
    tau_return = tdyn / Parameter.alpha_reheat

    mdot_return = Mres(y)/tau_return
    mZdot_return = MZres(y)/tau_return
    Jdot_return = Jres_f(y)/tau_return

    reinc_term = np.array([ 0, mdot_return, 0, -mdot_return, 0, mZdot_return, 0, -mZdot_return, 0, 0, Jdot_return, 0, -Jdot_return, 0, 0, 0])

    return reinc_term

def Calculate_Stripping_Term(t, y, halo):

    stripping_term = np.zeros_like(y)

    return stripping_term

def Galaxy_Formation_Function(t, y_total, halo):
    '''Calculate F(y,t) for arbitrary y and t. This routine calls seperate 
    physics functions to get the different terms that appear in F(y,t)'''
    
    # Create substep instances for the halo and it's subhaloes containing all the relevant halo property information evaluated at this substep
    # By substep I mean the halo properties evaluated at time, t
    halo.Create_Halo_Substep(t)
    for subhalo in halo.subhaloes:
        subhalo.Create_Subhalo_Substep(t)

    # Break apart y_total into halo and subhaloes
    y_nodes = Deconstruct_y_total(halo, y_total)
    nodes = [halo] + halo.subhaloes

    galaxy_formation_function_total = np.zeros_like(y_total)

    # Calculate mass/angular momentum exchange rates for each halo/subhalo individually
    for inode, node in enumerate(nodes):

        ind1 = inode *Parameter.n_linalg_variables
        ind2 = inode *Parameter.n_linalg_variables + Parameter.n_linalg_variables
        
        y = y_total[ind1:ind2]

        # Calculate cosmological gas accretion term
        accretion_term = Cosmological_Gas_Accretion_Term(node)

        # If the halo has no baryons, only compute accretion term
        if Mbaryon(node.y) == 0.0:
            galaxy_formation_function_total[ind1:ind2] = accretion_term

        # Normal step
        else:
            # Calculate gas reincorporation term
            reinc_term = Gas_Reincorporation_Term(t,y,node)

            # Calculate infall (cooling/freefall) term
            infall_term = Infall_Term(t,y,node,accretion_term,reinc_term)

            # Calculate disk sizes for this substep
            Calculate_Galaxy_Sizes(t, y, node)

            # Calculate combined star formation and feedback term
            sf_fb_term = Star_Formation_Feedback_Term(t, y, node)

            # Calculate stripping term
            stripping_term = Calculate_Stripping_Term(t,y,node)

            galaxy_formation_function = accretion_term + infall_term + sf_fb_term + reinc_term + stripping_term

            galaxy_formation_function_total[ind1:ind2] = galaxy_formation_function

    return galaxy_formation_function_total

def Evolve_System(halo, y_total):
    '''This is where the numerical integration occurs.
    We want to solve dy/dt = F(y,t) where y is a vector containing
    all the information (mass/angular momentum) we need to track to
    the next step'''
    
    t0 = halo.level.t # Start of the timestep in Gyr
    dt = halo.level.dt # Timestep in Gyr
    t1 = t0 + dt # End of the timestep in Gyr

    # For the first time a halo appears in the merger tree, I have made the decision to only allow cosmological accretion.
    # In effect, this is the same as just making the halo appear with mhot = mbaryons at the start of the first timestep where the halo was idenfied
    # In these cases, we don't need to call the differential equation solver because dy/dt is independent of both y and t
    # (Note that if we did call the solver we might confuse it for this very reason)
    if np.sum(y_total) == 0.0:
        y_total_next = np.zeros_like(y_total)
        nodes = [halo] + halo.subhaloes
        for inode, node in enumerate(nodes):
            ind1 = inode *Parameter.n_linalg_variables
            ind2 = inode *Parameter.n_linalg_variables + Parameter.n_linalg_variables
            accretion_term = Cosmological_Gas_Accretion_Term(node) # Comsological accretion onto the node in Msun Gyr^-1 (for mass and metal reservoirs)
            y_total_next[ind1:ind2] = accretion_term * dt
            

    # Otherwise, call scipy ordinary differential equation (ODE) solver
    else:

        # Get numerical tolerances for the ODE solver from input_parameter file
        atol_array = Parameter.atol_array
        n_nodes = 1 + len(halo.subhaloes)
        rtol = Parameter.rtol # Relative tolerance
        atol_total = np.tile(atol_array, n_nodes) # Absolute tolerance array for halo + subhaloes

        # Choose an ODE solver and specify various options
        # Available integrator options are "vode", "zvode", "lsoda", "dopri5", "dop853"

        integrator = integrate.ode(Galaxy_Formation_Function).set_integrator('vode', atol=atol_total, rtol=rtol)

        #integrator = integrate.ode(Galaxy_Formation_Function).set_integrator('vode')
        #integrator = integrate.ode(Galaxy_Formation_Function).set_integrator('dopri5')
        #integrator = integrate.ode(Galaxy_Formation_Function).set_integrator('dop853')
        #integrator = integrate.ode(Galaxy_Formation_Function).set_integrator('lsoda')
        #integrator = integrate.ode(Galaxy_Formation_Function).set_integrator('zvode')

        # Set initial condtions
        integrator.set_initial_value(y_total, t0).set_f_params(halo)
    
        # Perform integration
        while integrator.successful() and integrator.t < t1:
            integrator.integrate(t1)

        if not integrator.successful():
            print "Error: integrator was not able to successfully complete this timestep"
            quit()

        y_total_next = integrator.y

    # Consistency check for negative mass/J
    if y_total_next.min() < 0.0:
        print "Error, negative mass or angular momentum"
        print "try reducing atol or rtol"
        print y_total_next
        quit()

    return y_total_next

def Update_Halo(halo, y_total_next):
    '''Pass on baryon mass and angular momentum to descendant haloes on the next step'''

    # Break y_total_next array between halo and subhaloes
    y_nodes = Deconstruct_y_total(halo, y_total_next)

    # Update the halo and subhaloes on the next step
    nodes = [halo] + halo.subhaloes

    for inode, node in enumerate(nodes):

        if not hasattr(node.descendant, 'y'):
            node.descendant.y = np.zeros(Parameter.n_linalg_variables)

        node.descendant.y += y_nodes[inode]

def Save_Halo_Properties(halo):
    '''Save galaxy sizes/circular velocities etc for the main progenitor at each step'''
    
    nodes = [halo] + halo.subhaloes
    
    for node in nodes:
        if hasattr(node,"isMainProgenitor_Final") and hasattr(node,"Substep") and hasattr(node.Substep,"rdisks"):
            node.descendant.rdisks = node.Substep.rdisks
            node.descendant.vdisks = node.Substep.vdisks
