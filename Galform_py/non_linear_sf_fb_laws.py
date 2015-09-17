import numpy as np; from scipy import integrate; from input_parameters import *


def Sigma_Molecular_BR(r, rdisks, mcold, mstar_disk):
    """For a gas disk with an exponential profile, calculate the molecular gas mass per unit area
    at a given radius, using the empirical Blitz & Rosolowsky (2006) star formation law, with
    the implementation described in Lagos et al. (2011) 
    
    Inputs: 
         r = radius to evaluate Sigma_SFR in kpc
         rdisks = half mass radius of the [gas,stellar] disks in kpc
         mcold = total gas mass in the disk in Msun
         mstar_disk = total stellar mass in the disk in Msun
    Outputs:
         Sigma_SFR = star formation rate per unit area"""

    # Calculate radial disk scalelength
    reff_gas  = rdisks[0] / Constant.RDISK_HALF_SCALE
    reff_star = rdisks[1] / Constant.RDISK_HALF_SCALE

    # Calculate total gas surface density at radius r in Msun kpc^-2
    Sigma_Gas = mcold / (2. * np.pi * reff_gas**2) * np.exp(-r /reff_gas) 

    # Calculate stellar surface density at radius r in Msun kpc^-2
    if reff_star > 0.0:
        Sigma_Star = mstar_disk / (2. * np.pi * reff_star**2) * np.exp(-r /reff_star)
    else:
        Sigma_Star = 0.0

    # Calculate vertical scaleheight for stars in the disk
    h_star = reff_star / Parameter.f_star

    # Calculate stellar velocity dispersion in the disk
    sigma_star = max(Parameter.sigma_gas, np.sqrt(np.pi * Constant.G * h_star * Sigma_Star * Constant.Msun / Constant.kpc) / Constant.kms ) # kms^-1

    # Calculate the mid plane pressure of the disk according to the relation from Elmegreen 1993
    # Currently the unit conversions are not consistent with Claudia's code. This is problematic since I cannot be sure what units Pext0 are in based on the parameter file. To do this properly I need to email claudia
    #Pext = np.pi /2. *Constant.kb * Constant.G * Sigma_Gas * (Sigma_Gas + Parameter.sigma_gas/sigma_star * Sigma_Star * Constant.Msun / Constant.kpc**2) * Constant.Msun / Constant.kpc**2 * Constant.cm**3 # cm^-3 K
    
    #print "warning using hacked version of the pressure calculation to try and reproduce galform"
    cons1 = 4.33e-12 * 10**-6
    Pext = np.pi /2. *cons1 /Constant.kb * Constant.G * Sigma_Gas / Constant.km**2 * (Sigma_Gas + Parameter.sigma_gas/sigma_star * Sigma_Star / Constant.km**2)

    # Calculate the surface density of molecular gas in the disk at radius r in Msun kpc^-2
    Sigma_Mol = (Pext / Parameter.Pext0) ** Parameter.beta_press * Sigma_Gas / (1. + (Pext / Parameter.Pext0) ** Parameter.beta_press)

    return Sigma_Mol

def Molecular_Gas_BR_integrand(r, rdisks, mcold, mstar_disk):
    
    Sigma_Molecular = Sigma_Molecular_BR(r, rdisks, mcold, mstar_disk)

    molecular_gas_integrand = Sigma_Molecular *2 *np.pi * r # Msun kpc^-1

    return molecular_gas_integrand

def Sigma_SFR_BR(r, rdisks, mcold, mstar_disk):
    """For a gas disk with an exponential profile, calculate the star formation rate per unit area
    at a given radius. Uses the empirical Blitz & 
    Rosolowsky (2006) star formation law, with the implementation described in Lagos et al. (2011) where 
    nu_sf is constant such the SF law is linear in molecular hydrogen surface density.
    
    Inputs: 
         r = radius to evaluate Sigma_SFR in kpc
         rdisks = half mass radius of the [gas,stellar] disks in kpc
         mcold = total gas mass in the disk in Msun
         mstar_disk = total stellar mass in the disk in Msun
    Outputs:
         Sigma_SFR = star formation rate per unit area"""

    # Calculate the surface density of molecular gas
    Sigma_Mol = Sigma_Molecular_BR(r, rdisks, mcold, mstar_disk)

    # Calculate the star formation rate per unit area in Msun kpc^-2 Gyr^-1
    Sigma_SFR = Parameter.nu_sf * Sigma_Mol

    return Sigma_SFR

def SFR_BR_integrand(r, rdisks, mcold, mstar_disk):
    """For a gas disk with an exponential profile, calculate the star formation rate per unit area
    at a given radius multiplied by the circumference at that radius. Uses the empirical Blitz & 
    Rosolowsky (2006) star formation law, with the implementation described in Lagos et al. (2011) where 
    nu_sf is constant such the SF law is linear in molecular hydrogen surface density.
    
    Inputs: 
         r = radius to evaluate Sigma_SFR in kpc
         rdisks = half mass radius of the [gas,stellar] disks in kpc
         mcold = total gas mass in the disk in Msun
         mstar_disk = total stellar mass in the disk in Msun
    Outputs:
         SFR_integrand = star formation rate per unit area multiplied by circumference in Msun Gyr^-1 kpc^-1 """


    Sigma_SFR = Sigma_SFR_BR(r, rdisks, mcold, mstar_disk)

    SFR_integrand = Sigma_SFR * 2 * np.pi * r # Msun Gyr^-1 kpc^-1

    return SFR_integrand

def SFR_BR(rdisks, mcold, mstar_disk):
    """Calculate the total star formation rate of a galaxy disk using the empirical 
    Blitz & Rosolowsky (2006) star formation law.
    This function uses Romberg integration to integrate the SFR surface density over the disk
    Inputs:
           rdisks = [gas,stellar] disk half mass radius in kpc
           mcold = total gas mass in the disk in Msun
           mstar_disk = total stellar mass in the disk in Msun
    Outputs:
           sfr = star formation rate of the disk in Msun Gyr^-1"""

    # Set integration limits so that 99.95% of the total gas mass is enclosed
    rmin = 0.0
    rmax = 10.0 * np.array(rdisks).max() / Constant.RDISK_HALF_SCALE

    # Integrate the star formation rate surface density to calculate the total star formation rate
    sfr = integrate.romberg(SFR_BR_integrand, rmin, rmax, args = (rdisks, mcold, mstar_disk), tol=0.05,rtol=0.05)

    return sfr

def Molecular_Gas_Mass_BR(rdisks,mcold,mstar_disk):
    """Calculate the total molecular gas mass in a galaxy disk using the empirical 
    Blitz & Rosolowsky (2006) star formation law.
    This function uses Romberg integration to integrate the molecular gas surface density over the disk

    Inputs:
           rdisks = [gas,stellar] disk half mass radius in kpc
           mcold = total gas mass in the disk in Msun
           mstar_disk = total stellar mass in the disk in Msun
    Outputs:
           mgas_mol = molecular gas mass in Msun"""

    # Set integration limits so that 99.95% of the total gas mass is enclosed
    rmin = 0.0
    rmax = 10.0 * np.array(rdisks).max() / Constant.RDISK_HALF_SCALE

    # Integrate the molecular gas surface density to calculate the total molecular gas mass
    mgas_mol = integrate.romberg(Molecular_Gas_BR_integrand, rmin, rmax, args = (rdisks, mcold, mstar_disk), tol=0.05,rtol=0.05)

    return mgas_mol

def Jdot_star_BR_integrand(r, rdisks, mcold, mstar_disk):
    """For a given radius in a gas disk with an exponential profile, calculate that radius
    multiplied by the star formation rate per unit area at that radius and the circumference
    at that radius. Uses the empirical Blitz & Rosolowsky (2006) star formation law, with 
    the implementation described in Lagos et al. (2011) where nu_sf is constant such the SF 
    law is linear in molecular hydrogen surface density.
    
    Inputs: 
         r = radius to evaluate Sigma_SFR in kpc
         rdisks = half mass radius of the [gas, stelar] disks in kpc
         mcold = total gas mass in the disk in Msun
         mstar_disk = total stellar mass in the disk in Msun
    Outputs:
         Jdot_star_integrand = circumference x radius x star formation rate per unit area in Msun Gyr^-1"""

    Sigma_SFR = Sigma_SFR_BR(r, rdisks, mcold, mstar_disk)
    
    r_Sigma_SFR = r * Sigma_SFR # Msun Gyr^-1 kpc^-1

    Jdot_star_integrand = r_Sigma_SFR * 2 * np.pi * r # Msun Gyr^-1

    return Jdot_star_integrand

def Jdot_star_BR(rdisks, mcold, mstar_disk, vdisk_gas):
    """Calculate the rate at which angular momentum is transfered to the galaxy stellar disk 
    through star formation using the empirical Blitz & Rosolowsky (2006) star formation 
    law, assuming a flat rotation profile for the disk.

    Inputs:
           rdisks = [gas, stellar] disk half mass radius in kpc
           mcold = total gas mass in the disk in Msun
           mstar_disk = total stellar mass in the disk in Msun
           vdisk_gas = circular velocity of the gas disk
    Outputs:
           Jdot_star = dJ_star/dt = rate of change of angular momentum for the stellar component of the disk in Msun kpc kms^-1 Gyr^-1"""

    # Set integration limits so that 99.95% of the total gas mass is enclosed
    rmin = 0.0
    rmax = 10.0 * np.array(rdisks).max() / Constant.RDISK_HALF_SCALE

    Jdot_star = integrate.romberg(Jdot_star_BR_integrand, rmin, rmax, args = (rdisks, mcold, mstar_disk), tol=0.05,rtol=0.05) * vdisk_gas # Msun kpc kms^-1 Gyr^-1
    
    return Jdot_star

def Beta_Annulus(r, rdisks, mcold, mstar_disk):
    """Using the scaleheight parametrisation calculated by Claudia Lagos, calculate
    the mass loading factor in an annulus at a radius r"""
    
    # Calculate radial disk scalelengths in kpc
    reff_gas = rdisks[0] / Constant.RDISK_HALF_SCALE
    reff_star = rdisks[1] / Constant.RDISK_HALF_SCALE

    # Calculate total gas surface density at radius r in Msun kpc^-2
    Sigma_Gas = mcold / (2. * np.pi * reff_gas**2) * np.exp(-r /reff_gas) 

    # Calculate stellar surface density at radius r in Msun kpc^-2
    if reff_star > 0.0:
        Sigma_Star = mstar_disk / (2. * np.pi * reff_star**2) * np.exp(-r /reff_star)
    else:
        Sigma_Star = 0.0

    # Calculate vertical scaleheight for stars in the disk in kpc
    h_star = reff_star / Parameter.f_star

    # Calculate stellar velocity dispersion in the disk in kms^-1
    sigma_star = max(Parameter.sigma_gas, np.sqrt(np.pi * Constant.G * h_star * Sigma_Star * Constant.Msun / Constant.kpc) / Constant.kms ) # kms^-1

    # Calculate gas vertical scaleheight in kpc
    h_gas = Parameter.sigma_gas**2 * Constant.km**2 * Constant.kpc / ( np.pi * Constant.G * Constant.Msun * (Sigma_Gas + Parameter.sigma_gas / sigma_star * Sigma_Star)) # kpc

    beta_annulus = (h_gas / Parameter.h_gas_norm )**Parameter.beta_index

    #print h_gas, Parameter.h_gas_norm, beta_annulus, np.log10(Sigma_Gas), r/reff_gas

    return beta_annulus

def h_gas(r, rdisks, mcold, mstar_disk):
    """Using the scaleheight parametrisation calculated by Claudia Lagos, calculate
    the mass loading factor in an annulus at a radius r"""
    
    # Calculate radial disk scalelengths in kpc
    reff_gas = rdisks[0] / Constant.RDISK_HALF_SCALE
    reff_star = rdisks[1] / Constant.RDISK_HALF_SCALE

    # Calculate total gas surface density at radius r in Msun kpc^-2
    Sigma_Gas = mcold / (2. * np.pi * reff_gas**2) * np.exp(-r /reff_gas) 

    # Calculate stellar surface density at radius r in Msun kpc^-2
    if reff_star > 0.0:
        Sigma_Star = mstar_disk / (2. * np.pi * reff_star**2) * np.exp(-r /reff_star)
    else:
        Sigma_Star = 0.0

    # Calculate vertical scaleheight for stars in the disk in kpc
    h_star = reff_star / Parameter.f_star

    # Calculate stellar velocity dispersion in the disk in kms^-1
    sigma_star = max(Parameter.sigma_gas, np.sqrt(np.pi * Constant.G * h_star * Sigma_Star * Constant.Msun / Constant.kpc) / Constant.kms ) # kms^-1

    # Calculate gas vertical scaleheight in kpc
    h_gas = Parameter.sigma_gas**2 * Constant.km**2 * Constant.kpc / ( np.pi * Constant.G * Constant.Msun * (Sigma_Gas + Parameter.sigma_gas / sigma_star * Sigma_Star)) # kpc

    return h_gas

def Beta_BR_integrand(r, rdisks, mcold, mstar_disk):
    """Calculate the mass loading multiplied by the molecular gas surface density multiplied by the annulus
    circumference for an annulus in a galaxy disk. Uses the scaleheight parametrisation for the mass loading
    factor in annuli calculated by Claudia Lagos. Also uses the empirical Blitz & Rosolowsky (2006) star formation 
    law and assumes a flat rotation profile for the disk."""

    beta_integrand = 2 * np.pi * r * Sigma_Molecular_BR(r, rdisks, mcold, mstar_disk) * Beta_Annulus(r, rdisks, mcold, mstar_disk) # Msun kpc^-1

    #print r/rdisks[0], np.log10(Sigma_Molecular_BR(r, rdisks, mcold, mstar_disk)), Beta_Annulus(r, rdisks, mcold, mstar_disk), beta_integrand *rdisks[0] / mcold

    return beta_integrand

def Beta_disk_BR(rdisks, mcold, mstar_disk, mgas_mol):
    """Calculate the mass loading factor for the galaxy disk using the 
    empirical Blitz & Rosolowsky (2006) star formation law, assuming a flat rotation 
    profile for the disk.

    Inputs:
           rdisks = [gas, stellar] disk half mass radius in kpc
           mcold = total gas mass in the disk in Msun
           mstar_disk = total stellar mass in the disk in Msun
           mgas_mol = total molecular gas mass in Msun Gyr^-1
    Outputs:
           Jdot_star = dJ_star/dt = rate of change of angular momentum for the stellar component of the disk in Msun kpc kms^-1 Gyr^-1"""

    # Set integration limits so that 99.95% of the total gas mass is enclosed
    rmin = 0.0
    rmax = 10.0 * np.array(rdisks).max() / Constant.RDISK_HALF_SCALE

    # Calculate the integral of the mass loading factor in annuli, weighted by molecular gas surface density in the disk
    Sigma_Molecular_Weighted_Beta = integrate.romberg(Beta_BR_integrand, rmin, rmax, args = (rdisks, mcold, mstar_disk), tol=0.05,rtol=0.05) # Msun

    #print ""
    #print "beta_ann at rhalf"
    #Beta_BR_integrand(rdisks[0],rdisks, mcold, mstar_disk)
    #print ""
    
    # Renormalise to calculate the average mass loading factor of the entire disk
    Beta_disk = Sigma_Molecular_Weighted_Beta / mgas_mol

    return Beta_disk

def Beta_J_integrand1(r, rdisks, mcold, mstar_disk):
    """Calculate the mass loading multiplied by the molecular gas surface density multiplied by 
    radius multiplied by the annulus circumference for an annulus in a galaxy disk. Uses the 
    scaleheight parametrisation for beta_annulus calculated by Claudia Lagos. Also uses the 
    empirical Blitz & Rosolowsky (2006) star formation law and assumes a flat rotation profile 
    for the disk."""

    beta_J_integrand1 = Beta_BR_integrand(r, rdisks, mcold, mstar_disk) * r # Msun

    return beta_J_integrand1

def Beta_J_integrand2(r, rdisks, mcold, mstar_disk):
    
    beta_J_integrand2 = Molecular_Gas_BR_integrand(r, rdisks, mcold, mstar_disk) * r # Msun 

    return beta_J_integrand2

def J_molecular_BR(rdisks, mcold, mstar_disk, vdisk_gas):
    
    # Set integration limits so that 99.95% of the total gas mass is enclosed
    rmin = 0.0
    rmax = 10.0 * np.array(rdisks).max() / Constant.RDISK_HALF_SCALE

    J_mol = integrate.romberg(Beta_J_integrand2, rmin, rmax, args = (rdisks, mcold, mstar_disk), tol=0.05,rtol=0.05) * vdisk_gas # Msun kpc kms^-1
    
    return J_mol

def Beta_J_disk_BR(rdisks, mcold, mstar_disk, Jcold_molecular, vdisk_gas):
    """Calculate the angular momentum loading factor for the galaxy disk using the 
    empirical Blitz & Rosolowsky (2006) star formation law, assuming a flat rotation 
    profile for the disk.

    Inputs:
           rdisks = [gas, stellar] disk half mass radius in kpc
           mcold = total gas mass in the disk in Msun
           mstar_disk = total stellar mass in the disk in Msun
           mgas_mol = total molecular gas mass in Msun Gyr^-1
    Outputs:
           Beta_J_disk = dJ_star/dt = rate of change of angular momentum for the stellar component of the disk in Msun kpc kms^-1 Gyr^-1"""

    # Set integration limits so that 99.95% of the total gas mass is enclosed
    rmin = 0.0
    rmax = 10.0 * np.array(rdisks).max() / Constant.RDISK_HALF_SCALE

    # Calculate the integral of r multiplied by the mass loading factor in annuli, weighted by molecular gas surface density in the disk
    Sigma_Molecular_Weighted_Beta = integrate.romberg(Beta_J_integrand1, rmin, rmax, args = (rdisks, mcold, mstar_disk), tol=0.05,rtol=0.05) # Msun kpc
    
    Norm = Jcold_molecular / vdisk_gas

    # Renormalise to calculate the average mass loading factor of the entire disk
    Beta_J_disk = Sigma_Molecular_Weighted_Beta / Norm

    return Beta_J_disk



def nonlinear_sfr_fb_term(rdisks, vdisks, mcold, mstar_disk, Zcold, Jcold, mcold_start, Jcold_start, mZcold_start, dt):
    """Calculate mass/J exchange rates using the BR06 SF law"""

    '''#Norm_Sigma_0 = 2.90000E+03
    #Norm_fgas_Sigma_0 =   1.50000E-01

    Norm_Sigma_0 = 1600
    Norm_fgas_Sigma_0 = 0.12

    Norm_hgas_0 = 15
    Norm_fgas_hgas_0 = 0.02

    fgas_track = mcold / (mstar_disk+mcold) # dimensionless
    sdens_gas_track = mcold * np.exp(-Constant.RDISK_HALF_SCALE) / (2*np.pi*np.square(rdisks[0]*10**3/Constant.RDISK_HALF_SCALE)) # Msun pc^-2

    beta_track = (sdens_gas_track/Norm_Sigma_0)**-0.6 * (fgas_track/Norm_fgas_Sigma_0)**0.8

    sigma_gas = 10
    sigma_gas = 10.0 #kms-1
    G = 6.67384 * 10 **-11 # m3 kg-1 s-2
    Msun = 1.989*10**30 # kg
    pc = 3.08567758 * 10** 16 # m
    G = G * Msun / pc / (10**6) # Msun^-1 pc^-1 km^2 s^-2


    rconv = Constant.RDISK_HALF_SCALE
    sdens_star_track = mstar_disk * np.exp(-rconv) / (2*np.pi*np.square(rdisks[0]*10**3/rconv)) # Msun pc^-2

    sigma_star_track = max(np.sqrt(np.pi*G*0.14*rdisks[0]*10**3*sdens_star_track),sigma_gas) # kms-1
    
    h_gas_track = sigma_gas**2 / (np.pi * G * (sdens_gas_track + sigma_gas*sdens_star_track/sigma_star_track)) # pc
            
    beta_track2 = (h_gas_track/Norm_hgas_0)**1.1 * (fgas_track/Norm_fgas_hgas_0)**0.4'''


    if rdisks[0] == 0.0:
        nonlinear_sfr_fb_term = np.zeros(Parameter.n_linalg_variables)
        return nonlinear_sfr_fb_term, [0.0, 0.0]

    # Circular velocity of the gas disk at the half mass radius
    vdisk_gas = vdisks[0]

    # Calculate the total star formation rate of the galaxy disk
    sfr = SFR_BR(rdisks, mcold, mstar_disk) # Msun Gyr^-1
    
    # Calculate molecular gas mass. Note, this only works if the SFR law is linear in molecular gas surface density
    mcold_molecular = sfr / Parameter.nu_sf
    

    # Compute Jfr, the rate with which angular momentum in the cold gas is transfered to the stellar disk
    if not Parameter.new_Jfr_scheme:
        # As in old Galform, assume that the angular momentum transfer rate to the stellar disk is
        # just equal to the specific angular momentum of the total gas disk * sfr
        Jfr = sfr * Jcold / mcold
        Jcold_molecular = J_molecular_BR(rdisks, mcold, mstar_disk, vdisk_gas)
    else:
        # Calculate angular momentum transfer rate to the stellar disk from specific angular momentum profile of molecular gas
        
        Jfr = Jdot_star_BR(rdisks, mcold, mstar_disk, vdisk_gas) # Msun kpc kms^-1 Gyr^-1
        
        # Calculate molecular gas angular momentum. Note, this only works if the SFR law is linear in molecular gas surface density
        Jcold_molecular = Jfr / Parameter.nu_sf

    '''print "mmol/mcold, Jmol/Jcold, jmol, jcold, Jfr_new/Jfr_old"
    print mcold_molecular / mcold, J_molecular / Jcold, J_molecular / mcold_molecular, Jcold / mcold, Jfr/(sfr*Jcold/mcold)
    print J_molecular * 0.5, Jfr
    print mcold_molecular * 0.5, sfr
    exit()'''

    # Calculate the total mass loading factor of the galaxy disk
    if not Parameter.new_sne_scheme:
        beta = (Parameter.vhot/vdisk_gas)**Parameter.alphahot
        beta_J = beta

    else:
        # First calculate the molecular gas mass in the disk for use as a a normalisation factor when computing mass/J loading integrals

        # Compute the average mass loading factor of the disk
        beta = Beta_disk_BR(rdisks, mcold, mstar_disk, mcold_molecular)

        # Compute the average angular momentum loading factor of the disk
        if Parameter.new_betaJ_scheme:
            beta_J = Beta_J_disk_BR(rdisks, mcold, mstar_disk, Jcold_molecular, vdisk_gas)
        else:
            beta_J = beta



    '''r2 = np.arange(0,10*rdisks[0],0.01)
    x = np.zeros_like(r2)
    beta_a = np.zeros_like(r2)
    sigma_mol = np.zeros_like(r2)
    for n in range(len(r2)):
        x[n] = Beta_BR_integrand(r2[n],rdisks, mcold, mstar_disk)
        beta_a[n] = Beta_Annulus(r2[n], rdisks, mcold, mstar_disk)
        sigma_mol[n] = Molecular_Gas_BR_integrand(r2[n], rdisks, mcold, mstar_disk)

    print h_gas_track
    print h_gas(rdisks[0], rdisks, mcold, mstar_disk) * 1000

    beta_int = Beta_disk_BR(rdisks, mcold, mstar_disk, mcold_molecular)

    print beta_track, beta_track2, beta_int, beta

    import pylab as py
    py.plot(r2, np.log10(x), c="b")
    py.plot(r2, np.log10(beta_a), c="r")
    py.plot(r2, np.log10(sigma_mol), c="g")
    py.axvline(rdisks[0])
    py.axhline(np.log10(beta_track))
    py.show()

    print "here2"
    print beta, beta_int, beta_track, beta_track2, beta_int / beta_track
    print ""'''

    ######### Limit mass / angular momentum loading factors to prevent numerical errors #############

    '''if beta > 1000:
        beta = 1000
    if beta_J > 1000:
        beta = 1000'''

    ######### Limit cold gas mass/J depletion timescales to prevent numerical errors ################

    limited = False

    tau_star = mcold / sfr
    sfr_unlimited = sfr
    tau_effective = tau_star / (1-Parameter.R+beta)  # prevent t_effective from dropping below the timestep dize. This stops errors appearing in first few timesteps

    if tau_effective < dt:
        sfr = 0.999 * mcold / dt * (1-Parameter.R+beta)
        limited = True
        #print "limited sfr1"

    tau_Jstar = Jcold / Jfr
    tau_Jeffective = tau_Jstar / (1-Parameter.R+beta_J)

    if tau_Jeffective < dt:
        Jfr = 0.999 * Jcold / dt * (1-Parameter.R+beta_J)
        #print "limited J1"
        limited = True

    # limit to prevent cold gas depletion rate from being high enough to deplete the cold gas reservoir that is present at the start of the full RK step
    # Note, this isn't very rigorous and should probably be improved in the future
    if mcold_start > 0.0:
        mcold_dot_limit = mcold_start / (dt*2.0)
        sfr_limit = mcold_dot_limit / (1-Parameter.R+beta)

        if sfr > sfr_limit:
            sfr = 0.999 * sfr_limit
            limited = True
    else:
        sfr = 0.0 # prevent star formation until there is gas present in the disk at the start of the full RK step

    if mcold_start > 0.0:
        mZcold_dot_limit = mZcold_start / (dt*2.0)
        sfrZ_limit = mZcold_dot_limit / (1-Parameter.R+beta)

        if sfr * Zcold > sfrZ_limit:
            sfr = 0.999 * sfrZ_limit / Zcold
            limited = True

    if Jcold_start > 0.0:
        Jcold_dot_limit = Jcold_start / (dt*2.0)
        Jfr_limit = Jcold_dot_limit / (1-Parameter.R+beta_J)

        if Jfr > Jfr_limit:
            Jfr = 0.999 * Jfr_limit
            limited = True
    else:
        Jfr = 0.0 # prevent star formation until there is gas present in the disk at the start of the full RK step

    #print (Parameter.vhot/vdisk_gas)**Parameter.alphahot , Beta_disk_BR(rdisks, mcold, mstar_disk, mcold_molecular)

    R = Parameter.R
    p = Parameter.p   

    if Parameter.print_convergence and limited:
        print "limited sfr by a factor", sfr_unlimited / sfr

    #                                 halo  hot  cold             res       stars      hotZ coldZ                          resZ            stellarZ         haloJ hotJ coldJ             resJ        stellarJ
    nonlinear_sfr_fb_term = np.array([ 0.0, 0.0, -(1-R+beta)*sfr, beta*sfr, (1-R)*sfr, 0.0, -(1-R+beta)*sfr*Zcold + p*sfr, beta*sfr*Zcold, (1-R)*sfr*Zcold, 0.0,  0.0, -(1-R+beta_J)*Jfr, beta_J*Jfr, (1-R)*Jfr])

    return nonlinear_sfr_fb_term, [mcold_molecular,Jcold_molecular]


if __name__=='__main__':
    rdisks = [5., 5.]
    vdisks = [200., 200.]
    mcold = 10**10
    mstar_disk = 10**10.5
    Zcold = 0.02
    Jcold = 10
    mcold_start = 10
    Jcold_start = 10
    mZcold_start = 10
    dt = 10

    nonlinear_sfr_fb_term(rdisks, vdisks, mcold, mstar_disk, Zcold, Jcold, mcold_start, Jcold_start, mZcold_start, dt)
