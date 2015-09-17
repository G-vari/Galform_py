import numpy as np; from scipy.optimize import brentq; import time; from utilities_interpolation import *; from input_parameters import *
import scipy.optimize

print "Tabulating size quantities"
size_t0 = time.time()

def Tabulate_x_NFW():
    ''' Tabulate x for a given grid in y where y = x * ( ln(1+x) -x/(1+x) )'''

    # I'm not sure why Galform does it this way round, possibly to ensure even y spacing
    # In this case I switched to y log spacing so that we reduce the interpolation time per timestep

    NT=101
    ymax=10.0
    ymin=0.001
    
    logymax = np.log(ymax); logymin = np.log(ymin)
    
    y_NFW_root = lambda x, ytab: x*(np.log(1+x) -x/(x+1)) - ytab

    xtab = np.zeros(NT)
    ytab = np.zeros(NT)

    for n in range(NT):
        ytab[n]=np.exp(logymin+(logymax-logymin)*n/(NT-1.))
        xtab[n]=brentq(y_NFW_root,0,10,args=(ytab[n]))

    return xtab, ytab

xtab, ytab = Tabulate_x_NFW()

def fraction_NFW_m_interior_given_r(s,anfw):
    '''Calculate the fraction of mass of a NFW halo interior to a radius r where s = r / rhalo / a_nfw'''

    fanfw = 1./(np.log(1.+1./anfw)-1./(1.+anfw)) # normalisation term

    if s < 0.01:
        s2 = s**2 # Use Taylor expansion to avoid rounding error inaccuracies leading to negative interior mass
        other_term = (0.5-(2.0/3.0)*s+0.75*s2)*s2
    else:
        other_term = np.log(1.0+s)-s/(s+1.0)

    fraction_interior = fanfw * other_term

    return fraction_interior

def Calculate_Mintr_given_Mintr_r(mrhalo,mhalo,rhalo,strc):
    ''' Calculate M(<r) of a NFW halo where M(<r)r is equal to a given input value'''

    f_anfw=1./(np.log(1.0+1.0/strc)-1.0/(1.0+strc))
    
    const=mrhalo/(mhalo*rhalo*f_anfw*strc) # dimensionless

    # Now solve const = x * ( ln(1.0+x) - x/(1+x) ) where x = r/(rhalo*anfw_local).

    if const < 0.001 or const > 10.0:
        y_NFW_root = lambda x, ytab: x*(np.log(1+x) -x/(x+1)) - ytab

        ymax = 15.0

        if y_NFW_root(ymax,const) < 0:
            print "Warning in root finding. had to increase upper bound from ymax = ", ymax

            ymax = 100000000.0

            if y_NFW_root(ymax,const) < 0:

                import pdb
                pdb.set_trace()

                print "Error: ymax = ", ymax, " still too small"

                print "mrhalo, mhalo, rhalo, strc"
                print mrhalo, mhalo, rhalo, strc
                print y_NFW_root(0,const), y_NFW_root(ymax,const)
                quit()

        x = brentq(y_NFW_root,0,ymax,args=(const))
    else:
        x=Linear_Interpolation(ytab,xtab,[const])[0]

    # Return mass interior to this radius.      
    minterior = mhalo * fraction_NFW_m_interior_given_r(x,strc) # Msun

    return minterior

def Calculate_c(J_comp, mass_comp, rhalo, mhalo, type_comp):
    '''Calculate cdisk or cbulge = (J/m)**2/G in kpc Msun
    type_comp = "disk" for a disk
    type_comp = "bulge" for a bulge/spheroid'''
    
    if type_comp == "disk":
        c_comp =(J_comp/mass_comp/Constant.kstrc_2 *Constant.kms)**2/Constant.G  * Constant.kpc / Constant.Msun # kpc Msun - This is equal to M_halo(<rdisk) rdisk
    else:
        c_comp =(J_comp/mass_comp *Constant.kms)**2/Constant.G  * Constant.kpc / Constant.Msun

    if c_comp > rhalo * mhalo:
        print "Error, specific angular momentum of the disk/bulge is too large compared to the size/mass of the halo"
        print "c = ", c_comp, "rhalo x mhalo=", mhalo*rhalo
        print type_comp, "J", J_comp, "mass", mass_comp
        print "rhalo", rhalo, "mhalo", mhalo
        
        import pdb
        pdb.set_trace()

        quit()

    return c_comp

def Calculate_Disk_Size_No_Selfgravity(Jdisk, mdisk, strc, mhalo, rhalo):
    #Jdisk is total angular momentum

    if mdisk == 0.:
        return 0.0, 0.0

    cdisk=Calculate_c(Jdisk, mdisk, rhalo, mhalo, "disk") # kpc Msun - This is equal to M_halo(<rdisk) rdisk

    # Return mass interior to rdisk
    mhalo0_disk = Calculate_Mintr_given_Mintr_r(cdisk,mhalo,rhalo,strc) # Msun

    rdisk=cdisk/mhalo0_disk # kpc
    vdisk = np.sqrt(Constant.G *cdisk *Constant.kpc *Constant.Msun) / (rdisk *Constant.kpc) / Constant.kms # kms^-1

    return rdisk, vdisk

def Calculate_Bulge_Size_No_Selfgravity(Jbulge, mbulge, strc, mhalo, rhalo):
    
    if mbulge == 0.:
        return 0.0

    cbulge= Calculate_c(Jbulge, mbulge, rhalo, mhalo, "bulge") # kpc Msun - This is equal to M_halo(<rbulge) rbulge

    # Return mass interior to rbulge
    mhalo0_bulge = Calculate_Mintr_given_Mintr_r(cbulge,mhalo,rhalo,strc) # Msun

    rbulge=cbulge/mhalo0_bulge # kpc
    vbulge = np.sqrt(Constant.G * cbulge * Constant.kpc *Constant.Msun) / (rbulge *Constant.kpc) / Constant.kms # kms^-1

    return rbulge, vbulge

def M_r_interior_static(rdisk,rhalo,mdisk,mhalo,anfw,fhalo):
    '''Compute, for a static halo with no bulge, the total mass interior to a radius r multiplied by that radius
    Inputs: rdisk = disk radius / kpc
            rhalo = halo virial radius / kpc
            mdisk = gas+stellar mass of disk / Msun
            mhalo = total halo mass / Msun
            anfw = inverse NFW halo concentration
            fhalo = fraction of total halo mass that isn't in the disk
    Outputs: M_r = M_tot(<r) * r / (kpc Msun)'''
    
    s = rdisk / rhalo / anfw

    fraction = fraction_NFW_m_interior_given_r(s,anfw)
    
    mhalo_int_r = mhalo * fraction * fhalo

    M_r = rdisk*(mhalo_int_r + Constant.kstrc_1*mdisk)
    
    return M_r

def Root_eqn_static(rdisk,rhalo,mdisk,mhalo,anfw,fhalo, cdisk):
    '''Compute the equation cdisk = r * M_total(<r)'''
    
    M_r = M_r_interior_static(rdisk,rhalo,mdisk,mhalo,anfw,fhalo)

    value = cdisk - M_r

    return value

def Calculate_Disk_Size_Static_Halo(Jdisk, mdisk, strc, mhalo, rhalo):
    '''Compute disk size for a static halo with no bulge
    Use numerical root finding to solve for centrifugal equilibrium of the disk under disk + halo gravity
    Inputs: Jdisk = angular momentum of the disk in Msun kpc kms^-1
            mdisk = total disk mass in Msun
            strc = inverse halo concentration = a_nfw
            mhalo = total halo mass in Msun
            rhalo = virial radius in kpc
    Outputs:
            rdisk = disk half mass radius in kpc
            vdisk = corresponding disk cicular velocity at the half mass radius in kms^-1'''

    if mdisk == 0.:
        return 0.0, 0.0

    cdisk = Calculate_c(Jdisk, mdisk, rhalo, mhalo, "disk") # kpc Msun - This is equal to M_halo(<rdisk) rdisk

    fhalo = (mhalo - mdisk) / mhalo

    # If this line throws up an error, try increasing the maximum for f(b) i.e. make it bigger than 10
    rdisk = brentq(Root_eqn_static, 0, 100, args=(rhalo,mdisk,mhalo,strc,fhalo,cdisk)) #kpc

    vdisk = np.sqrt(Constant.G *cdisk *Constant.kpc *Constant.Msun) / (rdisk *Constant.kpc) / Constant.kms # kms^-1

    return rdisk, vdisk



def Root_eqn_dynamic(rdisk, cdisk, mhalo, rhalo, strc, fhalo, mdisk):
    '''Returns, for a given guess of rdisk, the discrepancy with respect to the equation:
    c_disk = r_disk ( M_halo(<rdisk) + k_strc1 * M_disk )'''

    # Compute M_halo,0 (<r_0,d) r_0,d
    mrhalo0_disk = cdisk - (Constant.kstrc_1 -0.5) * rdisk * mdisk

    # Compute M_halo,0 (<r_0,d) 
    mhalo0_disk = Calculate_Mintr_given_Mintr_r(mrhalo0_disk,mhalo,rhalo,strc)

    # Compute M_halo (<r_disk)
    mhalo_disk = mhalo0_disk * fhalo

    # Function = 0 once r_disk has been correctly iterated towards
    root_eqn = cdisk - rdisk * (mhalo_disk + Constant.kstrc_1 * mdisk)

    return root_eqn

def Calculate_Disk_Size_Dynamic_Halo(Jdisk, mdisk, strc, mhalo, rhalo):
    '''Compute disk size for a contracting halo with no bulge
    Use numerical root finding to solve for centrifugal equilibrium of the disk under disk + halo gravity
    This calculation includes contraction of the halo in response to disk gravity, assuming specific 
    angular momentum of shells of halo material is conserved through the contraction process.

    Inputs: Jdisk = angular momentum of the disk in Msun kpc kms^-1
            mdisk = total disk mass in Msun
            strc = inverse halo concentration = a_nfw
            mhalo = total halo mass in Msun
            rhalo = virial radius in kpc
    Outputs:
            rdisk = disk half mass radius in kpc
            vdisk = corresponding disk cicular velocity at the half mass radius in kms^-1'''

    if mdisk == 0.:
        return 0.0, 0.0

    cdisk= Calculate_c(Jdisk, mdisk, rhalo, mhalo, "disk") # kpc Msun - This is equal to M_halo(<rdisk) rdisk

    fhalo = (mhalo - mdisk) / mhalo

    # Compute the maximum bounding guess for rdisk. Above this value, M(<r_0,d) r_0,d becomes negative
    rdisk_guess_max = 0.99 * cdisk / (Constant.kstrc_1 -0.5) / mdisk

    # If this line throws up an error, try increasing the maximum for f(b) i.e. make it bigger than 10
    rdisk = brentq(Root_eqn_dynamic, 0, rdisk_guess_max, args=(cdisk,mhalo,rhalo,strc,fhalo,mdisk)) #kpc

    if rdisk == 0.0:
        print "Warning, size calculation returned rdisk = 0.0 when mdisk = ", mdisk
        return 0.0, 0.0

    vdisk= np.sqrt(Constant.G *cdisk *Constant.kpc *Constant.Msun) / (rdisk *Constant.kpc) /Constant.kms # kms^-1

    return rdisk, vdisk

def Vector_Root_eqn_Decoupled_Disks(rdisks, cdisks, mhalo, rhalo, strc, fhalo, mdisks):

    rdisk_gas = rdisks[0]; rdisk_star = rdisks[1] # kpc
    cdisk_gas = cdisks[0]; cdisk_star = cdisks[1] # kpc Msun
    mdisk_gas = mdisks[0]; mdisk_star = mdisks[1] # Msun

    # Code will crash if root finding algorithm guesses a negative size. Just return a very large number if this happens.
    if rdisk_star < 0.0 or rdisk_gas < 0.0:
        return 10.**20

    # Calculate M_d,gas(<r_d,star) and M_d,star(<r_d,gas)
    mdisk_gas_int_rstar = mdisk_gas  * (1 - (1 + Constant.RDISK_HALF_SCALE * rdisk_star / rdisk_gas) * np.exp(-Constant.RDISK_HALF_SCALE * rdisk_star / rdisk_gas)) # Msun
    mdisk_star_int_rgas = mdisk_star * (1 - (1 + Constant.RDISK_HALF_SCALE * rdisk_gas / rdisk_star) * np.exp(-Constant.RDISK_HALF_SCALE * rdisk_gas / rdisk_star)) # Msun

    # Compute the M_halo,0 (<r_0,d,gas) r_0,d,gas  &  M_halo,0 (<r_0,d,star) r_0,d,star
    mrhalo0_gas = max(cdisk_gas - 2 * (Constant.kstrc_1 -0.5) * rdisk_gas * ( 0.5 * mdisk_gas + mdisk_star_int_rgas ) , 0.)
    mrhalo0_star = max(cdisk_star - 2 * (Constant.kstrc_1-0.5) * rdisk_star * ( 0.5 * mdisk_star + mdisk_gas_int_rstar ), 0.)

    if mrhalo0_star > 10**30:
        print "yikes"
        print cdisk_star, rdisk_star, mdisk_star, mdisk_gas_int_rstar
        exit()

    # Compute M_halo,0 (<r_0,d) for both the gas and stellar disks
    mhalo0_gas = Calculate_Mintr_given_Mintr_r(mrhalo0_gas,mhalo,rhalo,strc)
    mhalo0_star = Calculate_Mintr_given_Mintr_r(mrhalo0_star,mhalo,rhalo,strc)

    # Compute M_halo (<r_disk) for both the gas and stellar disks
    mhalo_gas = mhalo0_gas * fhalo
    mhalo_star = mhalo0_star * fhalo

    # Functions that will both = 0 once r_disk,gas and r_disk,star have been correctly iterated towards
    root_eqn_gas  = cdisk_gas  - rdisk_gas  * (mhalo_gas  + Constant.kstrc_1 * mdisk_gas  + 2 * Constant.kstrc_1 * mdisk_star_int_rgas)
    root_eqn_star = cdisk_star - rdisk_star * (mhalo_star + Constant.kstrc_1 * mdisk_star + 2 * Constant.kstrc_1 * mdisk_gas_int_rstar) 

    return [root_eqn_gas, root_eqn_star]

def Calculate_Decoupled_Disk_Sizes(Jdisks, mdisks, strc, mhalo, rhalo):
    '''Compute the sizes of decoupled gas and stellar disks when there is no bulge
    Use numerical vector root finding to solve for centrifugal equilibrium of the
    two disks under the disks+halo gravity. This calculation includes contraction
    of the halo in response to disk gravity, assuming specific angular momentum
    of shells of halo material is conserved through the contraction process.

    Inputs: Jdisks = list of the angular momentum of the gas disk[0] and stellar disk[1] in Msun kpc kms^-1
            mdisks = total gas [0] and stellar [1] disk mass in Msun
            strc = inverse halo concentration = a_nfw
            mhalo = total halo mass in Msun
            rhalo = virial radius in kpc
    Outputs:
            rdisks = gas [0] and stellar [1] disk half mass radius in kpc
            vdisks = corresponding disk cicular velocities at the half mass radius in kms^-1'''

    mdisk_gas = mdisks[0]; mdisk_star = mdisks[1]
    Jdisk_gas = Jdisks[0]; Jdisk_star = Jdisks[1]
    mdisk = mdisk_gas + mdisk_star
    Jdisk = Jdisk_gas + Jdisk_star

    if mdisk_gas == 0. and mdisk_star == 0.0:
        return [0.0, 0.0], [0.0, 0.0]

    elif mdisk_gas == 0.:
        rdisk_star, vdisk_star = Calculate_Disk_Size_Dynamic_Halo(Jdisk_star, mdisk_star, strc, mhalo, rhalo)
        return [0.0, rdisk_star], [0.0, vdisk_star]
    
    elif mdisk_star == 0.:
        rdisk_gas, vdisk_gas = Calculate_Disk_Size_Dynamic_Halo(Jdisk_gas, mdisk_gas, strc, mhalo, rhalo)
        return [rdisk_gas, 0.0], [vdisk_gas, 0.0]
       
    # As an initial guess of the decoupled sizes of the two disks, use the no baryon self-gravity calculation and assume that the two disks are combined into 1 disk with a single scalelength
    else:
        rdisk_guess, vdisk_guess = Calculate_Disk_Size_No_Selfgravity(Jdisk, mdisk, strc, mhalo, rhalo)

    cdisk_gas = Calculate_c(Jdisk_gas, mdisk_gas, rhalo, mhalo, "disk") # kpc Msun - This is equal to M_halo(<rdisk_gas) rdisk_gas  
    cdisk_star = Calculate_c(Jdisk_star, mdisk_star, rhalo, mhalo, "disk") # kpc Msun - This is equal to M_halo(<rdisk_star) rdisk_star
    cdisks = [cdisk_gas, cdisk_star]

    fhalo = (mhalo - mdisk) / mhalo   

    solution = scipy.optimize.root(Vector_Root_eqn_Decoupled_Disks, x0 = [rdisk_guess, rdisk_guess], args=(cdisks,mhalo,rhalo,strc,fhalo,mdisks), jac=None, method='hybr')

    rdisk_gas = solution.x[0] # kpc
    rdisk_star = solution.x[1] # kpc

    if rdisk_gas == 0.0 or rdisk_star == 0.0:
        print "Warning, size calculation returned rdisk_gas or rdisk_star = 0.0 when mdisks = ", mdisks
        return [0.0, 0.0], [0.0,0.0]

    vdisk_gas = np.sqrt(Constant.G *cdisk_gas *Constant.kpc *Constant.Msun) / (rdisk_gas *Constant.kpc) /Constant.kms # kms^-1
    vdisk_star = np.sqrt(Constant.G *cdisk_star *Constant.kpc *Constant.Msun) / (rdisk_star *Constant.kpc) /Constant.kms # kms^-1

    return [rdisk_gas,rdisk_star], [vdisk_gas,vdisk_star]

def Vector_Root_eqn_Decoupled_Disks_plus_Bulge(rgal, cgal, mhalo, rhalo, strc, fhalo, mgal):
    '''Return the function "c - r ( M_halo(<r) + k * M_disk_gas
    '''

    rdisk_gas = rgal[0]; rdisk_star = rgal[1]; rbulge = rgal[2] # kpc
    cdisk_gas = cgal[0]; cdisk_star = cgal[1]; cbulge = cgal[2] # kpc Msun
    mdisk_gas = mgal[0]; mdisk_star = mgal[1]; mbulge = mgal[2] # Msun

    # Calculate M_d,gas(<r_d,star) and M_d,star(<r_d,gas)
    if mdisk_gas > 0.0:
        mdisk_gas_int_rstar = mdisk_gas  * (1 - (1 + Constant.RDISK_HALF_SCALE * rdisk_star / rdisk_gas) * np.exp(-Constant.RDISK_HALF_SCALE * rdisk_star / rdisk_gas)) # Msun
        mdisk_gas_int_rbulge  = mdisk_gas  * (1 - (1 + Constant.RDISK_HALF_SCALE * rbulge / rdisk_gas) * np.exp(-Constant.RDISK_HALF_SCALE * rbulge / rdisk_gas)) # Msun
    else:
        mdisk_gas_int_rstar = 0.0
        mdisk_gas_int_rbulge = 0.0

    if mdisk_star > 0.0:
        mdisk_star_int_rgas = mdisk_star * (1 - (1 + Constant.RDISK_HALF_SCALE * rdisk_gas / rdisk_star) * np.exp(-Constant.RDISK_HALF_SCALE * rdisk_gas / rdisk_star)) # Msun
        mdisk_star_int_rbulge = mdisk_star * (1 - (1 + Constant.RDISK_HALF_SCALE * rbulge / rdisk_star) * np.exp(-Constant.RDISK_HALF_SCALE * rbulge / rdisk_star)) # Msun
    else:
        mdisk_star_int_rgas = 0.0
        mdisk_star_int_rbulge = 0.0

    # Compute the M_halo,0 (<r_0,d,gas) r_0,d,gas ,   M_halo,0 (<r_0,d,star) r_0,d,star & M_halo,0 (<r_0,bulge)
    mrhalo0_gas = max(cdisk_gas - 2 * (Constant.kstrc_1 -0.5) * rdisk_gas * ( 0.5 * mdisk_gas + mdisk_star_int_rgas ) , 0.)
    mrhalo0_star = max(cdisk_star - 2 * (Constant.kstrc_1-0.5) * rdisk_star * ( 0.5 * mdisk_star + mdisk_gas_int_rstar ), 0.)
    mrhalo0_bulge = np.copy(cbulge)

    # Compute M_halo,0 (<r_0,d) for both the gas and stellar disks and the bulge
    mhalo0_gas = Calculate_Mintr_given_Mintr_r(mrhalo0_gas,mhalo,rhalo,strc)
    mhalo0_star = Calculate_Mintr_given_Mintr_r(mrhalo0_star,mhalo,rhalo,strc)
    mhalo0_bulge = Calculate_Mintr_given_Mintr_r(mrhalo0_bulge,mhalo,rhalo,strc)

    # Compute M_halo (<r_disk) for both the gas and stellar disks and the bulge
    mhalo_gas = mhalo0_gas * fhalo
    mhalo_star = mhalo0_star * fhalo
    mhalo_bulge = mhalo0_bulge * fhalo

    # Calculate M_bulge(r_disk) for both the gas and stellar disks
    if mdisk_star > 0.0:
        mbulge_int_rdisk_star = max(cdisk_star/rdisk_star - (mhalo_star + Constant.kstrc_1 * mdisk_star  + Constant.kstrc_1 * 2 * mdisk_gas_int_rstar),0.)
    else:
        mbulge_int_rdisk_star = 0.0
    
    if mdisk_gas > 0.0:  
        mbulge_int_rdisk_gas  = max(cdisk_gas /rdisk_gas -  (mhalo_gas  + Constant.kstrc_1 * mdisk_gas   + Constant.kstrc_1 * 2 * mdisk_star_int_rgas),0.)
    else:
        mbulge_int_rdisk_gas = 0.0

    # Functions that will both = 0 once r_disk,gas and r_disk,star have been correctly iterated towards
    root_eqn_gas  = cdisk_gas  - rdisk_gas  * (mhalo_gas  + Constant.kstrc_1 * mdisk_gas  + 2 * Constant.kstrc_1 * mdisk_star_int_rgas + mbulge_int_rdisk_gas)
    root_eqn_star = cdisk_star - rdisk_star * (mhalo_star + Constant.kstrc_1 * mdisk_star + 2 * Constant.kstrc_1 * mdisk_gas_int_rstar + mbulge_int_rdisk_star) 
    root_eqn_bulge= cbulge     - rbulge     * (mhalo_bulge+ mdisk_star_int_rbulge + mdisk_gas_int_rbulge + 0.5 * mbulge)

    return [root_eqn_gas, root_eqn_star, root_eqn_bulge]

def Calculate_Decoupled_Disk_Sizes_plus_Bulge(Jgal, mgal, strc, mhalo, rhalo):
    '''Compute the sizes of decoupled gas and stellar disks and a bulge
    Use numerical vector root finding to solve for centrifugal equilibrium of the
    two disks and bulge under the disks+bulge+halo gravity. This calculation includes contraction
    of the halo in response to disk and bulge gravity, assuming specific angular momentum
    of shells of halo material is conserved through the contraction process.

    Inputs: Jgal = list of the angular momentum of the gas disk[0], stellar disk[1] and pseudo-angular momentum of the bulge[2] in Msun kpc kms^-1
            mgal = total gas disk [0], stellar disk [1] and bulge [2] mass in Msun
            strc = inverse halo concentration = a_nfw
            mhalo = total halo mass in Msun
            rhalo = virial radius in kpc
    Outputs:
            rgal = gas disk [0], stellar disk [1] and bulge half mass radii in kpc
            vgal = corresponding disk and bulge circular velocities at the half mass radii in kms^-1'''

    mdisk_gas = mgal[0]; mdisk_star = mgal[1]; mbulge = mgal[2]
    Jdisk_gas = Jgal[0]; Jdisk_star = Jgal[1]; Jbulge = Jgal[2]
    mdisk = mdisk_gas + mdisk_star
    Jdisk = Jdisk_gas + Jdisk_star

    if mbulge < 0.0 or mdisk_gas < 0.0 or mdisk_star < 0.0 or Jdisk_gas < 0.0 or Jdisk_star < 0.0 or Jbulge < 0.0:
        print "Error, negative galaxy component mass or angular momentum"
        quit()

    elif mbulge == 0.0:
        mdisks = [mdisk_gas, mdisk_star]; Jdisks = [Jdisk_gas, Jdisk_star]
        rdisks, vdisks = Calculate_Decoupled_Disk_Sizes(Jdisks, mdisks, strc, mhalo, rhalo)

        return [rdisks[0], rdisks[1], 0.0], [vdisks[0], vdisks[1], 0.0]

    # Guess disk and bulge sizes using a no baryonic self-gravity calculation (assume gas and stellar disks are coupled)
    rdisk_guess, vdisk_guess = Calculate_Disk_Size_No_Selfgravity(Jdisk, mdisk, strc, mhalo, rhalo)
    rbulge_guess, vbulge_guess = Calculate_Bulge_Size_No_Selfgravity(Jbulge, mbulge, strc, mhalo, rhalo)
    rgal_guess = [rdisk_guess, rdisk_guess, rbulge_guess]

    if mdisk_gas > 0.0:
        cdisk_gas = Calculate_c(Jdisk_gas, mdisk_gas, rhalo, mhalo, "disk") # kpc Msun
    else:
        cdisk_gas = 0.0

    if mdisk_star > 0.0:
        cdisk_star = Calculate_c(Jdisk_star, mdisk_star, rhalo, mhalo, "disk") # kpc Msun
    else:
        cdisk_star = 0.0

    cbulge = Calculate_c(Jbulge, mbulge, rhalo, mhalo, "bulge") # kpc Msun

    cgal = [cdisk_gas, cdisk_star, cbulge]
    
    print Jbulge/mbulge, cbulge
    exit()

    fhalo = (mhalo - mdisk - mbulge) / mhalo

    solution = scipy.optimize.root(Vector_Root_eqn_Decoupled_Disks_plus_Bulge, x0 = rgal_guess, args=(cgal,mhalo,rhalo,strc,fhalo,mgal), jac=None, method='hybr')

    rdisk_gas = solution.x[0] # kpc
    rdisk_star = solution.x[1] # kpc
    rbulge = solution.x[2] # kpc

    if rbulge == 0.0:
        print "Warning, size calculation returned 0 size for the bulge"
        print "rbulge = 0, mbulge = ", mbulge
        quit()
    
    if mdisk_gas == 0.0:
        rdisk_gas = 0.0; vdisk_gas = 0.0
    else:
        vdisk_gas = np.sqrt(Constant.G *cdisk_gas *Constant.kpc *Constant.Msun) / (rdisk_gas *Constant.kpc) /Constant.kms # kms^-1
    
    if mdisk_star == 0.0:
        rdisk_star = 0.0; vdisk_star = 0.0
    else:
        vdisk_star = np.sqrt(Constant.G *cdisk_star *Constant.kpc *Constant.Msun) / (rdisk_star *Constant.kpc) /Constant.kms # kms^-1
    
    vbulge = np.sqrt(Constant.G *cbulge *Constant.kpc *Constant.Msun) / (rbulge *Constant.kpc) /Constant.kms # kms^-1

    return [rdisk_gas,rdisk_star,rbulge], [vdisk_gas,vdisk_star,vbulge]

size_t1 = time.time()
print "Done, time taken = ", size_t1 - size_t0

# Example usage of Calculate_Decoupled_Disk_Sizes_plus_Bulge
if __name__=='__main__':
    
    Jdisk_star = 0.0 # 10**13.5 # Msun kpc kms^-1
    Jdisk_gas = 0.0 # 10**13.75 # Msun kpc kms^-1
    mdisk_star = 0.0 # 10**10.1 # Msun
    mdisk_gas = 0.0 # 10**10.5 # Msun
    mbulge = 10.**9.
    Jbulge = 10.**11.5

    Jdisk = Jdisk_star + Jdisk_gas
    mdisk = mdisk_star + mdisk_gas
    
    Jdisks = [Jdisk_gas, Jdisk_star]
    mdisks = [mdisk_gas, mdisk_star]

    rhalo = 200.0 # kpc
    strc = 0.1
    mhalo = 4 * 10**12

    rdisk, vdisk = Calculate_Disk_Size_Dynamic_Halo(Jdisk, mdisk, strc, mhalo, rhalo)

    print "Coupled disk size"
    print rdisk

    (rdisk_gas, rdisk_star), (vdisk_gas, vdisk_star) = Calculate_Decoupled_Disk_Sizes(Jdisks, mdisks, strc, mhalo, rhalo)

    print "Decoupled disk no bulge"
    print rdisk_gas, rdisk_star


    print "Decoupled disk with bulge"

    Jgal = [Jdisk_gas, Jdisk_star, Jbulge]; mgal = [mdisk_gas, mdisk_star, mbulge]
    (rdisk_gas, rdisk_star, rbulge), (vdisk_gas, vdisk_star, vbulge) = Calculate_Decoupled_Disk_Sizes_plus_Bulge(Jgal, mgal, strc, mhalo, rhalo)
    print rdisk_gas, rdisk_star, rbulge
