# This module just defines what each quantity is in the y arrays
# In same, cases these functions are called to return dy/dt in which case the units will be in y Gyr^-1
# If you want to add an extra reservoir, you will need to modify this module (and change n_linalg_variables accordinly)

def Mhalo(y):
    '''Halo mass in Msun'''
    return y[0]

def Mhot(y):
    '''Mass in the host gas halo in Msun'''
    return y[1]

def Mcold(y):
    '''Mass in the cold gas disk in Msun'''
    return y[2]

def Mres(y):
    '''Mass in the ejected gas reservoir (SNe feedback) in Msun'''
    return y[3]

def Mstar(y):
    '''Return stellar mass in the disk in Msun'''
    return y[4]

def MZhot(y):
    '''Return mass of metals in the hot gas halo in Msun'''
    return y[5]

def MZcold(y):
    '''Return the mass of metals in the cold gas disk in Msun'''
    return y[6]

def MZres(y):
    '''Return the mass of metals in the ejected gas reservoir in Msun'''
    return y[7]

def MZstar(y):
    '''Return the mass of metals in stars in the stellar disk in Msun'''
    return y[8]

def Jhalo_f(y):
    '''Return the angular momentum of the halo in Msun kpc kms^-1'''
    return y[9]

def Jhot_f(y):
    '''Return the angular momentum of the hot gas halo in Msun kpc kms^-1'''
    return y[10]

def Jcold_f(y):
    '''Return the angular momentum of the cold gas disk in Msun kpc kms^-1'''
    return y[11]

def Jres_f(y):
    '''Return the angular momentum of the reservoir gas in Msun kpc kms^-1'''
    return y[12]

def Jstar_f(y):
    '''Return the angular momentum of stars in the stellar disk in Msun kpc kms^-1'''
    return y[13]

def Mnotional(y):
    '''Return the mass in hot gas notional profile in Msun'''
    return y[14]

def Jnotional_f(y):
    '''Return the angular momentum in the hot gas notional profile in Msun kpc kms^-1'''
    return y[15]

def Mbaryon(y):
    '''Return the baryonic mass in the halo in Msun'''
    mbaryon = Mhot(y) + Mcold(y) + Mres(y) + Mstar(y)
    return mbaryon

def Jbaryon_f(y):
    '''Return the baryonic mass in the halo in Msun kpc kms^-1'''
    Jbaryon = Jhot_f(y) + Jcold_f(y) + Jres_f(y) + Jstar_f(y)
    return Jbaryon
