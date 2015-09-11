from input_parameters import *; from utilities_cosmology import *; from halo_properties_routines import *

class MergerTrees:
    def __init__(self):
        self.trees = []

class MergerTree:
    def __init__(self):
        self.levels = []

    def Create_Level(self,a):
        self.levels.append(TreeLevel(a))

class TreeLevel:
    def __init__(self, a):
        self.haloes = []
        self.a = a
    
    def Create_Halo(self, node_data, node_mass):
        self.haloes.append(Halo(self,node_data, node_mass))

    # Construct ordered list of all haloes/subhaloes on this level.
    def NodesOnLevel(self):
        nodes_on_level = []
        for halo in self.haloes:
            nodes_on_level.append(halo)
            for subhalo in halo.subhaloes:
                nodes_on_level.append(subhalo)
        return nodes_on_level

class Halo:
    def __init__(self, level, node_data, node_mass):
        self.level = level

        self.subhaloes = []
        self.progenitors = []
        self.isType2 = False # Haloes are not type 2 satellites

        self.mchalo = 0.0
        self.vchalo = 0.0
        self.rchalo = 0.0
        self.tform = 0.0

        self.node_mass = node_mass
        self.node_index = node_data[0]
        self.host_index = node_data[1]
        self.descendant_index = node_data[2]

        self.isMainProgenitor = node_data[3]


    def Create_Subhalo(self, node_data, node_mass):
        self.subhaloes.append(Subhalo(self, self.level, node_data, node_mass))

    def IsHalo(self):
        """Check if this node is a Halo or Subhalo"""
        return True

    def Main_Progenitor(self):
        """Find the main progenitor (as defined by John's merger tree code) of this halo"""
    
        if len(self.progenitors) == 0:
            print "Error, tried to find the main progenitor of a halo that had no progenitors!"
            quit()

        for progenitor in self.progenitors:
            if progenitor.isMainProgenitor == 1:
                return progenitor

    def Calculate_Substep_Properties(self,t):
        """For a given time, t, calculate the relavent halo properties.
        By sustep here, I simply mean the properties of the halo evaluated at time, t"""

        #### Calculate halo mass at this substep
        mchalo_substep = self.mchalo + self.hfr * (t-self.level.t) # Halo mass at this substep

        #### Halo spin is constant
        spin_substep = self.spin

        #### Work out the time since the last formation event.

        # Determine if there has been a formation event since the beginning of the step
        if len(self.t_formation_events)>0:
            # In this case, there has been at least one formation event that occured in this step but before the substep
            if np.min(self.t_formation_events) < t:
                index_last_formation_event = np.argmax(self.t_formation_events[t>=self.t_formation_events])

                t_last_formation_event = self.t_formation_events[index_last_formation_event] # Time at the last formation event before this substep
                a_last_formation_event = self.a_formation_events[index_last_formation_event] # Expansion factor at the last formation event
                m_last_formation_event = self.m_formation_events[index_last_formation_event] # Halo mass at the last formation event

                tform_substep = t - t_last_formation_event # Time that has elapsed between the last formation event and this substep
                formation_event = True

            # In this case, the formation event(s) on this step occur after this substep. 
            # Hence, tform_substep is just tform + the time that has passed from the start of the step
            else:
                tform_substep = self.tform + t - self.level.t
                formation_event = False
        else:
            # In this case, there are no formation events inside this step, so tform_substep is just tform + the time that has passed from the start of the step
            tform_substep = self.tform + t - self.level.t
            formation_event = False

        ##### Now calculate vhalo at this substep
        a_substep = a_Universe(t, Parameter.omm, Parameter.h) # Calculate the expansion factor at this substep
        
        # In this case, we need to recalculate vhalo following the spherical collapse model applied at the previous formation event
        if Parameter.propagate_vhalo:
            if formation_event:
                vchalo_substep = Calculate_Halo_Virial_Velocity(m_last_formation_event, a_last_formation_event)
            
            # If there hasn't been a formation event in this step before this substep then the halo circular velocity is the same as it was at the start of the step.
            else:
                vchalo_substep = self.vchalo

        # In this case, we need to calculate vhalo following the spherical collapse model evaluated at time, t
        else:
            vchalo_substep = Calculate_Halo_Virial_Velocity(mchalo_substep, a_substep)


        #### Now compute rhalo at this substep
        rchalo_substep = Calculate_Halo_Virial_Radius(mchalo_substep, vchalo_substep)

        #### Now compute the halo concentration at this substep

        # In this case, we need to recalculate strc only if there has been a formation in this step but before this substep
        if Parameter.propagate_strc:
            if formation_event:
                strc_substep = NFW_Scale_Radius(m_last_formation_event, a_last_formation_event)
            # Otherwise, strc is equal to it's value at the start of the step
            else:
                strc_substep = self.strc

        # In this case, we need to recompute strc evaluated at time, t
        else:
            strc_substep = NFW_Scale_Radius(mchalo_substep, a_substep)
            
        return mchalo_substep, spin_substep, tform_substep, vchalo_substep, rchalo_substep, strc_substep

    def Create_Halo_Substep(self,t):
        """Create a substep oject that contains halo properties evaluated at some time, t, within the main step"""

        mchalo_substep, spin_substep, tform_substep, vchalo_substep, rchalo_substep, strc_substep = self.Calculate_Substep_Properties(t)

        self.Substep = Halo_Substep(self.level, mchalo_substep, spin_substep, tform_substep, vchalo_substep, rchalo_substep, strc_substep)

    def Calculate_Baryon_Accretion_Rates(self):
        """Calculate the mass and angular momentum accretion rates of baryons onto this halo"""

        dt = self.level.dt

        # Calculate the cosmological accretion array for this step
        if self.isMainProgenitor == 1:

            hfr = self.hfr #  Halo mass accretion rate in Msun Gyr^-1
            Jdot_halo = self.Jdot_halo # Halo angular momentum accretion rate in Msun kpc kms^-1 Gyr^-1

            if hfr < 0.0 or Jdot_halo < 0.0:
                print "Error, negative subhalo (infall) M/J accretion"
                quit()

            # Only allow gas infall if the descendant is a host halo on the next timestep.
            if self.descendant.IsHalo():

                # Calculate cosmological gas infall             
                mb_progenitors = 0.0 # Baryonic mass in progenitors of the descendant halo
                Jb_progenitors = 0.0 # Baryonic angular momentum in progenitors of the descendant halo

                # Calculate total mass and angular momentum in progenitors of the descendant
                for progenitor in self.descendant.progenitors:
                    if hasattr(progenitor, 'y'):
                        mb_progenitors += np.sum(progenitor.y[1:5])
                        Jb_progenitors += np.sum(progenitor.y[10:14])

                # Calculate total baryonic mass and angular momentrum in progenitors of subhaloes of the descendant
                if self.descendant.IsHalo():
                    for subhalo in self.descendant.subhaloes:
                        for progenitor in subhalo.progenitors:
                            if hasattr(progenitor, 'y'):
                                mb_progenitors += np.sum(progenitor.y[1:5])
                                Jb_progenitors += np.sum(progenitor.y[10:14])

                # Avoid roundoff errors
                mb_progenitors = min(mb_progenitors,self.descendant.mchalo*Constant.fb)
                Jb_progenitors = min(Jb_progenitors,self.descendant.J_halo*Constant.fb)

                bfr = (Constant.fb*self.descendant.mchalo - mb_progenitors ) /dt /Constant.fb
                Jdot_hot = (Constant.fb*self.descendant.J_halo - Jb_progenitors ) /dt /Constant.fb

                if bfr < 0.0 or Jdot_hot < 0.0:
                    print "Error, negative gas M/J infall rate"
                    print self, self.descendant
                    print bfr, Jdot_hot
                    print self.mchalo, self.descendant.mchalo, mb_progenitors/Constant.fb
                    print self.J_halo, self.descendant.J_halo, Jb_progenitors/Constant.fb
                    exit()

            else:
                bfr = 0.0; Jdot_hot = 0.0                

            self.hfr_array = np.array([ hfr, bfr, 0, 0, 0, bfr, 0, 0, 0, Jdot_halo, Jdot_hot, 0, 0, 0, bfr, Jdot_hot])
        else:
            self.hfr_array = np.zeros(Parameter.n_linalg_variables)


class Subhalo(Halo):
    def __init__(self, host_halo, level, node_data, node_mass):
        self.level = level # This gives the subhalo access to information in the hosting level object, such as timestep, expansion factor etc
        self.host_halo = host_halo # This gives the subhalo access to information in the hosting halo object
        
        self.progenitors = []
        self.isType2 = False # By default, not a type 2 satellite

        self.mchalo = 0.0
        self.vchalo = 0.0
        self.rchalo = 0.0

        self.tform = 0.0

        self.node_mass = node_mass
        self.node_index = node_data[0]
        self.host_index = node_data[1]
        self.descendant_index = node_data[2]

        self.isMainProgenitor = node_data[3]

    def IsHalo(self):
        """ Check if this node is a Halo or Subhalo"""
        return False
    
    def IsMerging(self):
        """Check if the baryons in the subhalo are currently merging onto the central galaxy"""
        if self.t_since_infall > self.t_merge:
            return True
        else:
            return False

    def Create_Subhalo_Substep(self,t):
        """Create a substep oject that contains subhalo properties evaluated at some time, t, within the main step"""

        # Call the Halo version of this routine to get most of the information
        mchalo_substep, spin_substep, tform_substep, vchalo_substep, rchalo_substep, strc_substep = self.Calculate_Substep_Properties(t)

        # Get additional information specific to subhaloes
        t_since_infall_substep = self.t_since_infall + t - self.level.t # Time since the satellite fell in
        t_merge_substep = self.t_merge # At present, t_merge doesn't change between snapshots

        self.Substep = Subhalo_Substep(self.level, self.host_halo, mchalo_substep, spin_substep, tform_substep, vchalo_substep, rchalo_substep, strc_substep, t_since_infall_substep, t_merge_substep)

class Halo_Substep:

    def __init__(self, level, mchalo, spin, tform, vchalo, rchalo, strc):
        self.level = level # Link to the merger tree level (snapshot) instance
        self.mchalo = mchalo
        self.spin = spin
        self.tform = tform
        self.vchalo = vchalo
        self.rchalo = rchalo
        self.strc = strc

    def Rcore(self):
        """Return the core radius for the hot gas profile in kpc"""
        return self.rchalo * Parameter.fcore

    def T_dyn(self):
        """Return the halo dynamical time in Gyr"""
        if self.vchalo > 0.0:
            tdyn =  Constant.G * self.mchalo * Constant.Msun / (self.vchalo*Constant.kms)**3 /Constant.Gyr # Gyr
            return tdyn
        else:
            return 0.0

    def Mgalaxy(self):
        """Return the total mass of baryons in the galaxy in Msun"""
        # In current implementation, there is no bulge
        mgal = self.y[2] + self.y[4]
        return mgal


class Subhalo_Substep(Halo_Substep):

    def __init__(self, level, host_halo, mchalo, spin, tform, vchalo, rchalo, strc, t_since_infall, t_merge):

        self.level = level
        self.host_halo = host_halo

        self.mchalo = mchalo
        self.tform = tform
        self.vchalo = vchalo
        self.rchalo = rchalo
        self.strc = strc
        self.spin = spin

        self.t_since_infall = t_since_infall
        self.t_merge = t_merge

    def IsMerging(self):
        """Check if the baryons in the subhalo are currently merging onto the central galaxy"""
        if self.t_since_infall > self.t_merge:
            return True
        else:
            return False
