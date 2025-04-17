"""

Copyright © 2024 Robert Behrens de Luna. All rights reserved.

This software project and all associated files are protected by copyright and may not
be copied, distributed, or modified without the express written permission of the 
copyright holder.

A significant portion of this project relies on code from the WISDEM/WEIS repository (https://github.com/WISDEM/WEIS),
which is licensed under the Apache 2.0 License. The terms of the Apache 2.0 License apply to those parts and can be
found in the LICENSE file. The remainder of the code, as indicated by this copyright notice, is protected by copyright
and may not be used without authorization.

"""

import numpy as np
import pandas as pd
import os
import shutil
import sys
import copy
import glob
import logging
import pickle
import subprocess
from pathlib import Path
from scipy.interpolate                      import PchipInterpolator
from openmdao.api                           import ExplicitComponent
from wisdem.commonse import NFREQ
from wisdem.commonse.cylinder_member import get_nfull
import wisdem.commonse.utilities              as util
from wisdem.rotorse.rotor_power             import eval_unsteady
from weis.aeroelasticse.FAST_post         import FAST_IO_timeseries
from wisdem.floatingse.floating_frame import NULL, NNODES_MAX, NELEM_MAX
from weis.dlc_driver.dlc_generator    import DLCGenerator
from weis.dlc_driver.dlc_generator    import DLCInstance
from weis.aeroelasticse.CaseGen_General import CaseGen_General
from functools import partial
from pCrunch import PowerProduction
from weis.aeroelasticse import FileTools
from weis.aeroelasticse.turbsim_util import Turbsim_wrapper
from weis.aeroelasticse.turbsim_util import generate_wind_files

from weis.aeroelasticse.utils import OLAFParams
from rosco.toolbox import control_interface as ROSCO_ci
from pCrunch.io import OpenFASTOutput
from pCrunch import LoadsAnalysis, PowerProduction, FatigueParams
from weis.control.dtqp_wrapper          import dtqp_wrapper
from weis.aeroelasticse.StC_defaults        import default_StC_vt
from weis.aeroelasticse.CaseGen_General import case_naming
from wisdem.inputs import load_yaml, write_yaml
## neccessary inputs:
import wisdem.commonse.cross_sections as cs

from weis.aeroelasticse.FAST_reader import InputReader_OpenFAST

from weis.aeroelasticse.QBlade_writer         import InputWriter_QBlade
import weis.aeroelasticse.QBlade_wrapper as qbwrap
import random
import base64

from scipy.special import gamma

_encoded_version = 'MS4wLjA='
__version__ = base64.b64decode(_encoded_version).decode('utf-8')

logger = logging.getLogger("wisdem/weis") 

weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def make_coarse_grid(s_grid, diam):

    s_coarse = [s_grid[0]]
    slope = np.diff(diam) / np.diff(s_grid)

    for k in range(slope.size-1):
        if np.abs(slope[k]-slope[k+1]) > 1e-2:
            s_coarse.append(s_grid[k+1])
    s_coarse.append(s_grid[-1])
    return np.array(s_coarse)

class QBLADELoadCases(ExplicitComponent):
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')
        self.options.declare('wt_init')

    def setup(self):
        modopt = self.options['modeling_options']
        rotorse_options  = modopt['WISDEM']['RotorSE']
        mat_init_options = modopt['materials']
        self.n_blades      = modopt['assembly']['number_of_blades']
        self.n_span        = n_span    = rotorse_options['n_span']
        self.n_pc          = n_pc      = rotorse_options['n_pc']

        # Environmental Conditions needed regardless of where model comes from
        self.add_input('V_cutin',     val=0.0, units='m/s',      desc='Minimum wind speed where turbine operates (cut-in)')
        self.add_input('V_cutout',    val=0.0, units='m/s',      desc='Maximum wind speed where turbine operates (cut-out)')
        self.add_input('Vrated',      val=0.0, units='m/s',      desc='rated wind speed')
        self.add_discrete_input('turbulence_class', val='A', desc='IEC turbulence class')
        self.add_discrete_input('turbine_class',    val='I', desc='IEC turbine class')
        self.add_input('shearExp',    val=0.0,                   desc='shear exponent')

        #va gt
        if self.options['modeling_options']['Floris']['flag']:
            self.add_input('site_weibull_Vmean', val=0.0, desc='site average wind speed')
            self.add_input('site_weibull_shape_factor', val=0.0, desc='site weibull shape factor')
        #va gt

        if not self.options['modeling_options']['Level4']['from_qblade']:

            self.n_xy          = n_xy      = rotorse_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
            self.n_Re          = n_Re      = rotorse_options['n_Re'] # Number of Reynolds, so far hard set at 1
            self.n_aoa         = n_aoa     = rotorse_options['n_aoa']# Number of angle of attacks
            self.n_tab         = n_tab     = rotorse_options['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
            self.n_mat         = n_mat     = mat_init_options['n_mat']
            self.n_layers      = n_layers  = rotorse_options['n_layers']

            # Simulation set-up inputs
            self.add_input('rho',                   val=0.0, units='kg/m**3',               desc='air density')
            self.add_input('mu',                    val=0.0, units='kg/(m*s)',              desc='air dynamic viscosity')
            self.add_input('rho_water',             val=0.0, units='kg/m**3',               desc='air density_water')
            self.add_input('mu_water',              val=0.0, units='kg/(m*s)',              desc='dynamic viscosity of water')
            self.add_input('water_depth',           val=0.0, units='m',                     desc='water depth for analysis.  Values > 0 mean offshore')

            # Main file inputs
            self.add_discrete_input('rotor_orientation',val='upwind', desc='Rotor orientation, either upwind or downwind.')
            self.add_input('cone',                  val=0.0, units='deg',    desc='Cone angle of the rotor. It defines the angle between the rotor plane and the blade pitch axis. A standard machine has positive values')
            self.add_input('tilt',                  val=0.0, units='deg',    desc='Nacelle uptilt angle. A standard machine has positive values')
            self.add_input('overhang',              val=0.0, units='m',      desc='Horizontal distance from tower top to hub center')
            self.add_input('twr2shft',              val=0.0, units='m',      desc='Vertical distance from tower top plane to shaft start')
            self.add_input('yaw_mass',              val=0.0, units='kg',     desc='Mass of yaw system')
            self.add_input('above_yaw_mass',        val=0.0, units='kg',     desc='Mass of the nacelle above the yaw system')
            self.add_input('nacelle_cm',            val=np.zeros(3), units='m',      desc='Center of mass of the component in [x,y,z] for an arbitrary coordinate system')
            self.add_input('hub_system_mass',       val=0.0, units='kg',     desc='mass of hub system')
            self.add_input('hub_system_I',          val=np.zeros(6), units='kg*m**2',      desc='mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.')
            self.add_input('gearbox_ratio',         val=1.0,                 desc='Gearbox ratio')
            self.add_input('gearbox_efficiency',    val=1.0,                 desc='Gearbox efficiency')
            self.add_input('GenIner',               val=0.0, units='kg*m**2',desc='Moments of inertia for the generator about high speed shaft')
            self.add_input('generator_efficiency',  val=1.0,                 desc='Generator efficiency')
            self.add_input('drivetrain_spring_constant',         val=0.0,         units='N*m/rad',   desc='Moments of inertia for the generator about high speed shaft')
            self.add_input("drivetrain_damping_coefficient",    val=0.0,          units="N*m*s/rad", desc='Equivalent damping coefficient for the drivetrain system')

            # self.add_input('',val=0.0, units='',    desc='')
            # self.add_input('',val=0.0, units='',    desc='')
            # self.add_input('',val=0.0, units='',    desc='')

            # Required to carry out conversion to QBlade format
            self.add_input('hub_height',            val=0.0, units='m',     desc='Hub height')
            self.add_input('distance_tt_hub',       val=0.0, units='m',     desc='Vertical distance from tower top plane to hub flange')
            self.add_input('nacelle_I_TT',          val=np.zeros(6), units='kg*m**2',    desc='moments of Inertia for the nacelle [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] about the tower top')
            self.add_input('tower_I_base',          val=np.zeros(6), units='kg*m**2',    desc='tower moments of inertia at the tower base')
            self.add_input('r',                     val=np.zeros(n_span), units='m', desc='radial positions. r[0] should be the hub location \
                while r[-1] should be the blade tip. Any number \
                of locations can be specified between these in ascending order.')
            self.add_input('Rhub',                  val=0.0,                     units='m',     desc='dimensional radius of hub')
            self.add_input('Rtip',                  val=0.0,                     units='m',     desc='dimensional radius of tip')
            self.add_input('le_location',           val=np.zeros(n_span),                       desc='Leading-edge positions from a reference blade axis (usually blade pitch axis). Locations are normalized by the local chord length. Positive in -x direction for airfoil-aligned coordinate system')

            # Blade Aero Definition Inputs
            self.add_input('ref_axis_blade',        val=np.zeros((n_span,3)),   units='m',      desc='2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.')
            self.add_input('chord',                 val=np.zeros(n_span),       units='m',      desc='chord at airfoil locations')
            self.add_input('theta',                 val=np.zeros(n_span),       units='deg',    desc='twist at airfoil locations')
            self.add_input('rthick',                val=np.zeros(n_span),                       desc='relative thickness of airfoil distribution')
            self.add_input('ac',                    val=np.zeros(n_span),                       desc='aerodynamic center of airfoil distribution')
            self.add_input('pitch_axis',            val=np.zeros(n_span),                       desc='1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.')
            self.add_input('coord_xy_interp',       val=np.zeros((n_span, n_xy, 2)),            desc='2D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The leading edge is placed at x=0 and y=0.')
            self.add_input('airfoils_Re',           val=np.zeros((n_Re)),                       desc='Reynolds numbers of polars')
            self.add_input('airfoils_cl',           val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='lift coefficients, spanwise')
            self.add_input('airfoils_cd',           val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='drag coefficients, spanwise')
            self.add_input('airfoils_cm',           val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='moment coefficients, spanwise')
            self.add_input('airfoils_aoa',          val=np.zeros((n_aoa)), units='deg', desc='angle of attack grid for polars')
            
            # CHRONO blade structural definition inputs
            self.add_input('flap_freq',             val=0.0,              units='Hz',       desc='Blade flapwise first natural frequency') 
            self.add_input('edge_freq',             val=0.0,              units='Hz',       desc='Blade edgewise first natural frequency')
            self.add_input('beam:rhoA',             val=np.zeros(n_span), units='kg/m',     desc='mass per unit length')
            self.add_input('beam:EIyy',             val=np.zeros(n_span), units='N*m**2',   desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
            self.add_input('beam:EIxx',             val=np.zeros(n_span), units='N*m**2',   desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
            self.add_input('beam:EA',               val=np.zeros(n_span), units='N',        desc='axial stiffness')
            self.add_input('beam:GJ',               val=np.zeros(n_span), units='N*m**2',   desc='torsional stiffness')
            # TODO self.add_input('beam:G',                val=np.zeros(n_span), units='N/m**2',   desc='shear modulus')
            self.add_input('beam:KSX',              val=np.zeros(n_span),                   desc='shear factor for force in principal bending axis x')
            self.add_input('beam:KSY',              val=np.zeros(n_span),                   desc='shear factor for force in principal bending axis y')
            self.add_input('beam:flap_iner',        val=np.zeros(n_span), units='kg/m',     desc='radius of inertia corresponding to a rotation around the elastic axis X')
            self.add_input('beam:edge_iner',        val=np.zeros(n_span), units='kg/m',     desc='radius of inertia corresponding to a rotation around the elastic axis Y')
            self.add_input('beam:x_cg',             val=np.zeros(n_span), units='m',        desc='center of mass position X (in OF referecence COS)')
            self.add_input('beam:y_cg',             val=np.zeros(n_span), units='m',        desc='center of mass position Y (in OF referecence COS)')
            self.add_input('beam:x_ec',             val=np.zeros(n_span), units='m',        desc='center of elasticity position X (in OF referecence COS)')
            self.add_input('beam:y_ec',             val=np.zeros(n_span), units='m',        desc='center of elasticity position Y(in OF referecence COS)')
            self.add_input('beam:x_sc',             val=np.zeros(n_span), units='m',        desc='center of shear position X (in OF referecence COS)')
            self.add_input('beam:y_sc',             val=np.zeros(n_span), units='m',        desc='center of shear position Y (in OF referecence COS)')
            # Bug to fix: m->deg
            self.add_input('beam:Tw_iner',          val=np.zeros(n_span), units='m',        desc='orientation (twist) of the section principal inertia axes with respect the blade reference plane') 
            
            # Chrono tower structural definition inputs
            n_height_tow = modopt['WISDEM']['TowerSE']['n_height']
            self.add_input('twr_freq',              val=0.0,                                units='Hz',     desc='Tower natural frequency')
            self.add_input('twr:outer_diameter',    val=np.zeros(n_height_tow),             units='m',      desc='cylinder diameter at corresponding locations')
            self.add_input('twr:wall_thickness',    val=np.zeros(n_height_tow-1),           units='m',      desc='cylinder wall thickness at corresponding locations')
            self.add_input('twr:z',                 val=np.zeros(n_height_tow),             units='m',      desc='z-coordinates of tower and monopile used in TowerSE')
            self.add_input('twr:rhoA',              val=np.zeros(n_height_tow-1),           units='kg/m',   desc='sectional mass per unit length')
            self.add_input('twr:EIyy',              val=np.zeros(n_height_tow-1),           units='N*m**2', desc='sectional fore-aft bending stiffness per unit length')
            self.add_input('twr:EIxx',              val=np.zeros(n_height_tow-1),           units='N*m**2', desc='sectional side-side bending stiffness per unit length')
            self.add_input('twr:EA',                val=np.zeros(n_height_tow-1),           units='N',      desc='axial stiffness per unit length')
            self.add_input('twr:GJ',                val=np.zeros(n_height_tow-1),           units='N*m**2', desc='torsional stiffness per unit length')
            self.add_input('twr:G',                 val=np.zeros(n_height_tow-1),           units='Pa',      desc='shear modulus per unit length')
            self.add_input('twr:cd',                val=np.zeros(n_height_tow),             desc='drag coefficients along tower height at corresponding locations')        
            
            # More tower properties
            n_full_tow   = get_nfull(n_height_tow, nref=modopt['WISDEM']['TowerSE']['n_refine'])
            self.add_input('tower_z_full',          val=np.zeros(n_full_tow),               units='m',      desc='z-coordinates of tower and monopile used in TowerSE')
            self.add_input('tower_base_height',         val=0.0, units='m', desc='tower base height from the ground or mean sea level')
            self.add_input('transition_piece_mass', val=0.0, units='kg')
            self.add_input('transition_piece_I', val=np.zeros(3), units='kg*m**2')
            if modopt['flags']['monopile']:
                n_height_mon = modopt['WISDEM']['FixedBottomSE']['n_height']
                n_full_mon   = get_nfull(n_height_mon, nref=modopt['WISDEM']['FixedBottomSE']['n_refine'])
                self.add_input('monopile_z',                val=np.zeros(n_height_mon),     units='m',      desc='z-coordinates of tower and monopile used in TowerSE')
                self.add_input('monopile_z_full',           val=np.zeros(n_full_mon),       units='m',      desc='z-coordinates of tower and monopile used in TowerSE')
                self.add_input('monopile_outer_diameter',   val=np.zeros(n_height_mon),     units='m',      desc='cylinder diameter at corresponding locations')
                self.add_input('monopile_wall_thickness',   val=np.zeros(n_height_mon-1),   units='m')
                self.add_input('monopile_E',                val=np.zeros(n_height_mon-1),   units='Pa')
                self.add_input('monopile_G',                val=np.zeros(n_height_mon-1),   units='Pa')
                self.add_input('monopile_rho',              val=np.zeros(n_height_mon-1),   units='kg/m**3')
                self.add_input('gravity_foundation_mass',   val=0.0,                        units='kg')
                self.add_input('gravity_foundation_I',      val=np.zeros(3),                units='kg*m**2')
                monlen = max(0, n_height_mon-1)
                monlen_full = max(0, n_full_mon-1)

            # QBladeOcean Substructure floating platform inputs
            self.add_input('transition_node',               val=np.zeros(3),    units='m')
            self.add_input('platform_mass',                 val=0.0,            units='kg')
            self.add_input('platform_total_center_of_mass', val=np.zeros(3),    units='m')
            self.add_input('platform_I_total',              val=np.zeros(6),    units='kg*m**2')
            self.add_input('platform_displacement',         val=1.0,            units='m**3', desc='Volumetric platform displacement')
            self.add_input("platform_nodes",                NULL * np.ones((NNODES_MAX, 3)), units="m")
            self.add_input("platform_elem_n1",              NULL * np.ones(NELEM_MAX, dtype=np.int_))
            self.add_input("platform_elem_n2",              NULL * np.ones(NELEM_MAX, dtype=np.int_))
            self.add_discrete_input("platform_elem_memid", [0]*NELEM_MAX)

            if modopt['flags']["floating"]:
                n_member = modopt["floating"]["members"]["n_members"]
                for k in range(n_member):
                    n_height_mem = modopt["floating"]["members"]["n_height"][k]
                    self.add_input(f"member{k}:joint1", np.zeros(3), units="m")
                    self.add_input(f"member{k}:joint2", np.zeros(3), units="m")
                    self.add_input(f"member{k}:Cd", 0)
                    self.add_input(f"member{k}:Ca", 0)
                    self.add_input(f"member{k}:s", np.zeros(n_height_mem))
                    self.add_input(f"member{k}:s_ghost1", 0.0)
                    self.add_input(f"member{k}:s_ghost2", 0.0)
                    self.add_input(f"member{k}:outer_diameter", np.zeros(n_height_mem), units="m")
                    self.add_input(f"member{k}:wall_thickness", np.zeros(n_height_mem-1), units="m")
            
            # Blade composit layup info (used for fatigue analysis)
            self.add_input('sc_ss_mats',   val=np.zeros((n_span, n_mat)),        desc="spar cap, suction side,  boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
            self.add_input('sc_ps_mats',   val=np.zeros((n_span, n_mat)),        desc="spar cap, pressure side, boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
            self.add_input('te_ss_mats',   val=np.zeros((n_span, n_mat)),        desc="trailing edge reinforcement, suction side,  boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
            self.add_input('te_ps_mats',   val=np.zeros((n_span, n_mat)),        desc="trailing edge reinforcement, pressure side, boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
            self.add_discrete_input('definition_layer', val=np.zeros(n_layers),  desc='1D array of flags identifying how layers are specified in the yaml. 1) all around (skin, paint, ) 2) offset+rotation twist+width (spar caps) 3) offset+user defined rotation+width 4) midpoint TE+width (TE reinf) 5) midpoint LE+width (LE reinf) 6) layer position fixed to other layer (core fillers) 7) start and width 8) end and width 9) start and end nd 10) web layer')

            mooropt = modopt["mooring"]
            if self.options["modeling_options"]["flags"]["mooring"]:
                n_nodes = mooropt["n_nodes"]
                n_lines = mooropt["n_lines"]
                self.add_input('line_diameter',                 val=np.zeros(n_lines), units='m')
                self.add_input('line_mass_density',             val=np.zeros(n_lines), units='kg/m')
                self.add_input('line_stiffness',                val=np.zeros(n_lines), units='N')
                self.add_input('unstretched_length',            val=np.zeros(n_lines), units="m")
                self.add_input('nodes_location_full',           val=np.zeros((n_nodes, 3)), units="m") # TODO clarify where this is first used
                self.add_input('line_transverse_added_mass',    val=np.zeros(n_lines), units='kg/m')
                self.add_input('line_transverse_drag',          val=np.zeros(n_lines))
                self.add_discrete_input("node_names", val=[""] * n_nodes)

            # Inputs required for fatigue processing
            self.add_input('lifetime',                      val=25.0, units='yr',   desc='Turbine design lifetime')
            self.add_input('blade_sparU_wohlerexp',         val=1.0,                desc='Blade root Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_sparU_wohlerA',           val=1.0, units="Pa",    desc='Blade root parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_sparU_ultstress',         val=1.0, units="Pa",    desc='Blade root ultimate stress for material')
            self.add_input('blade_sparL_wohlerexp',         val=1.0,                desc='Blade root Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_sparL_wohlerA',           val=1.0, units="Pa",    desc='Blade root parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_sparL_ultstress',         val=1.0, units="Pa",    desc='Blade root ultimate stress for material')
            self.add_input('blade_teU_wohlerexp',           val=1.0,                desc='Blade root Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_teU_wohlerA',             val=1.0, units="Pa",    desc='Blade root parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_teU_ultstress',           val=1.0, units="Pa",    desc='Blade root ultimate stress for material')
            self.add_input('blade_teL_wohlerexp',           val=1.0,                desc='Blade root Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_teL_wohlerA',             val=1.0, units="Pa",    desc='Blade root parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_teL_ultstress',           val=1.0, units="Pa",    desc='Blade root ultimate stress for material')
            self.add_input('blade_root_sparU_load2stress',  val=np.ones(6), units="m**2",  desc='Blade root upper spar cap coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('blade_root_sparL_load2stress',  val=np.ones(6), units="m**2",  desc='Blade root lower spar cap coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('blade_maxc_teU_load2stress',    val=np.ones(6), units="m**2",  desc='Blade max chord upper trailing edge coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('blade_maxc_teL_load2stress',    val=np.ones(6), units="m**2",  desc='Blade max chord lower trailing edge coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('lss_wohlerexp',                 val=1.0,                desc='Low speed shaft Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('lss_wohlerA',                   val=1.0,                desc='Low speed shaft parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('lss_ultstress',                 val=1.0, units="Pa",    desc='Low speed shaft Ultimate stress for material')
            self.add_input('lss_axial_load2stress',         val=np.ones(6), units="m**2",  desc='Low speed shaft coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('lss_shear_load2stress',         val=np.ones(6), units="m**2",  desc='Low speed shaft coefficient between shear load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('tower_wohlerexp',               val=np.ones(n_height_tow-1),   desc='Tower Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('tower_wohlerA',                 val=np.ones(n_height_tow-1),   desc='Tower parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('tower_ultstress',               val=np.ones(n_height_tow-1),        units="Pa",    desc='Tower ultimate stress for material')
            self.add_input('tower_axial_load2stress',       val=np.ones([n_height_tow-1,6]),    units="m**2",  desc='Tower coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('tower_shear_load2stress',       val=np.ones([n_height_tow-1,6]),    units="m**2",  desc='Tower coefficient between shear load and stress S=C^T [Fx-z;Mx-z]')
           
            if modopt['flags']['monopile']:
                self.add_input('monopile_wohlerexp',            val=np.ones(monlen),                               desc='Tower Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
                self.add_input('monopile_wohlerA',              val=np.ones(monlen),                               desc='Tower parameter, A, in S/N curve S=A*N^-(1/m)')
                self.add_input('monopile_ultstress',            val=np.ones(monlen),                units="Pa",    desc='Tower ultimate stress for material')
                self.add_input('monopile_axial_load2stress',    val=np.ones([monlen,6]),            units="m**2",  desc='Tower coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
                self.add_input('monopile_shear_load2stress',    val=np.ones([monlen,6]),            units="m**2",  desc='Tower coefficient between shear load and stress S=C^T [Fx-z;Mx-z]')
        
            # environment inputs
            self.add_input('Hsig_wave',     val=0.0, units='m', desc='Significant wave height of incident waves')
            self.add_input('Tsig_wave',     val=0.0, units='s', desc='Peak-spectral period of incident waves')

            # Initial conditions
            self.add_input('U',             val=np.zeros(n_pc), units='m/s', desc='wind speeds')
            self.add_input('Omega',         val=np.zeros(n_pc), units='rpm', desc='rotation speeds to run')
            self.add_input('pitch',         val=np.zeros(n_pc), units='deg', desc='pitch angles to run')
            self.add_input("Ct_aero",       val=np.zeros(n_pc), desc="rotor aerodynamic thrust coefficient")

        if modopt['Level4']['simulation']['DLCGenerator']:
            n_ws = modopt['DLC_driver']['n_cases']
        elif modopt['Level4']['simulation']['WNDTYPE'] == 1:
            n_ws = len(modopt['Level4']['QTurbSim']['URef'])
        else:
            n_ws = len(modopt['Level4']['simulation']['MEANINF'])
        
        # QBlade options
        QBmgmt = modopt['General']['qblade_configuration']
        self.model_only = QBmgmt['model_only']
        QBLADE_directory_base = QBmgmt['QB_run_dir']
        # If the path is relative, make it an absolute path to current working directory
        if not os.path.isabs(QBLADE_directory_base):
            QBLADE_directory_base = os.path.join(os.getcwd(), QBLADE_directory_base)
        # Flag to clear QBlade run folder. Use it only if disk space is an issue
        self.clean_QBLADE_directory = False
        self.QBLADE_InputFile = QBmgmt['QB_run_mod']
        self.QBLADE_runDirectory = QBLADE_directory_base
        self.QBLADE_namingOut = self.QBLADE_InputFile
        if modopt['Level4']['simulation']['DLCGenerator'] or modopt['Level4']['simulation']['WNDTYPE']== 1:
            self.wind_directory = os.path.join(self.QBLADE_runDirectory, 'wind')
            if not os.path.exists(self.wind_directory):
                os.makedirs(self.wind_directory, exist_ok=True) 

        self.turbsim_exe = shutil.which('turbsim')

        # Outpus

        # Rotor power outputs
        self.add_output('V_out',        val=np.zeros(n_ws),   units='m/s',    desc='wind speed vector from the OF simulations')
        self.add_output('P_out',        val=np.zeros(n_ws),   units='W',      desc='rotor electrical power')
        self.add_output('Cp_out',       val=np.zeros(n_ws),                   desc='rotor aero power coefficient')
        self.add_output('Ct_out',       val=np.zeros(n_ws),                   desc='rotor aero thrust coefficient')
        self.add_output('Omega_out',    val=np.zeros(n_ws),   units='rpm',    desc='rotation speeds to run')
        self.add_output('pitch_out',    val=np.zeros(n_ws),   units='deg',    desc='pitch angles to run')
        self.add_output('AEP',          val=0.0,                    units='kW*h',   desc='annual energy production reconstructed from the openfast simulations')


        # Control outputs
        self.add_output('rotor_overspeed',      val=0.0,                                    desc='Maximum percent overspeed of the rotor during all OpenFAST simulations')  # is this over a set of sims?
        self.add_output('max_nac_accel',        val=0.0,                    units='m/s**2', desc='Maximum nacelle acceleration magnitude all OpenFAST simulations')  # is this over a set of sims?
        self.add_output('avg_pitch_travel',     val=0.0,                    units='deg/s',  desc='Average pitch travel')  # is this over a set of sims?
        self.add_output('pitch_duty_cycle',     val=0.0,                    units='deg/s',  desc='Average pitch travel')  # is this over a set of sims?
        self.add_output('max_pitch_rate_sim',   val=0.0,                    units='deg/s',  desc='Maximum pitch command rate over all simulations')  # is this over a set of sims?

        # Blade related outputs
        self.add_output('max_TipDxc',           val=0.0,                    units='m',      desc='Maximum of channel TipDxc, i.e. out of plane tip deflection. For upwind rotors, the max value is tower the tower')
        self.add_output('blade_maxTD_Mx',       val=np.zeros(n_span),       units='kN*m',   desc='distributed moment around blade-aligned x-axis corresponding to maximum blade tip deflection')
        self.add_output('blade_maxTD_My',       val=np.zeros(n_span),       units='kN*m',   desc='distributed moment around blade-aligned y-axis corresponding to maximum blade tip deflection')
        self.add_output('blade_maxTD_Fz',       val=np.zeros(n_span),       units='kN',     desc='distributed force in blade-aligned z-direction corresponding to maximum blade tip deflection')
        self.add_output('max_RootMyb',          val=0.0,                    units='kN*m',   desc='Maximum of the signals RootMyb1, RootMyb2, ... across all n blades representing the maximum blade root flapwise moment')
        self.add_output('max_RootMyc',          val=0.0,                    units='kN*m',   desc='Maximum of the signals RootMyb1, RootMyb2, ... across all n blades representing the maximum blade root out of plane moment')
        self.add_output('max_RootMzb',          val=0.0,                    units='kN*m',   desc='Maximum of the signals RootMzb1, RootMzb2, ... across all n blades representing the maximum blade root torsional moment')
        self.add_output('DEL_RootMyb',          val=0.0,                    units='kN*m',   desc='damage equivalent load of blade root flap bending moment in out-of-plane direction')
        self.add_output('max_aoa',              val=np.zeros(n_span),       units='deg',    desc='maxima of the angles of attack distributed along blade span')
        self.add_output('std_aoa',              val=np.zeros(n_span),       units='deg',    desc='standard deviation of the angles of attack distributed along blade span')
        self.add_output('mean_aoa',             val=np.zeros(n_span),       units='deg',    desc='mean of the angles of attack distributed along blade span')
        
        # Hub outputs
        self.add_output('hub_Fxyz',             val=np.zeros(3),    	    units='kN',     desc = 'Maximum hub forces in the non rotating frame')
        self.add_output('hub_Mxyz',             val=np.zeros(3),    	    units='kN*m',   desc = 'Maximum hub moments in the non rotating frame')

        # Tower related outputs
        self.add_output('max_TwrBsMyt',         val=0.0,                    units='kN*m',   desc='maximum of tower base bending moment in fore-aft direction')
        self.add_output('max_TwrBsMyt_ratio',   val=0.0,                                    desc='ratio of maximum of tower base bending moment in fore-aft direction to maximum allowable bending moment')
        self.add_output('DEL_TwrBsMyt',         val=0.0,                    units='kN*m',   desc='damage equivalent load of tower base bending moment in fore-aft direction')
        self.add_output('DEL_TwrBsMyt_ratio',   val=0.0,                                    desc='ratio of damage equivalent load of tower base bending moment in fore-aft direction to maximum allowable bending moment')
        # Tower outputs
        if not self.options['modeling_options']['Level4']['from_qblade']:
            self.add_output('tower_maxMy_Fx',       val=np.zeros(n_full_tow-1), units='kN',     desc='distributed force in tower-aligned x-direction corresponding to maximum fore-aft moment at tower base')
            self.add_output('tower_maxMy_Fy',       val=np.zeros(n_full_tow-1), units='kN',     desc='distributed force in tower-aligned y-direction corresponding to maximum fore-aft moment at tower base')
            self.add_output('tower_maxMy_Fz',       val=np.zeros(n_full_tow-1), units='kN',     desc='distributed force in tower-aligned z-direction corresponding to maximum fore-aft moment at tower base')
            self.add_output('tower_maxMy_Mx',       val=np.zeros(n_full_tow-1), units='kN*m',   desc='distributed moment around tower-aligned x-axis corresponding to maximum fore-aft moment at tower base')
            self.add_output('tower_maxMy_My',       val=np.zeros(n_full_tow-1), units='kN*m',   desc='distributed moment around tower-aligned x-axis corresponding to maximum fore-aft moment at tower base')
            self.add_output('tower_maxMy_Mz',       val=np.zeros(n_full_tow-1), units='kN*m',   desc='distributed moment around tower-aligned x-axis corresponding to maximum fore-aft moment at tower base')

             # Monopile outputs
            if modopt['flags']['monopile']:
                self.add_output('max_M1N1MKye',      val=0.0,                   units='kN*m',   desc='maximum of My moment of member 1 at node 1 (base of the monopile)')
                self.add_output('monopile_maxMy_Fx', val=np.zeros(monlen_full), units='kN',     desc='distributed force in monopile-aligned x-direction corresponding to max_M1N1MKye')
                self.add_output('monopile_maxMy_Fy', val=np.zeros(monlen_full), units='kN',     desc='distributed force in monopile-aligned y-direction corresponding to max_M1N1MKye')
                self.add_output('monopile_maxMy_Fz', val=np.zeros(monlen_full), units='kN',     desc='distributed force in monopile-aligned z-direction corresponding to max_M1N1MKye')
                self.add_output('monopile_maxMy_Mx', val=np.zeros(monlen_full), units='kN*m',   desc='distributed moment around tower-aligned x-axis corresponding to max_M1N1MKye')
                self.add_output('monopile_maxMy_My', val=np.zeros(monlen_full), units='kN*m',   desc='distributed moment around tower-aligned x-axis corresponding to max_M1N1MKye')
                self.add_output('monopile_maxMy_Mz', val=np.zeros(monlen_full), units='kN*m',   desc='distributed moment around tower-aligned x-axis corresponding to max_M1N1MKye')

        # Floating outputs
        self.add_output('Max_PtfmPitch',        val=0.0,                                    desc='Maximum platform pitch angle over a set of OpenFAST simulations')
        self.add_output('Std_PtfmPitch',        val=0.0,                    units='deg',    desc='standard deviation of platform pitch angle')
        self.add_output('Max_Offset',           val=0.0,                    units='m',      desc='Maximum distance in surge/sway direction')

        # Fatigue output
        self.add_output('damage_blade_root_sparU',  val=0.0, desc="Miner's rule cumulative damage to upper spar cap at blade root")
        self.add_output('damage_blade_root_sparL',  val=0.0, desc="Miner's rule cumulative damage to lower spar cap at blade root")
        self.add_output('damage_blade_maxc_teU',    val=0.0, desc="Miner's rule cumulative damage to upper trailing edge at blade max chord")
        self.add_output('damage_blade_maxc_teL',    val=0.0, desc="Miner's rule cumulative damage to lower trailing edge at blade max chord")
        self.add_output('damage_lss',               val=0.0, desc="Miner's rule cumulative damage to low speed shaft at hub attachment")
        self.add_output('damage_tower_base',        val=0.0, desc="Miner's rule cumulative damage at tower base")
        self.add_output('damage_monopile_base',     val=0.0, desc="Miner's rule cumulative damage at monopile base")
        
        # Simulation output
        self.add_output('qblade_failed',             val=0.0, desc="Numerical value for whether any qblade runs failed. 0 if false, 2 if true")

        self.add_discrete_output('ts_out_dir', val={})

        # iteration counter used as model name appendix
        self.qb_inumber = 0

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        print("############################################################")
        print(f"The WEIS-QBlade component with version number: {__version__} is called")
        print("############################################################")
        
        modopt = self.options['modeling_options']
        sys.stdout.flush() 
        qb_vt = self.init_QBlade_model()
        
        if not modopt['Level4']['from_qblade']:
            qb_vt = self.update_QBLADE_model(qb_vt, inputs, discrete_inputs)
        

        if self.model_only == True:
            # Write input QB files, but do not run QB
            self.write_QBLADE(qb_vt, inputs, discrete_inputs)
        else:
            # Write input QB files and run QB
            if not qb_vt['QSim']['DLCGenerator']:
                self.write_QBLADE(qb_vt, inputs, discrete_inputs)
            summary_stats, extreme_table, DELs, Damage, chan_time, dlc_generator = self.run_QBLADE(inputs, discrete_inputs, qb_vt)
            # post process results
            self.post_process(summary_stats, extreme_table, DELs, Damage, chan_time, inputs, outputs, discrete_inputs, dlc_generator, discrete_outputs)

            self.qb_inumber += 1

    def update_QBLADE_model(self, qb_vt, inputs, discrete_inputs):
        modopt = self.options['modeling_options']
        precision = int(5) # Number of decimal places to round to

        if modopt['flags']['offshore']:
            # qb_vt['QSim']['ISOFFSHORE'] = 0 # Use QBladeOcean if not set in modeling inputs TODO: should be 1 once testing is done
            qb_vt['QSim']['WATERDEPTH'] = float(inputs['water_depth'])
        qb_vt['QSim']['DENSITYAIR'] = float(inputs['rho'])
        qb_vt['QSim']['VISCOSITYAIR'] = inputs['mu'][0] / inputs['rho'][0]
        qb_vt['QSim']['DENSITYWATER'] = float(inputs['rho_water'])
        qb_vt['QSim']['VISCOSITYWATER'] = inputs['mu_water'][0] /inputs['rho_water'][0]  
        
        # if DLCGenerator is true, NUMT
        if qb_vt['QSim']['TMax'] > 0:
            qb_vt['QSim']['NUMTIMESTEPS'] = int(qb_vt['QSim']['TMax'] / qb_vt['QSim']['TIMESTEP'])

        twr_top_height = float(inputs['hub_height'][0]) - float(inputs['distance_tt_hub'][0])
        tower_base_height = max(float(inputs['tower_base_height']), float(inputs["platform_total_center_of_mass"][2]))

        qb_vt['Main']['TurbineName'] = modopt['General']['qblade_configuration']['QB_run_mod']        
        # if discrete_inputs['rotor_orientation'] == 'upwind':
        #     k = -1.
        # else:
        #     k = 1
        qb_vt['Main']['PreCone']        = round(inputs['cone'][0], precision) 
        qb_vt['Main']['ShftTilt']       = round(inputs['tilt'][0], precision)
        qb_vt['Main']['OverHang']       = round(inputs['overhang'][0], precision)
        qb_vt['Main']['Twr2Shft']       = round(float(inputs['hub_height']) - twr_top_height - abs(qb_vt['Main']['OverHang'])*np.sin(np.deg2rad(inputs['tilt'][0])), precision)
        
        qb_vt['Main']['YawBrMass']      = round(inputs['yaw_mass'][0], precision)
        qb_vt['Main']['NacMass']        = round(inputs['above_yaw_mass'][0], precision)
        qb_vt['Main']['NacCMx']         = round(inputs['nacelle_cm'][0], precision)
        qb_vt['Main']['NacCMy']         = round(inputs['nacelle_cm'][1], precision)
        qb_vt['Main']['NacCMz']         = round(inputs['nacelle_cm'][2], precision)
        qb_vt['Main']['NacYIner']       = round(inputs['nacelle_I_TT'][2] + inputs['tower_I_base'][2]/3.0, precision)
        qb_vt['Main']['HubMass']        = round(inputs['hub_system_mass'][0], precision)
        qb_vt['Main']['HubIner']        = round(inputs['hub_system_I'][0], precision)

        qb_vt['Main']['GBRATIO']        = round(inputs['gearbox_ratio'][0], precision)
        qb_vt['Main']['GBOXEFF']        = round(inputs['gearbox_efficiency'][0], precision)
        qb_vt['Main']['GENEFF']         = round(float(inputs['generator_efficiency']/inputs['gearbox_efficiency']), precision)

        qb_vt['Main']['GENINER']        = round(float(inputs['GenIner']), precision)
        qb_vt['Main']['DTTORSPR']       = round(float(inputs['drivetrain_spring_constant']), precision)
        qb_vt['Main']['DTTORDMP']       = round(float(inputs['drivetrain_damping_coefficient']), precision)

        qb_vt['Main']['NUMBLD']         = self.n_blades
        qb_vt['Main']['TWRHEIGHT']      = round(twr_top_height - tower_base_height, precision)


        ## Blade Aerodnyamic Definition Inputs
        r = (inputs['r']-inputs['Rhub']) # get blade radius
        r[0]  = inputs['Rhub']
        r[-1] = inputs['Rtip']
         # qb_vt['Aero']['BlPos']          = r # --> BlSpn in OpenFast
        qb_vt['Aero']['BlPos']          = inputs['ref_axis_blade'][:,2] + inputs['Rhub']
        qb_vt['Aero']['NumBlNds']       = self.n_span
        qb_vt['Aero']['BlTwist']        = inputs['theta']
        qb_vt['Aero']['BlChord']        = inputs['chord']
        qb_vt['Aero']['XOffset']        =  inputs['ref_axis_blade'][:,1]
        qb_vt['Aero']['YOffset']        =  inputs['ref_axis_blade'][:,0]
        qb_vt['Aero']['Paxis']          =  inputs['le_location']
        qb_vt['Aero']['af_data']        = []
        
        # Set the Aero flag AFTabMod, deciding whether we use more Re per airfoil or user-defined tables (used for example in distributed aerodynamic control)
        # Double check definition of flaps in QBlade and adapt this snipped accordingly
        if qb_vt['Aero']['AFTabMod'] == 1:
            # If AFTabMod is the default coming form the schema, check the value from WISDEM, which might be set to 2 if more Re per airfoil are defined in the geometry yaml
            qb_vt['Aero']['AFTabMod'] = modopt["WISDEM"]["RotorSE"]["AFTabMod"]
        if self.n_tab > 1 and qb_vt['Aero']['AFTabMod'] == 1:
            raise Exception('This case has to be implemented in the WEIS QBldae coupling')
        elif self.n_tab > 1 and qb_vt['Aero']['AFTabMod'] == 2:
            raise Exception('This case has to be implemented in the WEIS QBldae coupling. The case of multiple user defined tables and multiple RE was triggered')
        
        for i in range(self.n_span): # No of blade radial stations
            qb_vt['Aero']['af_data'].append([])

            if qb_vt['Aero']['AFTabMod'] == 1:
                loop_index = 1
            elif qb_vt['Aero']['AFTabMod'] == 2:
                loop_index = self.n_Re
            else:
                loop_index = self.n_tab

            for j in range(loop_index): # Number of tabs or Re
                if qb_vt['Aero']['AFTabMod'] == 1:
                    unsteady = eval_unsteady(inputs['airfoils_aoa'], inputs['airfoils_cl'][i,:,0,0], inputs['airfoils_cd'][i,:,0,0], inputs['airfoils_cm'][i,:,0,0])
                elif qb_vt['Aero']['AFTabMod'] == 2:
                    unsteady = eval_unsteady(inputs['airfoils_aoa'], inputs['airfoils_cl'][i,:,j,0], inputs['airfoils_cd'][i,:,j,0], inputs['airfoils_cm'][i,:,j,0])
                else:
                    unsteady = eval_unsteady(inputs['airfoils_aoa'], inputs['airfoils_cl'][i,:,0,j], inputs['airfoils_cd'][i,:,0,j], inputs['airfoils_cm'][i,:,0,j])

                qb_vt['Aero']['af_data'][i].append({})

                qb_vt['Aero']['af_data'][i][j]['Re']        = inputs['airfoils_Re'][j]
                qb_vt['Aero']['af_data'][i][j]['Alpha']     = np.array(unsteady['Alpha'])
                qb_vt['Aero']['af_data'][i][j]['Cl']        = np.array(unsteady['Cl'])
                qb_vt['Aero']['af_data'][i][j]['Cd']        = np.array(unsteady['Cd'])
                qb_vt['Aero']['af_data'][i][j]['Cm']        = np.array(unsteady['Cm'])
        
        qb_vt['Aero']['af_coord']       = []
        qb_vt['Aero']['rthick']         = np.zeros(self.n_span)
        qb_vt['Aero']['ac']             = np.zeros(self.n_span)
        for i in range(self.n_span):
            qb_vt['Aero']['af_coord'].append({})
            qb_vt['Aero']['af_coord'][i]['x']  = inputs['coord_xy_interp'][i,:,0]
            qb_vt['Aero']['af_coord'][i]['y']  = inputs['coord_xy_interp'][i,:,1]
            qb_vt['Aero']['rthick'][i]         = inputs['rthick'][i]
            qb_vt['Aero']['ac'][i]             = inputs['ac'][i]

        # spanwise output positions of aerodynamic forces and moments
        r_out_target  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        r = r/r[-1]
        idx_out       = [np.argmin(abs(r-ri)) for ri in r_out_target]
        self.R_out_AD = [qb_vt['Aero']['BlPos'][i] for i in idx_out] 
        if len(self.R_out_AD) != len(np.unique(self.R_out_AD)):
            raise Exception('ERROR: the spanwise resolution is too coarse and does not support 9 channels along blade span. Please increase it in the modeling_options.yaml.')
        
        ## Blade structural definition inputs    

        # get the damping as a function of critical damping in case user didn't RAYLEIGHDMP or used USECRITDAMP
        if qb_vt['Blade']['USECRITDAMP'] or qb_vt['Blade']['RAYLEIGHDMP'] == 0:
            if qb_vt['Blade']['RAYLEIGHDMP'] == 0:
                logger.warning(f"Tower RAYLEIGHDMP was zero. Updated RAYLEIGHDMP to equivalent to {qb_vt['Blade']['CRITDAMP']}% of critical damping")
            beta =  (qb_vt['Blade']['CRITDAMP']/100) / (np.pi * inputs['flap_freq'])
            qb_vt['Blade']['RAYLEIGHDMP'] = float(beta)

        if not modopt['SONATA']['flag']:
            strpit    =  inputs['beam:Tw_iner'] - inputs['theta']
        
            qb_vt['Blade']['r_curved'], qb_vt['Blade']['LENFRACT'] = self.calc_fractional_curved_length(inputs['ref_axis_blade'])
            qb_vt['Blade']['MASSD']     =  inputs['beam:rhoA']
            # rotation_angle = np.radians(90.0 - strpit) # Calculate the rotation angle for coordinate transformation from OpenFAST to QBlade Chrono
            # qb_vt['Blade']['EIx']       =  abs(inputs['beam:EIxx'] * np.cos(rotation_angle) - inputs['beam:EIyy'] * np.sin(rotation_angle))
            # qb_vt['Blade']['EIy']       =  abs(inputs['beam:EIxx'] * np.sin(rotation_angle) + inputs['beam:EIyy'] * np.cos(rotation_angle))
            qb_vt['Blade']['EIx']       =  inputs['beam:EIyy']
            qb_vt['Blade']['EIy']       =  inputs['beam:EIxx']
            qb_vt['Blade']['EA']        =  inputs['beam:EA']
            qb_vt['Blade']['GJ']        =  inputs['beam:GJ']
            qb_vt['Blade']['GA']        =  np.zeros_like(inputs['beam:EA']) # only Euler beams for now 
            qb_vt['Blade']['STRPIT']    =  strpit
            qb_vt['Blade']['KSX']       =  np.zeros_like(inputs['beam:EA']) # only Euler beams for now
            qb_vt['Blade']['KSY']       =  np.zeros_like(inputs['beam:EA']) # only Euler beams for now
            qb_vt['Blade']['RGX']       =  np.sqrt(inputs['beam:flap_iner'] / inputs['beam:rhoA']) / inputs['chord']
            qb_vt['Blade']['RGY']       =  np.sqrt(inputs['beam:edge_iner'] / inputs['beam:rhoA']) / inputs['chord']
            qb_vt['Blade']['XCM']       =  inputs['beam:y_cg'] / inputs['chord'] # careful with the reference system conversion between QBlade CHRONO and OpenFAST
            qb_vt['Blade']['YCM']       =  inputs['beam:x_cg'] / inputs['chord']
            qb_vt['Blade']['XCE']       =  inputs['beam:y_ec'] / inputs['chord']
            qb_vt['Blade']['YCE']       =  inputs['beam:x_ec'] / inputs['chord']
            qb_vt['Blade']['XCS']       =  inputs['beam:y_sc'] / inputs['chord']
            qb_vt['Blade']['YCS']       =  inputs['beam:x_sc'] / inputs['chord']
        else:
            # path to beamdyn file in temporary folder, created by running sonata
            beamdyn_blade_file = os.path.join(weis_dir,'sonata_temp', self.QBLADE_namingOut + '_BeamDyn_Blade.dat')
            
            # read sonata output with fast beam dyn reader
            fast = InputReader_OpenFAST()
            fast.read_BeamDynBlade(beamdyn_blade_file)
            
            if os.path.exists(os.path.join(weis_dir, 'sonata_temp')):
                shutil.rmtree(os.path.join(weis_dir, 'sonata_temp'))
                print(f"Directory {os.path.join(weis_dir, 'sonata_temp')} has been deleted.")
            else:
                print(f"Directory {os.path.join(weis_dir, 'sonata_temp')} does not exist.")
            
            blade_6x6 = fast.fst_vt['BeamDynBlade']

            # Map beamdyn matrices to qblade format

            qb_vt['Blade_6x6']['LENFRACT'] = np.array(blade_6x6['radial_stations'])
            qb_vt['Blade_6x6']['XCB'] = np.zeros_like(qb_vt['Blade_6x6']['LENFRACT'])
            qb_vt['Blade_6x6']['YCB'] = np.zeros_like(qb_vt['Blade_6x6']['LENFRACT'])
            qb_vt['Blade_6x6']['PITCH'] = np.zeros_like(qb_vt['Blade_6x6']['LENFRACT'])
            # Initialize lists for each K and M component
            qb_vt['Blade_6x6']['K11'] = []
            qb_vt['Blade_6x6']['K12'] = []
            qb_vt['Blade_6x6']['K13'] = []
            qb_vt['Blade_6x6']['K14'] = []
            qb_vt['Blade_6x6']['K15'] = []
            qb_vt['Blade_6x6']['K16'] = []
            qb_vt['Blade_6x6']['K22'] = []
            qb_vt['Blade_6x6']['K23'] = []
            qb_vt['Blade_6x6']['K24'] = []
            qb_vt['Blade_6x6']['K25'] = []
            qb_vt['Blade_6x6']['K26'] = []
            qb_vt['Blade_6x6']['K33'] = []
            qb_vt['Blade_6x6']['K34'] = []
            qb_vt['Blade_6x6']['K35'] = []
            qb_vt['Blade_6x6']['K36'] = []
            qb_vt['Blade_6x6']['K44'] = []
            qb_vt['Blade_6x6']['K45'] = []
            qb_vt['Blade_6x6']['K46'] = []
            qb_vt['Blade_6x6']['K55'] = []
            qb_vt['Blade_6x6']['K56'] = []
            qb_vt['Blade_6x6']['K66'] = []

            qb_vt['Blade_6x6']['M11'] = []
            qb_vt['Blade_6x6']['M12'] = []
            qb_vt['Blade_6x6']['M13'] = []
            qb_vt['Blade_6x6']['M14'] = []
            qb_vt['Blade_6x6']['M15'] = []
            qb_vt['Blade_6x6']['M16'] = []
            qb_vt['Blade_6x6']['M22'] = []
            qb_vt['Blade_6x6']['M23'] = []
            qb_vt['Blade_6x6']['M24'] = []
            qb_vt['Blade_6x6']['M25'] = []
            qb_vt['Blade_6x6']['M26'] = []
            qb_vt['Blade_6x6']['M33'] = []
            qb_vt['Blade_6x6']['M34'] = []
            qb_vt['Blade_6x6']['M35'] = []
            qb_vt['Blade_6x6']['M36'] = []
            qb_vt['Blade_6x6']['M44'] = []
            qb_vt['Blade_6x6']['M45'] = []
            qb_vt['Blade_6x6']['M46'] = []
            qb_vt['Blade_6x6']['M55'] = []
            qb_vt['Blade_6x6']['M56'] = []
            qb_vt['Blade_6x6']['M66'] = []

            # Loop over each radial station and extract K and M values
            for i in range(len(blade_6x6['radial_stations'])):
                stiff_matrix = blade_6x6['beam_stiff'][i]
                inertia_matrix = blade_6x6['beam_inertia'][i]

                # Extract stiffness values (K values)
                qb_vt['Blade_6x6']['K11'].append(stiff_matrix[0, 0])
                qb_vt['Blade_6x6']['K12'].append(stiff_matrix[0, 1])
                qb_vt['Blade_6x6']['K13'].append(stiff_matrix[0, 2])
                qb_vt['Blade_6x6']['K14'].append(stiff_matrix[0, 3])
                qb_vt['Blade_6x6']['K15'].append(stiff_matrix[0, 4])
                qb_vt['Blade_6x6']['K16'].append(stiff_matrix[0, 5])
                qb_vt['Blade_6x6']['K22'].append(stiff_matrix[1, 1])
                qb_vt['Blade_6x6']['K23'].append(stiff_matrix[1, 2])
                qb_vt['Blade_6x6']['K24'].append(stiff_matrix[1, 3])
                qb_vt['Blade_6x6']['K25'].append(stiff_matrix[1, 4])
                qb_vt['Blade_6x6']['K26'].append(stiff_matrix[1, 5])
                qb_vt['Blade_6x6']['K33'].append(stiff_matrix[2, 2])
                qb_vt['Blade_6x6']['K34'].append(stiff_matrix[2, 3])
                qb_vt['Blade_6x6']['K35'].append(stiff_matrix[2, 4])
                qb_vt['Blade_6x6']['K36'].append(stiff_matrix[2, 5])
                qb_vt['Blade_6x6']['K44'].append(stiff_matrix[3, 3])
                qb_vt['Blade_6x6']['K45'].append(stiff_matrix[3, 4])
                qb_vt['Blade_6x6']['K46'].append(stiff_matrix[3, 5])
                qb_vt['Blade_6x6']['K55'].append(stiff_matrix[4, 4])
                qb_vt['Blade_6x6']['K56'].append(stiff_matrix[4, 5])
                qb_vt['Blade_6x6']['K66'].append(stiff_matrix[5, 5])

                # Extract inertia values (M values)
                qb_vt['Blade_6x6']['M11'].append(inertia_matrix[0, 0])
                qb_vt['Blade_6x6']['M12'].append(inertia_matrix[0, 1])
                qb_vt['Blade_6x6']['M13'].append(inertia_matrix[0, 2])
                qb_vt['Blade_6x6']['M14'].append(inertia_matrix[0, 3])
                qb_vt['Blade_6x6']['M15'].append(inertia_matrix[0, 4])
                qb_vt['Blade_6x6']['M16'].append(inertia_matrix[0, 5])
                qb_vt['Blade_6x6']['M22'].append(inertia_matrix[1, 1])
                qb_vt['Blade_6x6']['M23'].append(inertia_matrix[1, 2])
                qb_vt['Blade_6x6']['M24'].append(inertia_matrix[1, 3])
                qb_vt['Blade_6x6']['M25'].append(inertia_matrix[1, 4])
                qb_vt['Blade_6x6']['M26'].append(inertia_matrix[1, 5])
                qb_vt['Blade_6x6']['M33'].append(inertia_matrix[2, 2])
                qb_vt['Blade_6x6']['M34'].append(inertia_matrix[2, 3])
                qb_vt['Blade_6x6']['M35'].append(inertia_matrix[2, 4])
                qb_vt['Blade_6x6']['M36'].append(inertia_matrix[2, 5])
                qb_vt['Blade_6x6']['M44'].append(inertia_matrix[3, 3])
                qb_vt['Blade_6x6']['M45'].append(inertia_matrix[3, 4])
                qb_vt['Blade_6x6']['M46'].append(inertia_matrix[3, 5])
                qb_vt['Blade_6x6']['M55'].append(inertia_matrix[4, 4])
                qb_vt['Blade_6x6']['M56'].append(inertia_matrix[4, 5])
                qb_vt['Blade_6x6']['M66'].append(inertia_matrix[5, 5])

        ## Tower structural definition inputs
        # TODO OpenFAST seperates the tower dfinition in sectional and nodal properties. Nodal being the description used for Aerodyn while the sectional 
        # properties are used for ElastoDyn. QBlade defines the tower within 1 file only, hence, the nodal properties are interpolated onto the sectional positions
        twr_elev_nodes  = inputs['twr:z']
        twr_d_nodes = inputs['twr:outer_diameter']
        twr_elev_sections, _ = util.nodal2sectional(twr_elev_nodes) # equivalent to OpenFAST's z_sec
        twr_d_sections, _ = util.nodal2sectional(twr_d_nodes)
        twr_wallT_sections = inputs['twr:wall_thickness']
        sec_loc = (twr_elev_sections - twr_elev_sections[0]) / (twr_elev_sections[-1] - twr_elev_sections[0])

        # The diameter and drag coefficients can't be averged between nodes since it is used for aero drag calculations
        # the diameters is hence interpolated linearla between the first and last node onto the sectional positions
        twr_aero_d_sections = np.interp(np.linspace(twr_elev_nodes[0],twr_elev_nodes[-1],len(sec_loc)), twr_elev_nodes, twr_d_nodes)
        twr_aero_cd = np.interp(np.linspace(twr_elev_nodes[0],twr_elev_nodes[-1],len(sec_loc)), twr_elev_nodes, inputs['twr:cd'])
        itube = cs.Tube(twr_d_sections, twr_wallT_sections)
        Az = itube.Area

        if qb_vt['Tower']['USECRITDAMP'] or qb_vt['Tower']['RAYLEIGHDMP'] == 0:
            if qb_vt['Tower']['RAYLEIGHDMP'] == 0:
                logger.warning(f"Tower RAYLEIGHDMP was zero. Updated RAYLEIGHDMP to equivalent to {qb_vt['Blade']['CRITDAMP']}% of critical damping")
            beta =  (qb_vt['Tower']['CRITDAMP']/100) / (np.pi * inputs['twr_freq'])
            qb_vt['Tower']['RAYLEIGHDMP'] = float(beta)

        qb_vt['Tower']['LENFRACT']  = sec_loc
        qb_vt['Tower']['MASSD']     = inputs['twr:rhoA']
        qb_vt['Tower']['EIx']       = inputs['twr:EIxx']
        qb_vt['Tower']['EIy']       = inputs['twr:EIyy']
        qb_vt['Tower']['EA']        = inputs['twr:EA']
        qb_vt['Tower']['GJ']        = inputs['twr:GJ']
        qb_vt['Tower']['GA']        = inputs['twr:G']*Az
        qb_vt['Tower']['STRPIT']    = np.zeros_like(sec_loc)
        qb_vt['Tower']['KSX']       = np.ones_like(sec_loc)*0.5
        qb_vt['Tower']['KSY']       = np.ones_like(sec_loc)*0.5
        qb_vt['Tower']['RGX']       = np.zeros_like(sec_loc)
        qb_vt['Tower']['RGY']       = np.zeros_like(sec_loc)
        qb_vt['Tower']['XCM']       = np.zeros_like(sec_loc)
        qb_vt['Tower']['YCM']       = np.zeros_like(sec_loc)
        qb_vt['Tower']['XCE']       = np.zeros_like(sec_loc)
        qb_vt['Tower']['YCE']       = np.zeros_like(sec_loc)
        qb_vt['Tower']['XCS']       = np.zeros_like(sec_loc)
        qb_vt['Tower']['YCS']       = np.zeros_like(sec_loc)
        qb_vt['Tower']['DIA']       = twr_aero_d_sections
        qb_vt['Tower']['CD']        = twr_aero_cd
    
        ## Sub-Structure QBladeOcean structural definition inputs
        if modopt['flags']['offshore']: # only if an offshore turbine is modeled
            
            qb_vt['QBladeOcean']['WATERDEPTH'] = float(inputs['water_depth'])
            qb_vt['QBladeOcean']['WATERDENSITY'] = float(inputs['rho_water'])
            
            if not qb_vt['QBladeOcean']['override_wave']:
                qb_vt['QBladeOcean']['SIGHEIGHT'] = float(inputs['Hsig_wave'])
                qb_vt['QBladeOcean']['PEAKPERIOD'] = float(inputs['Tsig_wave'])
            else:
                if len(qb_vt['QSim']['MEANINF']) != len(qb_vt['QBladeOcean']['SIGHEIGHT']) != len(qb_vt['QBladeOcean']['PEAKPERIOD']):
                    qb_vt['QBladeOcean']['SIGHEIGHT'] = np.ones_like(qb_vt['QSim']['MEANINF']) * 5
                    qb_vt['QBladeOcean']['PEAKPERIOD'] = np.ones_like(qb_vt['QSim']['MEANINF']) * 1
                    logger.warning("WARNING: inconsistent number of cases found len(SIGHEIGHT) != len(PEAKPERIOD) != len(MEANINF)! 'SIGHEIGHT' and 'PEAKPERIOD' are set to default values (5m and 10s)")
            
            if modopt['flags']['monopile']:
                qb_vt['QBladeOcean']['ISFLOATING']  =  False
                qb_vt['QBladeOcean']['SUB_MASS']    =  float(inputs["platform_mass"])
                qb_vt['QBladeOcean']['SUB_INER']    = inputs["platform_I_total"][0:3]
                qb_vt['QBladeOcean']['REF_COG_POS'] = inputs['platform_total_center_of_mass']
                qb_vt['QBladeOcean']['SUB_HYDROADDEDMASS'] = np.vstack( tuple([qb_vt['QBladeOcean']['SUB_HYDROADDEDMASS'+str(m+1)] for m in range(6)]) )
                qb_vt['QBladeOcean']['SUB_HYDROSTIFFNESS'] = np.vstack( tuple([qb_vt['QBladeOcean']['SUB_HYDROSTIFFNESS'+str(m+1)] for m in range(6)]) )
                qb_vt['QBladeOcean']['SUB_HYDRODAMPING'] = np.vstack( tuple([qb_vt['QBladeOcean']['SUB_HYDRODAMPING'+str(m+1)] for m in range(6)]) )
                qb_vt['QBladeOcean']['SUB_HYDROQUADDAMPING'] = np.vstack( tuple([qb_vt['QBladeOcean']['SUB_HYDROQUADDAMPING'+str(m+1)] for m in range(6)]))
                qb_vt['QBladeOcean']['NMooLines'] = 0
                qb_vt['QBladeOcean']['NMooMembers'] = 0

                subconstraint_tp = np.array([])
                subconstraint_gd = np.array([])
                
                mono_d = inputs['monopile_outer_diameter']
                mono_t = inputs['monopile_wall_thickness']                
                mono_elev = inputs['monopile_z']
                n_joints = len(mono_d[1:]) # Omit submerged pile
                n_members = n_joints - 1
                joints_xyz = np.c_[np.zeros((n_joints,2)), mono_elev[1:]]
                joints_xyz[:,2] += float(inputs['water_depth'])
                N1 = np.arange( n_members, dtype=np.int_ ) + 1
                N2 = np.arange( n_members, dtype=np.int_ ) + 2
                ijoints = np.arange(n_joints, dtype=np.int_) + 1
                imembers = np.arange(n_members, dtype=np.int_) + 1
                members_name = []
                for k in range(n_members):
                    members_name.append(f"member_{k}")

                # find and set constraints to the TP and and the ground
                tp_index = np.argmax(joints_xyz[:, 2])
                g_index = np.argmin(joints_xyz[:, 2])
                subconstraint_tp = np.append(subconstraint_tp, tp_index + 1).astype(int)
                subconstraint_gd  = np.append(subconstraint_gd, g_index +1).astype(int)
                
                # flexible members are required for monopile structures
                d_coarse = util.nodal2sectional(mono_d[1:])[0] # Don't need deriv
                t_coarse = mono_t[1:]
                A_coarse = (d_coarse**2 - (d_coarse-2*t_coarse)**2) * np.pi/4.  # area
                J_coarse = (d_coarse**4 - (d_coarse-2*t_coarse)**4) * np.pi/32. # polar moment of inertia
                I_coarse = (d_coarse**4 - (d_coarse-2*t_coarse)**4)  * np.pi/64 # area moment of inertia
                qb_vt['QBladeOcean']['SubElemElemID']   = imembers
                qb_vt['QBladeOcean']['NElements']       = imembers
                qb_vt['QBladeOcean']['SubElemMASSD']    = inputs['monopile_rho'][1] * A_coarse
                qb_vt['QBladeOcean']['SubElemEIx']      = inputs['monopile_E'][1] * I_coarse
                qb_vt['QBladeOcean']['SubElemEIy']      = inputs['monopile_E'][1] * I_coarse
                qb_vt['QBladeOcean']['SubElemEA']       = inputs['monopile_E'][1] * A_coarse
                qb_vt['QBladeOcean']['SubElemGJ']       = inputs['monopile_G'][1] * J_coarse
                qb_vt['QBladeOcean']['SubElemGA']       = inputs['monopile_G'][1] * A_coarse
                qb_vt['QBladeOcean']['SubElemSTRPIT']   = np.zeros_like(imembers)
                qb_vt['QBladeOcean']['SubElemKSX']      = np.ones_like(imembers)*0.5
                qb_vt['QBladeOcean']['SubElemKSY']      = np.ones_like(imembers)*0.5
                qb_vt['QBladeOcean']['SubElemRGX']      = np.sqrt(I_coarse/A_coarse)
                qb_vt['QBladeOcean']['SubElemRGY']      = np.sqrt(I_coarse/A_coarse)
                qb_vt['QBladeOcean']['SubElemXCM']      = np.zeros_like(imembers)
                qb_vt['QBladeOcean']['SubElemYCM']      = np.zeros_like(imembers)
                qb_vt['QBladeOcean']['SubElemXCE']      = np.zeros_like(imembers)
                qb_vt['QBladeOcean']['SubElemYCE']      = np.zeros_like(imembers)
                qb_vt['QBladeOcean']['SubElemXCS']      = np.zeros_like(imembers)
                qb_vt['QBladeOcean']['SubElemYCS']      = np.zeros_like(imembers)
                qb_vt['QBladeOcean']['SubElemDIA']      = d_coarse
                qb_vt['QBladeOcean']['SubElemDAMP']     = np.ones_like(imembers) * 0.01 # TODO
                
                subconstraint = np.array([subconstraint_tp[0], subconstraint_gd[0]])
                n_constraints = subconstraint.shape[0]
                iconstraints = np.arange(subconstraint.shape[0], dtype=np.int_) + 1
                qb_vt['QBladeOcean']['NSubConstr'] = n_constraints
                qb_vt['QBladeOcean']['SubConstr_ID'] = iconstraints
                qb_vt['QBladeOcean']['SubConstr_JntID'] = subconstraint.astype(int)
                qb_vt['QBladeOcean']['SubConstr_Jnt2ID'] = np.zeros_like(subconstraint)
                qb_vt['QBladeOcean']['SubConstr_TP']     = np.ones_like(subconstraint) * [1,0]
                qb_vt['QBladeOcean']['SubConstr_Ground'] = np.ones_like(subconstraint) * [0,1]
                qb_vt['QBladeOcean']['SubConstr_Spring'] = np.zeros_like(subconstraint)
                qb_vt['QBladeOcean']['SubConstr_DoF_tX'] = np.ones_like(subconstraint)
                qb_vt['QBladeOcean']['SubConstr_DoF_tY'] = np.ones_like(subconstraint)
                qb_vt['QBladeOcean']['SubConstr_DoF_tZ'] = np.ones_like(subconstraint)
                qb_vt['QBladeOcean']['SubConstr_DoF_rX'] = np.ones_like(subconstraint)
                qb_vt['QBladeOcean']['SubConstr_DoF_rY'] = np.ones_like(subconstraint)
                qb_vt['QBladeOcean']['SubConstr_DoF_rZ'] = np.ones_like(subconstraint)
                qb_vt['QBladeOcean']['TP_INTERFACE_POS'] = joints_xyz[tp_index, :] # this is not an input in windIO for monopiles. we'll use the maximum z elevation instead

                # Find the members where the 9 channels of SubDyn should be placed for pos-processing purposes
                grid_joints_monopile = (joints_xyz[:,2] - joints_xyz[0,2]) / (joints_xyz[-1,2] - joints_xyz[0,2])
                n_channels = 9
                grid_target = np.linspace(0., 0.999999999, n_channels)
                idx_out = [np.where(grid_i >= grid_joints_monopile)[0][-1] for grid_i in grid_target]
                idx_out = np.unique(idx_out)
                qb_vt['QBladeOcean']['NSub_Sensors'] = len(idx_out)
                qb_vt['QBladeOcean']['SUB_Sensors'] = [idx+1 for idx in idx_out] # index of the member
                qb_vt['QBladeOcean']['SUB_Sensors_RelPos'] =  [0] * (len(idx_out) - 1) + [1.0]  # relative position along the member, should be 0 except for the last one
                self.Z_out_QBO_mpl = [grid_joints_monopile[i] for i in idx_out]
                
            elif modopt['flags']['floating']: 
                qb_vt['QBladeOcean']['ISFLOATING']  =  True
                qb_vt['QBladeOcean']['SUB_MASS']    =  float(inputs["platform_mass"])
                qb_vt['QBladeOcean']['SUB_INER']    = inputs["platform_I_total"][0:3]
                qb_vt['QBladeOcean']['SUB_HYDROADDEDMASS'] = np.vstack( tuple([qb_vt['QBladeOcean']['SUB_HYDROADDEDMASS'+str(m+1)] for m in range(6)]) )
                qb_vt['QBladeOcean']['SUB_HYDROSTIFFNESS'] = np.vstack( tuple([qb_vt['QBladeOcean']['SUB_HYDROSTIFFNESS'+str(m+1)] for m in range(6)]) )
                qb_vt['QBladeOcean']['SUB_HYDRODAMPING'] = np.vstack( tuple([qb_vt['QBladeOcean']['SUB_HYDRODAMPING'+str(m+1)] for m in range(6)]) )
                qb_vt['QBladeOcean']['SUB_HYDROQUADDAMPING'] = np.vstack( tuple([qb_vt['QBladeOcean']['SUB_HYDROQUADDAMPING'+str(m+1)] for m in range(6)]))
                
                # Floater geometry definition
                joints_xyz = np.empty((0, 3))
                N1 = np.array([], dtype=np.int_)
                N2 = np.array([], dtype=np.int_)
                d_coarse = np.array([])
                t_coarse = np.array([])
                subconstraint_TP = np.array([])
                members_name = []

                # Look over members and grab all nodes and internal connections
                n_member = modopt["floating"]["members"]["n_members"]
                for k in range(n_member):
                    s_grid = inputs[f"member{k}:s"]         
                    idiam = inputs[f"member{k}:outer_diameter"]
                    s_coarse = make_coarse_grid(s_grid, idiam)
                    s_coarse = np.unique( np.minimum( np.maximum(s_coarse, inputs[f"member{k}:s_ghost1"]), inputs[f"member{k}:s_ghost2"]) )
                    id_coarse = np.interp(s_coarse, s_grid, idiam)
                    it_coarse = util.sectional_interp(s_coarse, s_grid, inputs[f"member{k}:wall_thickness"])
                    xyz0 = inputs[f"member{k}:joint1"]
                    xyz1 = inputs[f"member{k}:joint2"]
                    dxyz = xyz1 - xyz0
                    inode_xyz = np.outer(s_coarse, dxyz) + xyz0[np.newaxis, :]

                    # interpolate tapered members if necessary
                    if not all(i == id_coarse[0] for i in id_coarse) :
                        id_members = np.array([])
                        it_members = np.array([])
                        inode_xyz_interp = inode_xyz
                        inserted_nodes = 0
                        imembers_name = []
                        for idx in range(1,len(id_coarse)):
                            if id_coarse[idx] != id_coarse[idx - 1]:
                                node_distance = np.linalg.norm(inode_xyz[idx-1, :] - inode_xyz[idx, :]) # distance between joints
                                num_samples = self.calculate_num_samples(node_distance)
                                nodes_interp =  np.zeros((num_samples, 3)) 
                                id_interp = np.array([])
                                it_interp = np.array([])
                                for dim in range(3):
                                    nodes_interp[:, dim] = np.linspace(inode_xyz[idx-1, dim], inode_xyz[idx, dim], num_samples)
                                    if nodes_interp[0, dim] != nodes_interp[-1, dim]:
                                        # midpoints = (nodes_interp[:-1, dim] + nodes_interp[1:, dim]) / 2
                                        id_interp = np.interp(nodes_interp[:, dim], [inode_xyz[idx-1, dim], inode_xyz[idx, dim]], [id_coarse[idx-1], id_coarse[idx]])
                                        it_interp = np.interp(nodes_interp[:, dim], [inode_xyz[idx-1, dim], inode_xyz[idx, dim]], [it_coarse[idx-1], it_coarse[idx]])
                                id_interp_members = (id_interp[:-1] + id_interp[1:]) / 2
                                it_interp_member = (it_interp[:-1] + it_interp[1:]) / 2

                                inode_xyz_interp = np.insert(inode_xyz_interp, idx+inserted_nodes, nodes_interp[1:-1], axis=0)
                                id_members = np.append(id_members, id_interp_members)
                                it_members = np.append(it_members, it_interp_member)
                                inserted_nodes += nodes_interp[1:-1].shape[0]
                            else:
                                id_member = (id_coarse[idx] + id_coarse[idx - 1]) / 2
                                it_member = (it_coarse[idx] + it_coarse[idx - 1]) / 2
                                id_members = np.append(id_members, id_member)
                                it_members = np.append(it_members, it_member)
                            
                        inode_xyz = inode_xyz_interp # update the interpolated nodes
                        id_coarse = id_members
                        it_coarse = it_members

                        for member in range(len(id_members)):
                            members_name.append(f"{modopt['floating']['members']['name'][k]}_{member}") # add an extension to the name of the interpolated members:
                            # members_cd = np.append(members_cd, modopt['floating']['members']['Cd']) # dubplicate the drag coefficient for the interpolated members
                    else:
                        id_members = np.array([])
                        it_members = np.array([])
                        # use a member diameter instead of joint diameter also if the member is not tapered
                        id_member = (id_coarse[:-1] + id_coarse[1:]) / 2
                        it_member = (it_coarse[:-1] + it_coarse[1:]) / 2
                        id_members = np.append(id_members, id_member)
                        it_members = np.append(it_members, it_member)
                        id_coarse = id_members
                        it_coarse = it_members
                        members_name.append(modopt["floating"]["members"]["name"][k])
                        # members_cd = np.append(members_cd, modopt['floating']['members']['Cd'])
                    
                    inode_range = np.arange(inode_xyz.shape[0] - 1)

                    nk = joints_xyz.shape[0]
                    N1 = np.append(N1, nk + inode_range + 1)
                    N2 = np.append(N2, nk + inode_range + 2)
                    d_coarse = np.append(d_coarse, id_coarse) 
                    t_coarse = np.append(t_coarse, it_coarse)  
                    joints_xyz = np.append(joints_xyz, inode_xyz, axis=0)

                    # check if member is connected to transition piece
                    if inputs['transition_node'] in joints_xyz:
                        subconstraint_TP = np.append(subconstraint_TP, nk + 1).astype(int)

                if qb_vt['QBladeOcean']['POTFLOW']:
                    # TODO either load pot flow files or let PyHAMS calculate the pot flow coefficients here
                    qb_vt['QBladeOcean']['USE_RADIATION'] = True
                    qb_vt['QBladeOcean']['USE_EXCITATION'] = True
                    qb_vt['QBladeOcean']['STATICBUOYANCY'] = True                

                qb_vt['QBladeOcean']['REF_COG_POS']         = inputs['platform_total_center_of_mass']
                qb_vt['QBladeOcean']['TP_INTERFACE_POS']    = inputs['transition_node'] # this is a seperate input in the substructure file required in QBladeOcean

                if not qb_vt['QBladeOcean']['IsBuoy'] and len(qb_vt['QBladeOcean']['POT_HST_FILE']) > 0:
                    qb_vt['QBladeOcean']['SUB_HYDROCONSTFORCE'][2] = float(inputs['platform_displacement']) * float(inputs['rho_water']) * 9.81
                elif qb_vt['QBladeOcean']['IsBuoy'] and len(qb_vt['QBladeOcean']['POT_HST_FILE']) > 0:
                    qb_vt['QBladeOcean']['POT_HST_FILE'] = ''
                    logger.warning('WARNING: isBuoy is set to 1 AND an .hst file is provided. Buoyancy will be estimated based on explicit buoyancy!')
                elif not qb_vt['QBladeOcean']['IsBuoy'] and qb_vt['QBladeOcean']['ADVANCEDBUOYANCY'] and len(qb_vt['QBladeOcean']['POT_HST_FILE']) > 0:
                    qb_vt['QBladeOcean']['ADVANCEDBUOYANCY'] = False
                    logger.warning('WARNING: An .hst file plus a constant force is used for buoyancy calculation. ADVANCEDBUOYANCY is hence set to False')                    

                # Joint table information
                idx = np.where(joints_xyz[:,2] == -qb_vt['QBladeOcean']['WATERDEPTH'])[0]
                if len(idx) > 0:
                    joints_xyz[idx,2] += 1e-2

                # Store data
                n_joints = joints_xyz.shape[0]
                n_members = N1.shape[0]
                ijoints = np.arange(n_joints, dtype=np.int_) + 1
                imembers = np.arange(n_members, dtype=np.int_) + 1

                ########## This is required to interpret the HEXAFLOAT ##########
                # If the user wants to remove the constrain to the TP for a member (e.g. hexafloat counter weight)  
                # for idx in range(len(qb_vt['QBladeOcean']['remove_TP_constraint'])): 
                #     del_member = members_name.index(qb_vt['QBladeOcean']['remove_TP_constraint'][idx])
                #     subconstraint_TP = np.delete(subconstraint_TP , del_member)
                
                # Substructure constraints
                n_constraints = subconstraint_TP.shape[0]
                iconstraints = np.arange(subconstraint_TP.shape[0], dtype=np.int_) + 1
                qb_vt['QBladeOcean']['NSubConstr'] = n_constraints
                qb_vt['QBladeOcean']['SubConstr_ID'] = iconstraints
                qb_vt['QBladeOcean']['SubConstr_JntID'] = subconstraint_TP.astype(int)
                qb_vt['QBladeOcean']['SubConstr_Jnt2ID'] = np.zeros_like(subconstraint_TP)
                qb_vt['QBladeOcean']['SubConstr_TP'] = np.ones_like(subconstraint_TP)
                qb_vt['QBladeOcean']['SubConstr_Ground'] = np.zeros_like(subconstraint_TP)
                qb_vt['QBladeOcean']['SubConstr_Spring'] = np.zeros_like(subconstraint_TP)
                qb_vt['QBladeOcean']['SubConstr_DoF_tX'] = np.ones_like(subconstraint_TP)
                qb_vt['QBladeOcean']['SubConstr_DoF_tY'] = np.ones_like(subconstraint_TP)
                qb_vt['QBladeOcean']['SubConstr_DoF_tZ'] = np.ones_like(subconstraint_TP)
                qb_vt['QBladeOcean']['SubConstr_DoF_rX'] = np.ones_like(subconstraint_TP)
                qb_vt['QBladeOcean']['SubConstr_DoF_rY'] = np.ones_like(subconstraint_TP)
                qb_vt['QBladeOcean']['SubConstr_DoF_rZ'] = np.ones_like(subconstraint_TP)


            ## The following needs to be set for all offshore substructure files:
            # Hydro Coefficients floater    
            floater_hydro_cdN = np.empty(0)
            floater_hydro_caN = np.empty(0)
            floater_hydro_cpN = np.empty(0) 
            floater_hydro_cdA = np.empty(0)   
            floater_hydro_caA = np.empty(0)
            floater_hydro_cpA = np.empty(0)
            nfloater_hydro_coeffs = 0

            # if 1 global hydro coefficient is set by the user, it is used for all members
            if qb_vt['QBladeOcean']['override_morison_coefficients']:
                floater_hydro_cdN = np.append(floater_hydro_cdN, qb_vt['QBladeOcean']['HydroCdN'])
                floater_hydro_caN = np.append(floater_hydro_caN, qb_vt['QBladeOcean']['HydroCaN'])
                floater_hydro_cpN = np.append(floater_hydro_cpN, qb_vt['QBladeOcean']['HydroCpN'])    
                floater_hydro_cdA = np.append(floater_hydro_cdA, qb_vt['QBladeOcean']['HydroCdA'])
                floater_hydro_caA = np.append(floater_hydro_caA, qb_vt['QBladeOcean']['HydroCaA'])
                floater_hydro_cpA = np.append(floater_hydro_cpA, qb_vt['QBladeOcean']['HydroCpA'])   
                nfloater_hydro_coeffs = 1    

            qb_vt['QBladeOcean']['NJoints'] = n_joints
            qb_vt['QBladeOcean']['JointID'] = ijoints
            qb_vt['QBladeOcean']['Jointxi'] = joints_xyz[:,0]
            qb_vt['QBladeOcean']['Jointyi'] = joints_xyz[:,1]
            qb_vt['QBladeOcean']['Jointzi'] = joints_xyz[:,2]
            if not 'NElements' in qb_vt['QBladeOcean']: # if flexible members were defined we don't want to assign regid members as well, #TODO: should both be possible?
                qb_vt['QBladeOcean']['NElementsRigid'] = imembers
                qb_vt['QBladeOcean']['DIAMETER'] = d_coarse # members have constant diameters in QBlade
            qb_vt['QBladeOcean']['ElemID'] = imembers
            qb_vt['QBladeOcean']['MASSD'] = np.ones(n_members) * 0.0001 # no distributed mass for now, set tiny mass to not confuse QBladeOcean
            qb_vt['QBladeOcean']['NSubMembers'] = n_members
            qb_vt['QBladeOcean']['MemID'] = imembers
            qb_vt['QBladeOcean']['Jnt1ID'] = N1
            qb_vt['QBladeOcean']['Jnt2ID'] = N2
            qb_vt['QBladeOcean']['ElmID'] = imembers
            qb_vt['QBladeOcean']['ElmRot'] = np.zeros_like(imembers)
            if qb_vt['QBladeOcean']['override_morison_coefficients']:
                qb_vt['QBladeOcean']['HyCoID'] = np.ones_like(imembers) # member coefficients
                qb_vt['QBladeOcean']['AxHyCoID'] = ijoints
                qb_vt['QBladeOcean']['AxHyCoJnts'] = ijoints # joint coefficients
                qb_vt['QBladeOcean']['CdA'] = np.ones_like(ijoints)*floater_hydro_cdA
                qb_vt['QBladeOcean']['CaA'] = np.ones_like(ijoints)*floater_hydro_caA
                qb_vt['QBladeOcean']['CpA'] = np.ones_like(ijoints)*floater_hydro_cpA
            else:
                qb_vt['QBladeOcean']['HyCoID'] = np.zeros_like(imembers)
                qb_vt['QBladeOcean']['AxHyCoID'] = np.empty(0)
            qb_vt['QBladeOcean']['IsBuoy'] = np.ones_like(imembers)*qb_vt['QBladeOcean']['IsBuoy']
            qb_vt['QBladeOcean']['MaGrID'] = np.zeros_like(imembers)
            qb_vt['QBladeOcean']['FldArea'] = np.zeros_like(imembers)
            qb_vt['QBladeOcean']['MemberName'] = members_name

            # Determine discretization length of the members. The length of a discretized element is set to 10% of the distance between the joints
            ElmDsc = np.zeros(0)
            for i in range(len(imembers)):
                idx1 = N1[i] - 1
                idx2 = N2[i] - 1
                joint_distance = np.linalg.norm(joints_xyz[idx1, :] - joints_xyz[idx2, :]) # distance between joints
                if joint_distance < 10:
                    ElmDsc = np.append(ElmDsc, 1)
                else:
                    ElmDsc = np.append(ElmDsc, joint_distance //10)

            qb_vt['QBladeOcean']['ElmDsc'] = ElmDsc
        
            if modopt['flags']['mooring']:
                mooropt = modopt["mooring"]
                
                # As in OpenFAST: "Creating a line type for each line, regardless of whether it is unique or not"
                n_lines = mooropt["n_lines"]
                ilines = np.arange(n_lines, dtype=np.int_) + 1
                line_names = ['line'+str(m+1) for m in range(n_lines)]
                qb_vt['QBladeOcean']['NMooLines'] = n_lines
                qb_vt['QBladeOcean']['MooID'] = ilines
                qb_vt['QBladeOcean']['MooMass'] = inputs["line_mass_density"]
                qb_vt['QBladeOcean']['MooEA'] = inputs["line_stiffness"]

                mooring_hydro_cd = inputs["line_transverse_drag"]
                mooring_hydro_ca = inputs["line_transverse_added_mass"]
                moorint_hydro_cp = np.ones(len(inputs["line_transverse_added_mass"]))
                # these two don't really exist in QBlade for Mooring Lines
                
                # calculate E from EA
                moo_diameter = inputs['line_diameter']
                moo_area = np.pi * (moo_diameter / 2)**2
                moo_e = inputs["line_stiffness"] / moo_area
                moo_iy = np.pi * (moo_diameter / 2)**4 / 64 
                qb_vt['QBladeOcean']['MooEI'] = moo_e * moo_iy
                qb_vt['QBladeOcean']['MooDiameter'] = moo_diameter

                # Mooring members
                # TODO generalize a bit
                con1 = []
                con2 = []
                for i, row in enumerate(inputs['nodes_location_full']):
                    if mooropt["node_type"][i] == "fixed":
                        con1.append(f"GRD_{row[0]}_{row[1]}")
                    elif mooropt["node_type"][i] == "vessel":
                        con2.append(f"FLT_{row[0]}_{row[1]}_{row[2]}")
                
                ########## This is required to interpret the HEXAFLOAT ##########
                # for i, node in enumerate(mooropt['node_names']): 
                #     row = inputs['nodes_location_full'][i]
                #     if node in qb_vt['QBladeOcean']['Tendon_A']:
                #         con1.append(f"FLT_{row[0]}_{row[1]}_{row[2]}")
                #     elif node in qb_vt['QBladeOcean']['Tendon_B']:
                #         con2.append(f"FLT_{row[0]}_{row[1]}_{row[2]}")
                #         n_lines += 1
                #         qb_vt['QBladeOcean']['MooID'] = np.append(qb_vt['QBladeOcean']['MooID'],qb_vt['QBladeOcean']['MooID'][-1])

                qb_vt['QBladeOcean']['NMooMembers'] = n_lines
                qb_vt['QBladeOcean']['CONN_1'] = con1
                qb_vt['QBladeOcean']['CONN_2'] = con2
                qb_vt['QBladeOcean']['MooLength'] = inputs['unstretched_length']
                qb_vt['QBladeOcean']['MooHyCoID'] = 1 + np.arange(n_lines+len(mooring_hydro_cd)) + nfloater_hydro_coeffs # the mooring coefficients are listed below the floater coefficients
                qb_vt['QBladeOcean']['MooIsBuoy'] = np.ones(n_lines).astype(int)
                qb_vt['QBladeOcean']['MooMaGrID'] = np.zeros(n_lines).astype(int)
                qb_vt['QBladeOcean']['MooName'] = line_names

                 ########## This is required to interpret the HEXAFLOAT ##########
                # qb_vt['QBladeOcean']['MooLength'] = np.append(inputs['unstretched_length'],qb_vt['QBladeOcean']['Tendon_lengths'])
                # qb_vt['QBladeOcean']['MooName'] = np.append(line_names,qb_vt['QBladeOcean']['Tendon_names'])                
           
            # Hydrodynamic Coeffficents 
            try:
                ncoefficients = np.concatenate((floater_hydro_cdN, mooring_hydro_cd), axis=0).shape[0]
                qb_vt['QBladeOcean']['HydroCdN'] = np.concatenate((floater_hydro_cdN, mooring_hydro_cd), axis=0)
                qb_vt['QBladeOcean']['HydroCaN'] =  np.concatenate((floater_hydro_caN, mooring_hydro_ca), axis=0)
                qb_vt['QBladeOcean']['HydroCpN'] =  np.concatenate((floater_hydro_cpN, moorint_hydro_cp), axis=0)
            except NameError:
                ncoefficients = floater_hydro_cdN.shape[0]
                qb_vt['QBladeOcean']['HydroCdN'] = floater_hydro_cdN
                qb_vt['QBladeOcean']['HydroCaN'] = floater_hydro_caN
                qb_vt['QBladeOcean']['HydroCpN'] = floater_hydro_cpN
            icoefficients = np.arange(ncoefficients, dtype=np.int_) + 1
            qb_vt['QBladeOcean']['CoeffID']  =  icoefficients
            qb_vt['QBladeOcean']['MCFC']     =  np.ones_like(imembers)*qb_vt['QBladeOcean']['MCFC']
        
        # Tower inputs for rigid simulations
        if qb_vt['Turbine']['NOSTRUCTURE']:
            qb_vt['Turbine']['ROTORCONE'] = round(inputs['cone'][0], precision) 
            qb_vt['Turbine']['SHAFTTILT'] = round(inputs['tilt'][0], precision)
            qb_vt['Turbine']['OVERHANG']  = round(inputs['overhang'][0], precision)
            qb_vt['Turbine']['TOWERHEIGHT'] = inputs['hub_height'][0] # without CHRONO the tower top height is equivalent to the hub height
            qb_vt['Turbine']['TOWERTOPRAD'] = twr_aero_d_sections[-1]/2
            qb_vt['Turbine']['TOWERBOTRAD'] = twr_aero_d_sections[0]/2
            qb_vt['Turbine']['TOWERDRAG']   = qb_vt['Tower']['CD'].mean()

        ## Set initial conditions if not fully defined in the modeling file. Fully Defined means (len(wind_reference) = len(INITIAL_PITCH) = len(RPMPRESCRIBED))

        # Determine if we run steady or unsteady wind and take the respective input from the modeling options
        if qb_vt['QSim']['WNDTYPE'] == 1:
            cases = len(qb_vt['QTurbSim']['URef'])
            wind_reference = qb_vt['QTurbSim']['URef']

            # Also set some default values to make TurbSim not crash in case the user didn't provide them
            hubht = float(inputs['hub_height'][0])
            PLExp = float(inputs['shearExp'])
            if qb_vt['QTurbSim']['IECturbc'] == '':
                qb_vt['QTurbSim']['IECturbc'] = discrete_inputs['turbulence_class'] # use windIO input if not set by user
            if qb_vt['QTurbSim']['RefHt'] == 0.0:
                qb_vt['QTurbSim']['RefHt'] = hubht
            if qb_vt['QTurbSim']['HubHt'] == 0.0:
                qb_vt['QTurbSim']['HubHt'] = hubht
            if qb_vt['QTurbSim']['GridHeight'] == 0.0:
                qb_vt['QTurbSim']['GridHeight'] = 2. * hubht - 1.e-3
            if qb_vt['QTurbSim']['GridWidth'] == 0.0:
                qb_vt['QTurbSim']['GridWidth'] = 2. * hubht - 1.e-3
            if PLExp < 0:
                qb_vt['QTurbSim']['PLExp'] = PLExp
            if qb_vt['QTurbSim']['PLExp'] < 0:
                qb_vt['QTurbSim']['PLExp'] = 'default'
            if  qb_vt['QTurbSim']['AnalysisTime'] < qb_vt['QSim']['TMax']:
                qb_vt['QTurbSim']['AnalysisTime'] = qb_vt['QSim']['TMax'] 
            if qb_vt['QTurbSim']['usableTimeLabel'] < 0:
                qb_vt['QTurbSim']['usableTimeLabel'] = qb_vt['QTurbSim']['AnalysisTime']
            
            if len(qb_vt['QTurbSim']['URef']) > len(qb_vt['QTurbSim']['RandSeed1']):
                missing_count = len(qb_vt['QTurbSim']['URef']) - len(qb_vt['QTurbSim']['RandSeed1'])
                new_random_seeds = [random.randint(-2147483648, 2147483647) for _ in range(missing_count)]
                qb_vt['QTurbSim']['RandSeed1'].extend(new_random_seeds)

        else:
            cases = len(qb_vt['QSim']['MEANINF'])
            wind_reference = qb_vt['QSim']['MEANINF']

        if len(wind_reference) != len(qb_vt['QSim']['RPMPRESCRIBED']) or len(wind_reference) != len(qb_vt['QSim']['INITIAL_PITCH']):
            automatic_init_conditions = True 
            qb_vt['QSim']['RPMPRESCRIBED'] = []
            qb_vt['QSim']['INITIAL_PITCH'] = []
            logger.warning("WARNING: The input arrays for wind speed, initial rpm and pitch don't have the same length! Automatic values for 'RPMPRESCRIBED' and 'INITIAL_PITCH' are set")
        else:
            automatic_init_conditions = False

        # Only show warnings for these two once to not overload the user
        warned_yaw = False 
        warned_azimuth = False 
        for idx, u_ref in enumerate(wind_reference):
            # We have initial conditions from WISDEM
            if ('U' in inputs) and ('Omega' in inputs) and ('pitch' in inputs) and automatic_init_conditions:
                qb_vt['QSim']['RPMPRESCRIBED'].append(np.interp(u_ref, inputs['U'], inputs['Omega']))
                qb_vt['QSim']['INITIAL_PITCH'].append(np.interp(u_ref, inputs['U'], inputs['pitch']))
            elif automatic_init_conditions:
                qb_vt['QSim']['RPMPRESCRIBED'].append(qb_vt['DISCON_in']['PC_RefSpd'] * 30 / np.pi / qb_vt['Main']['GBRATIO'])
                qb_vt['QSim']['INITIAL_PITCH'].append(15)
                
            if not warned_yaw and len(qb_vt['QSim']['INITIAL_YAW']) != len(wind_reference):
                qb_vt['QSim']['INITIAL_YAW'] = np.zeros_like(wind_reference)
                logger.warning("WARNING: The input arrays for wind speed and 'INITIAL_YAW' don't have the same length. INITIAL_YAW is set to zero for each wind speed")
                warned_yaw = True

            if not warned_azimuth and len(qb_vt['QSim']['INITIAL_AZIMUTH']) != len(wind_reference):
                qb_vt['QSim']['INITIAL_AZIMUTH'] = np.zeros_like(wind_reference)
                logger.warning("WARNING: The input arrays for wind speed and 'INITIAL_AZIMUTH' don't have the same length. INITIAL_AZIMUTH is set to zero for each wind speed")
                warned_azimuth = True 
        return qb_vt

    def run_QBLADE(self, inputs, discrete_inputs, qb_vt):
        modopt          = self.options['modeling_options']
        path2qb_dll     = modopt['General']['qblade_configuration']['path2qb_dll']
        path2qb_libs    = modopt['General']['qblade_configuration']['path2qb_libs']
        self.qb_vt = qb_vt 
        
        dlc_generator = None # Do this to avoid error when no DLCs are generated

        if qb_vt['QSim']['DLCGenerator']:
            modopt = self.options['modeling_options']
            DLCs = modopt['DLC_driver']['DLCs']
            # Initialize the DLC generator
            cut_in = float(inputs['V_cutin'])
            cut_out = float(inputs['V_cutout'])
            rated = float(inputs['Vrated'])
            ws_class = discrete_inputs['turbine_class']
            wt_class = discrete_inputs['turbulence_class']
            hub_height = float(inputs['hub_height'])
            rotorD = float(inputs['Rtip'])*2.
            PLExp = float(inputs['shearExp'])
            fix_wind_seeds = modopt['DLC_driver']['fix_wind_seeds']
            fix_wave_seeds = modopt['DLC_driver']['fix_wave_seeds']
            metocean = modopt['DLC_driver']['metocean_conditions']

            U_interp = inputs['U']
            pitch_interp = inputs['pitch']
            rot_speed_interp = inputs['Omega']
            Ct_aero_interp = inputs['Ct_aero']

            # Makes life easier in post-processing
            self.qb_vt['QTurbSim']['URef']  = DLCs[0]['wind_speed']
            self.qb_vt['QSim']['STOREFROM'] = DLCs[0]['transient_time']
            # Necessary to make SIL run the appropriate number of timesteps
            self.qb_vt['QSim']['TMax'] = DLCs[0]['analysis_time'] + DLCs[0]['transient_time']
            
            tau1_const_interp = np.zeros_like(Ct_aero_interp)
            for i in range(len(Ct_aero_interp)):
                a = 1. / 2. * (1. - np.sqrt(1. - np.min([Ct_aero_interp[i],1])))    # don't allow Ct_aero > 1
                tau1_const_interp[i] = 1.1 / (1. - 1.3 * np.min([a, 0.5])) * inputs['Rtip'][0] / U_interp[i]

            initial_condition_table = {}
            initial_condition_table['U'] = U_interp
            initial_condition_table['pitch_initial'] = pitch_interp
            initial_condition_table['rot_speed_initial'] = rot_speed_interp
            initial_condition_table['Ct_aero'] = Ct_aero_interp
            initial_condition_table['tau1_const'] = tau1_const_interp

            dlc_generator = DLCGenerator(
                cut_in, 
                cut_out, 
                rated, 
                ws_class, 
                wt_class, 
                fix_wind_seeds, 
                fix_wave_seeds, 
                metocean, 
                initial_condition_table,
                )
        
            # Generate cases from user inputs
            for i_DLC in range(len(DLCs)):
                DLCopt = DLCs[i_DLC]
                dlc_generator.generate(DLCopt['DLC'], DLCopt)
            
            # Initialize parametric inputs
            WindFile_type = np.zeros(dlc_generator.n_cases, dtype=int)
            WindFile_name = [''] * dlc_generator.n_cases

            self.TMax = np.zeros(dlc_generator.n_cases)
            self.TStart = np.zeros(dlc_generator.n_cases)

            for i_case in range(dlc_generator.n_cases):
                if dlc_generator.cases[i_case].turbulent_wind:
                    # Assign values common to all DLCs
                    # Wind turbulence class
                    if dlc_generator.cases[i_case].IECturbc > 0:    # use custom TI for DLC case
                        dlc_generator.cases[i_case].IECturbc = str(dlc_generator.cases[i_case].IECturbc)
                        dlc_generator.cases[i_case].IEC_WindType = 'NTM'        # must use NTM for custom TI
                    else:
                        dlc_generator.cases[i_case].IECturbc = wt_class
                    # Reference height for wind speed
                    if not dlc_generator.cases[i_case].RefHt:   # default RefHt is 0, use hub_height if not set
                        dlc_generator.cases[i_case].RefHt = hub_height
                    # Center of wind grid (TurbSim confusingly calls it HubHt)
                    if not dlc_generator.cases[i_case].HubHt:   # default HubHt is 0, use hub_height if not set
                        dlc_generator.cases[i_case].HubHt = hub_height

                    if not dlc_generator.cases[i_case].GridHeight:   # default GridHeight is 0, use hub_height if not set
                        dlc_generator.cases[i_case].GridHeight =  2. * hub_height - 1.e-3

                    if not dlc_generator.cases[i_case].GridWidth:   # default GridWidth is 0, use hub_height if not set
                        dlc_generator.cases[i_case].GridWidth =  2. * hub_height - 1.e-3

                    # Power law exponent of wind shear
                    if dlc_generator.cases[i_case].PLExp < 0:    # use PLExp based on environment options (shear_exp), otherwise use custom DLC PLExp
                        dlc_generator.cases[i_case].PLExp = PLExp
                    # Length of wind grids
                    dlc_generator.cases[i_case].AnalysisTime = dlc_generator.cases[i_case].total_time
            
            for i_case in range(dlc_generator.n_cases):
                WindFile_type[i_case] , WindFile_name[i_case] = generate_wind_files(
                        dlc_generator, self.QBLADE_namingOut, self.wind_directory, rotorD, hub_height, self.turbsim_exe, i_case, generate_for_qblade=True)
            
            script_path = os.path.join(weis_dir, 'weis', 'aeroelasticse', 'QTurbSim.py')  # Path to the TurbSim runner script      
            wind_directory = self.wind_directory    
            number_of_workers = modopt['General']['qblade_configuration']['number_of_workers']

            # Prepare command
            turbsim_params = [
                wind_directory,
                str(number_of_workers),
            ]

            cmd = ['python', script_path] + turbsim_params

            # Run TurbSim
            subprocess.run(cmd, check=True)

            # Parameteric inputs
            case_name = []
            case_list = []
            for i_case, case_inputs in enumerate(dlc_generator.qblade_case_inputs):
                # Generate case list for DLC i
                dlc_label = DLCs[i_case]['DLC']
                case_list_i, case_name_i = CaseGen_General(case_inputs, self.QBLADE_runDirectory, self.QBLADE_InputFile, filename_ext=f'_DLC{dlc_label}_{i_case}')
                # Add DLC to case names
                case_name_i = [f'DLC{dlc_label}_{i_case}_{cni}' for cni in case_name_i]
                
                # Extend lists of cases
                case_list.extend(case_list_i)
                case_name.extend(case_name_i)

            # Apply wind files to case_list (this info will be in combined case matrix, but not individual DLCs)
            for case_i, wt, wf in zip(case_list,WindFile_type,WindFile_name):
                # TODO: not sure if we need this in QBlade but is helpful to define the wind type
                case_i[('QSim','WNDTYPE')] = wt
                case_i[('QTurbSim','TurbSimInp')] = wf
                
                # case_i[('QTurbSim','FileName_BTS')] = wf

            # Save some case info
            self.TMax = [c.total_time for c in dlc_generator.cases]
            self.TStart = [c.transient_time for c in dlc_generator.cases]
            dlc_label = [c.label for c in dlc_generator.cases]
            
            # Merge various cases into single case matrix
            case_df = pd.DataFrame(case_list)
            case_df.index = case_name
            # Add case name and dlc label to front for readability
            case_df.insert(0,'DLC',dlc_label)
            case_df.insert(0,'case_name',case_name)
            text_table = case_df.to_string(index=False)

            self.write_QBLADE_DLCGenerator(qb_vt, inputs, discrete_inputs,case_list,case_name)

            # Write the text table to a yaml, text file
            write_yaml(case_df.to_dict(),os.path.join(self.QBLADE_runDirectory,'case_matrix_combined.yaml'))
            with open(os.path.join(self.QBLADE_runDirectory,'case_matrix_combined.txt'), 'w') as file:
                file.write(text_table)

        elif qb_vt['QSim']['WNDTYPE'] == 1:
            script_path = os.path.join(weis_dir, 'weis', 'aeroelasticse', 'QTurbSim.py')  # Path to the TurbSim runner script
            wind_directory = self.wind_directory       
            number_of_workers = modopt['General']['qblade_configuration']['number_of_workers']                                        

            turbsim_params = [
                wind_directory,
                str(number_of_workers),
            ]
            cmd = ['python', script_path] + turbsim_params
            # Run TurbSim
            subprocess.run(cmd, check=True)

               
        qblade                      = qbwrap.QBladeWrapper()
        qblade.QBlade_dll           = os.path.join(weis_dir,path2qb_dll)
        qblade.QBlade_libs          = os.path.join(weis_dir,path2qb_libs)
        qblade.QBLADE_runDirectory  = self.QBLADE_runDirectory
        qblade.QBLADE_namingOut     = self.QBLADE_namingOut
        qblade.qb_vt                = self.qb_vt
        qblade.number_of_workers    = modopt['General']['qblade_configuration']['number_of_workers']
        qblade.no_structure         = modopt['Level4']['Turbine']['NOSTRUCTURE']
        qblade.store_qprs           = modopt['General']['qblade_configuration']['store_qprs']
        qblade.chunk_size           = modopt['General']['qblade_configuration']['chunk_size']
        qblade.out_file_format      = modopt['General']['qblade_configuration']['out_file_format']
        qblade.delete_out_files     = modopt['General']['qblade_configuration']['delete_out_files']
        
        qblade.channels = self.output_channels()
        magnitude_channels = dict( qbwrap.magnitude_channels_default )
        fatigue_channels =  dict( qbwrap.fatigue_channels_default )

        if not modopt['Level4']['from_qblade']:
            ## TODO: This entire section must be redone to be compatible with QBlade!! - leads to many NaN's at the moment
            for u in ['U','L']:
                blade_fatigue_root = FatigueParams(load2stress=1.0,
                                                lifetime=inputs['lifetime'],
                                                slope=inputs[f'blade_spar{u}_wohlerexp'],
                                                ult_stress=1e-3*inputs[f'blade_spar{u}_ultstress'],
                                                S_intercept=1e-3*inputs[f'blade_spar{u}_wohlerA'])
                blade_fatigue_te = FatigueParams(load2stress=1.0,
                                                lifetime=inputs['lifetime'],
                                                slope=inputs[f'blade_te{u}_wohlerexp'],
                                                ult_stress=1e-3*inputs[f'blade_te{u}_ultstress'],
                                                S_intercept=1e-3*inputs[f'blade_te{u}_wohlerA'])
                
                for k in range(1,self.n_blades+1):
                    blade_root_Fz = blade_fatigue_root.copy()
                    blade_root_Fz.load2stress = inputs[f'blade_root_spar{u}_load2stress'][2]
                    fatigue_channels[f'RootSpar{u}_Fzb{k}'] = blade_root_Fz
                    magnitude_channels[f'RootSpar{u}_Fzb{k}'] = [f'Z_b Root For. BLD {k}']

                    blade_root_Mx = blade_fatigue_root.copy()
                    blade_root_Mx.load2stress = inputs[f'blade_root_spar{u}_load2stress'][3]
                    fatigue_channels[f'RootSpar{u}_Mxb{k}'] = blade_root_Mx
                    magnitude_channels[f'RootSpar{u}_Mxb{k}'] = [f'X_b RootBend. Mom. BLD {k}']

                    blade_root_My = blade_fatigue_root.copy()
                    blade_root_My.load2stress = inputs[f'blade_root_spar{u}_load2stress'][4]
                    fatigue_channels[f'RootSpar{u}_Myb{k}'] = blade_root_My
                    magnitude_channels[f'RootSpar{u}_Myb{k}'] = [f'Y_b RootBend. Mom. BLD {k}']

                    blade_maxc_Fz = blade_fatigue_te.copy()
                    blade_maxc_Fz.load2stress = inputs[f'blade_maxc_te{u}_load2stress'][2]
                    fatigue_channels[f'Spn2te{u}_FLzb{k}'] = blade_maxc_Fz
                    magnitude_channels[f'Spn2te{u}_FLzb{k}'] = [f'Z_l For. BLD_{k} pos 0.200']

                    blade_maxc_Mx = blade_fatigue_te.copy()
                    blade_maxc_Mx.load2stress = inputs[f'blade_maxc_te{u}_load2stress'][3]
                    fatigue_channels[f'Spn2te{u}_MLxb{k}'] = blade_maxc_Mx
                    magnitude_channels[f'Spn2te{u}_MLxb{k}'] = [f'X_l Mom. BLD_{k} pos 0.200']

                    blade_maxc_My = blade_fatigue_te.copy()
                    blade_maxc_My.load2stress = inputs[f'blade_maxc_te{u}_load2stress'][4]
                    fatigue_channels[f'Spn2te{u}_MLyb{k}'] = blade_maxc_My
                    magnitude_channels[f'Spn2te{u}_MLyb{k}'] = [f'Y_l Mom. BLD_{k} pos 0.200']
            
            # Low speed shaft fatigue
            # Convert ultstress and S_intercept values to kPa with 1e-3 factor # 
            lss_fatigue = FatigueParams(load2stress=1.0,
                                        lifetime=inputs['lifetime'],
                                        slope=inputs['lss_wohlerexp'],
                                        ult_stress=1e-3*inputs['lss_ultstress'],
                                        S_intercept=1e-3*inputs['lss_wohlerA'])        
            for s in ['Ax','Sh']:
                sstr = 'axial' if s=='Ax' else 'shear'
                for ik, k in enumerate(['F','M']):
                    for ix, x in enumerate(['x','yz']):
                        idx = 3*ik+ix
                        lss_fatigue_ii = lss_fatigue.copy()
                        lss_fatigue_ii.load2stress = inputs[f'lss_{sstr}_load2stress'][idx]
                        fatigue_channels[f'LSShft{s}{k}{x}a'] = lss_fatigue_ii
                        if ix==0:
                            magnitude_channels[f'LSShft{s}{k}{x}a'] = ['X_s For. Shaft Const.'] if ik==0 else ['Aero. LSS Torque']
                        else:
                            magnitude_channels[f'LSShft{s}{k}{x}a'] = ['Y_h For. Hub Const.', 'Z_h For. Hub Const.'] if ik==0 else ['Y_h Mom. Hub Const.', 'Z_h Mom. Hub Const.'] # TODO: Equivalent sensors in QBlade are required

            # Fatigue at the tower base
            # Convert ultstress and S_intercept values to kPa with 1e-3 factor
            tower_fatigue_base = FatigueParams(load2stress=1.0,
                                               lifetime=inputs['lifetime'],
                                               slope=inputs['tower_wohlerexp'][0],
                                               ult_stress=1e-3*inputs['tower_ultstress'][0],
                                               S_intercept=1e-3*inputs['tower_wohlerA'][0])
            for s in ['Ax','Sh']:
                sstr = 'axial' if s=='Ax' else 'shear'
                for ik, k in enumerate(['For','Mom']):
                    for ix, x in enumerate(['Z','XY']):
                        idx = 3*ik+2*ix
                        tower_fatigue_ii = tower_fatigue_base.copy()
                        tower_fatigue_ii.load2stress = inputs[f'tower_{sstr}_load2stress'][0,idx]
                        fatigue_channels[f'TwrBs{s}{k}{x}t'] = tower_fatigue_ii
                        # magnitude_channels[f'TwrBs{s}{k}{x}t'] = [f'TwrBs{k}{x}t'] if x=='Z' else [f'TwrBs{k}xt', f'TwrBs{k}yt']
                        magnitude_channels[f'TwrBs{s}{k}{x}t'] = [f'{x}_tb {k}. TWR Bot. Constr.'] if x=='Z' else [f'X_tb {k}. TWR Bot. Constr.', f'Y_tb {k}. TWR Bot. Constr.']

            # Fatigue at monopile base (mudline)
            if modopt['flags']['monopile']:
                monopile_fatigue_base = FatigueParams(load2stress=1.0,
                                                      lifetime=inputs['lifetime'],
                                                      slope=inputs['monopile_wohlerexp'][0],
                                                      ult_stress=inputs['monopile_ultstress'][0],
                                                      S_intercept=inputs['monopile_wohlerA'][0])
                for s in ['Ax','Sh']:
                    sstr = 'axial' if s=='Ax' else 'shear'
                    for ik, k in enumerate(['For','Mom']):
                        for ix, x in enumerate(['Z','XY']):
                            idx = 3*ik+2*ix
                            monopile_fatigue_ii = monopile_fatigue_base.copy()
                            monopile_fatigue_ii.load2stress = inputs[f'monopile_{sstr}_load2stress'][0,idx]
                            fatigue_channels[f'M1N1{s}{k}K{x}e'] = monopile_fatigue_ii
                            # magnitude_channels[f'M1N1{s}{k}K{x}e'] = [f'M1N1{k}K{x}e'] if x=='z' else [f'M1N1{k}Kxe', f'M1N1{k}Kye']
                            magnitude_channels[f'M1N1{s}{k}K{x}e'] = [f'{x}_l {k}. SUB_member_0 pos 0.000'] if x=='Z' else [f'X_l {k}. SUB_member_0 pos 0.000', f'Y_l {k}. SUB_member_0 pos 0.000']

            # X_l For. SUB_member_{member-1} pos {rel_member_pos:.3f}
            qblade.fatigue_channels   = fatigue_channels
            qblade.magnitude_channels = magnitude_channels
            self.la = LoadsAnalysis(
                outputs=[],
                magnitude_channels=magnitude_channels,
                fatigue_channels=fatigue_channels,
            )
            self.magnitude_channels = magnitude_channels

        summary_stats, extreme_table, DELs, Damage, chan_time = qblade.run_qblade_cases()

        return summary_stats, extreme_table, DELs, Damage, chan_time, dlc_generator

    def run_TurbSim(self, qb_vt):
        self.qb_vt = qb_vt

        for idx in np.arange(len(qb_vt['QTurbSim']['URef'])):
            # add apendix based on wind speed to the file name
            QBLADE_namingOut_appendix = f'_{idx}'
            wind_directory = os.path.join(self.QBLADE_runDirectory, self.QBLADE_namingOut + QBLADE_namingOut_appendix, 'wind')
            wrapper = Turbsim_wrapper()
            wrapper.run_dir = wind_directory
            wrapper.turbsim_exe = shutil.which('turbsim')
            wrapper.turbsim_input = self.QBLADE_namingOut + QBLADE_namingOut_appendix + '.inp'
            wrapper.execute()

    def output_channels(self):

        modopt = self.options['modeling_options']       

        bld_1_stations = self.qb_vt['Main']['BLD_1']
        bld_2_stations = self.qb_vt['Main']['BLD_2']
        bld_3_stations = self.qb_vt['Main']['BLD_3']
        twr_stations = self.qb_vt['Main']['TWR']
        # Validate that all required stations are provided.
        # pCrunch relies on station data to function correctly, so this ensures 
        # no mandatory inputs are missing before proceeding.
        try:
            self.validate_stations(bld_1_stations, "BLD_1")
            self.validate_stations(bld_2_stations, "BLD_2")
            self.validate_stations(bld_3_stations, "BLD_3")
            self.validate_stations(twr_stations, "TWR")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        
        if not self.qb_vt['Turbine']['NOSTRUCTURE']:
            
            channels_out = []

            channels_out += ["Time [s]"]
            # Add distributed blade and tower stations
            for bld_station in zip(bld_1_stations, bld_2_stations, bld_3_stations):
                # blade 1
                channels_out += [f'Z_l For. BLD_1 pos {bld_station[0]:.3f} [N]']
                channels_out += [f'X_l Mom. BLD_1 pos {bld_station[0]:.3f} [Nm]']
                channels_out += [f'Y_l Mom. BLD_1 pos {bld_station[0]:.3f} [Nm]']
                channels_out += [f'Angle of Attack BLD_1 pos {bld_station[0]:.3f} [deg]']
                channels_out += [f'X_b For. BLD_1 pos {bld_station[0]:.3f} [N]']
                channels_out += [f'Y_b For. BLD_1 pos {bld_station[0]:.3f} [N]']
                channels_out += [f'Z_b For. BLD_1 pos {bld_station[0]:.3f} [N]']
                channels_out += [f'X_b Mom. BLD_1 pos {bld_station[0]:.3f} [Nm]']
                channels_out += [f'Y_b Mom. BLD_1 pos {bld_station[0]:.3f} [Nm]']
                channels_out += [f'Z_b Mom. BLD_1 pos {bld_station[0]:.3f} [Nm]']
                channels_out += [f'X_b Trl.Def. BLD_1 pos {bld_station[0]:.3f} [m]']
                channels_out += [f'Y_b Trl.Def. BLD_1 pos {bld_station[0]:.3f} [m]']
                channels_out += [f'Z_b Trl.Def. BLD_1 pos {bld_station[0]:.3f} [m]']
                channels_out += [f'X_b Rot.Def. BLD_1 pos {bld_station[0]:.3f} [deg]']
                channels_out += [f'Y_b Rot.Def. BLD_1 pos {bld_station[0]:.3f} [deg]']
                channels_out += [f'Z_b Rot.Def. BLD_1 pos {bld_station[0]:.3f} [deg]']
                channels_out += [f'X_c Aero. Force BLD_1 pos {bld_station[0]:.3f} [N/m]']
                channels_out += [f'Y_c Aero. Force BLD_1 pos {bld_station[0]:.3f} [N/m]']
                channels_out += [f'Z_l For. BLD_2 pos {bld_station[1]:.3f} [N]']
                channels_out += [f'X_l Mom. BLD_2 pos {bld_station[1]:.3f} [Nm]']
                channels_out += [f'Y_l Mom. BLD_2 pos {bld_station[1]:.3f} [Nm]']
                channels_out += [f'Angle of Attack BLD_2 pos {bld_station[1]:.3f} [deg]']
                
                if self.n_blades == 3:
                    channels_out += [f'Z_l For. BLD_3 pos {bld_station[2]:.3f} [N]']
                    channels_out += [f'X_l Mom. BLD_3 pos {bld_station[2]:.3f} [Nm]']
                    channels_out += [f'Y_l Mom. BLD_3 pos {bld_station[2]:.3f} [Nm]']
                    channels_out += [f'Angle of Attack BLD_3 pos {bld_station[2]:.3f} [deg]']
            for twr_station in twr_stations:
                channels_out += [f'X_l For. TWR pos {twr_station:.3f} [N]']
                channels_out += [f'Y_l For. TWR pos {twr_station:.3f} [N]']
                channels_out += [f'Z_l For. TWR pos {twr_station:.3f} [N]']
                channels_out += [f'X_l Mom. TWR pos {twr_station:.3f} [Nm]']
                channels_out += [f'Y_l Mom. TWR pos {twr_station:.3f} [Nm]']
                channels_out += [f'Z_l Mom. TWR pos {twr_station:.3f} [Nm]']

            channels_out += ["X_c Tip Trl.Def. (OOP) BLD 1 [m]", "Y_c Tip Trl.Def. (IP) BLD 1 [m]", "Z_c Tip Trl.Def. BLD 1 [m]", "X_c Tip Trl.Def. (OOP) BLD 2 [m]", "Y_c Tip Trl.Def. (IP) BLD 2 [m]", "Z_c Tip Trl.Def. BLD 2 [m]"]
            channels_out += ["X_c RootBend. Mom. (IP) BLD 1 [Nm]", "Y_c RootBend. Mom. (OOP) BLD 1 [Nm]", "Z_c RootBend. Mom. BLD 1 [Nm]", "X_c RootBend. Mom. (IP) BLD 2 [Nm]", "Y_c RootBend. Mom. (OOP) BLD 2 [Nm]", "Z_c RootBend. Mom. BLD 2 [Nm]"]
            channels_out += ["X_b Tip Trl.Def. (FLAP) BLD 1 [m]", "Y_b Tip Trl.Def. (EDGE) BLD 1 [m]", "Z_b Tip Trl.Def. (LONG) BLD 1 [m]", "X_b Tip Trl.Def. (FLAP) BLD 2 [m]", "Y_b Tip Trl.Def. (EDGE) BLD 2 [m]", "Z_b Tip Trl.Def. (LONG) BLD 2 [m]"]
            channels_out += ["X_b RootBend. Mom. BLD 1 [Nm]", "Y_b RootBend. Mom. BLD 1 [Nm]", "Z_b RootBend. Mom. BLD 1 [Nm]", "X_b RootBend. Mom. BLD 2 [Nm]", "Y_b RootBend. Mom. BLD 2 [Nm]", "Z_b RootBend. Mom. BLD 2 [Nm]"]
            channels_out += ["X_c Root For. BLD 1 [N]","Y_c Root For. BLD 1 [N]","Z_c Root For. BLD 1 [N]", "X_c Root For. BLD 2 [N]","Y_c Root For. BLD 2 [N]","Z_c Root For. BLD 2 [N]"]
            channels_out += ["X_b Root For. BLD 1 [N]",  "Y_b Root For. BLD 1 [N]", "Z_b Root For. BLD 1 [N]", "X_b Root For. BLD 2 [N]",  "Y_b Root For. BLD 2 [N]", "Z_b Root For. BLD 2 [N]"]
            channels_out += ["Aero. Power Coefficient [-]", "Thrust Coefficient [-]"]
            channels_out += ["Rotational Speed [rpm]", "HSS Rpm [rpm]", "Yaw Angle [deg]", "LSS Azimuthal Pos. [deg]"]
            channels_out += ["Gen. Elec. Power [W]", "Gen. HSS Torque [Nm]", "Pitch Angle Blade 1 [deg]", "Pitch Angle Blade 2 [deg]"]
            channels_out += ["Abs Inflow Vel. at Hub [m/s]", "X_g Inflow Vel. at Hub [m/s]", "Y_g Inflow Vel. at Hub [m/s]", "Z_g Inflow Vel. at Hub [m/s]"]
            channels_out += ["X_g Inflow Vel. Rotor Avg. [m/s]", "Y_g Inflow Vel. Rotor Avg. [m/s]", "Z_g Inflow Vel. Rotor Avg. [m/s]"]
            channels_out += ["X_tb For. TWR Bot. Constr. [N]", "Y_tb For. TWR Bot. Constr. [N]", "Z_tb For. TWR Bot. Constr. [N]", "X_tb Mom. TWR Bot. Constr. [Nm]", "Y_tb Mom. TWR Bot. Constr. [Nm]", "Z_tb Mom. TWR Bot. Constr. [Nm]"]
            channels_out += ["X_tt For. TWR Top Constr. [N]", "Y_tt For. TWR Top Constr. [N]", "Z_tt For. TWR Top Constr. [N]", "X_tt Mom. TWR Top Constr. [Nm]", "Y_tt Mom. TWR Top Constr. [Nm]", "Z_tt Mom. TWR Top Constr. [Nm]"]
            channels_out += ["X_h For. Hub Const. [N]", "Y_h For. Hub Const. [N]", "Z_h For. Hub Const. [N]"] # equivalent to "LSShftFxa", "LSShftFya", "LSShftFza"] rotating 
            channels_out += ["X_s For. Shaft Const. [N]", "Y_s For. Shaft Const. [N]", "Z_s For. Shaft Const. [N]"]  # ["LSShftFxs", "LSShftFys", "LSShftFzs" non-rotating
            channels_out += ["Aero. LSS Torque [Nm]", "X_s Mom. Shaft Const. [Nm]", "Y_s Mom. Shaft Const. [Nm]", "Z_s Mom. Shaft Const. [Nm]", "Y_h Mom. Hub Const. [Nm]", "Z_h Mom. Hub Const. [Nm]"]
            channels_out += ["X_n Nac. Acc. [m^2/s]", "Y_n Nac. Acc. [m^2/s]", "Z_n Nac. Acc. [m^2/s]"]
            channels_out += ["Aero. Power [W]", "Wave Elev. HYDRO WavekinEval. Pos. [m]"]
            channels_out += ["Pitch Vel. BLD 1 [deg/s]", "Pitch Vel. BLD 2 [deg/s]"]

            if self.n_blades == 3:
                channels_out += ["X_c Tip Trl.Def. (OOP) BLD 3 [m]", "Y_c Tip Trl.Def. (IP) BLD 3 [m]", "Z_c Tip Trl.Def. BLD 3 [m]"]
                channels_out += ["X_c RootBend. Mom. (IP) BLD 3 [Nm]", "Y_c RootBend. Mom. (OOP) BLD 3 [Nm]", "Z_c RootBend. Mom. BLD 3 [Nm]"]
                channels_out += ["X_b Tip Trl.Def. (FLAP) BLD 3 [m]", "Y_b Tip Trl.Def. (EDGE) BLD 3 [m]", "Z_b Tip Trl.Def. (LONG) BLD 3 [m]"]
                channels_out += ["X_b RootBend. Mom. BLD 3 [Nm]", "Y_b RootBend. Mom. BLD 3 [Nm]", "Z_b RootBend. Mom. BLD 3 [Nm]"]
                channels_out += ["X_c Root For. BLD 3 [N]","Y_c Root For. BLD 3 [N]","Z_c Root For. BLD 3 [N]"]
                channels_out += ["X_b Root For. BLD 3 [N]",  "Y_b Root For. BLD 3 [N]", "Z_b Root For. BLD 3 [N]"]
                channels_out += ["Pitch Angle Blade 3 [deg]"]
                channels_out += ["Pitch Vel. BLD 3 [deg/s]"]
            
            if modopt['flags']['floating']:
                channels_out += ["NP Trans. X_g [m]", "NP Trans. Y_g [m]", "NP Trans. Z_g [m]", "NP Roll X_l [deg]", "NP Pitch Y_l [deg]", "NP Yaw Z_l [deg]"]

            # Sensors required for monopile post-processing
            if modopt['flags']['monopile']:
                for idx, member in enumerate (self.qb_vt['QBladeOcean']['SUB_Sensors']):
                    if idx == 8:
                        rel_member_pos = 1
                    else:
                        rel_member_pos = 0
                    channels_out += [f"X_l For. SUB_member_{member-1} pos {rel_member_pos:.3f} [N]"]
                    channels_out += [f"Y_l For. SUB_member_{member-1} pos {rel_member_pos:.3f} [N]"]
                    channels_out += [f"Z_l For. SUB_member_{member-1} pos {rel_member_pos:.3f} [N]"]
                    channels_out += [f"X_l Mom. SUB_member_{member-1} pos {rel_member_pos:.3f} [Nm]"]
                    channels_out += [f"Y_l Mom. SUB_member_{member-1} pos {rel_member_pos:.3f} [Nm]"]
                    channels_out += [f"Z_l Mom. SUB_member_{member-1} pos {rel_member_pos:.3f} [Nm]"]

            # Add user defined channels from modeling_options
            if self.qb_vt['QSim']['ADDCHANNELS']:
                channels_out += self.qb_vt['QSim']['ADDCHANNELS']
        else:
            logger.warning("NOSTRUCTURE is set to True, Only channels and hence DVs, constraints and merit figures that don't depend on CHRONO are available")
            channels_out = ["Time [s]"]
            channels_out += ["Power Coefficient [-]", "Thrust Coefficient [-]"]
            channels_out += ["Rotational Speed [rpm]", "Yaw Angle [deg]"]
            channels_out += ["Pitch Angle Blade 1 [deg]", "Pitch Angle Blade 2 [deg]"]
            channels_out += ["X_g Inflow Vel. at Hub [m/s]", "Y_g Inflow Vel. at Hub [m/s]", "Z_g Inflow Vel. at Hub [m/s]"]
            channels_out += ["Aerodynamic Power [W]"]

            if self.n_blades == 3:
                    channels_out += ["Pitch Angle Blade 3 [deg]"]
        
        return channels_out

    def get_ac_axis(self, inputs):
    
        # Get the absolute offset between pitch axis (rotation center) and aerodynamic center
        ch_offset = inputs['chord'] * (inputs['ac'] - inputs['le_location'])
        # Rotate it by the twist using the AD15 coordinate system
        x , y = util.rotate(0., 0., 0., ch_offset, -np.deg2rad(inputs['theta']))
        # Apply offset to determine the AC axis
        BlCrvAC = inputs['ref_axis_blade'][:,0] + x
        BlSwpAC = inputs['ref_axis_blade'][:,1] + y
    
        return BlCrvAC, BlSwpAC

    def write_QBLADE(self, qb_vt, inputs, discrete_inputs):
        modopt = self.options['modeling_options']
        writer = InputWriter_QBlade()
        
        # For each wind speed we want to generate a QBlade simulation - so we iterate through them
        i_qb_vt = copy.deepcopy(qb_vt) # create one instance of qb_vt per case to be simulated
        
        if qb_vt['QSim']['WNDTYPE'] == 1:
            cases = len(qb_vt['QTurbSim']['URef'])
            module = 'QTurbSim'
            wind_ref = 'URef'
            i_qb_vt['QSim']['MEANINF'] = 0 # has to be set as a number and not an array in order to avoid QBlade crashing
        else:
            cases = len(qb_vt['QSim']['MEANINF'])
            module = 'QSim'
            wind_ref = 'MEANINF'
        
        for idx in range(cases):
            i_qb_vt[module][wind_ref]          = float(qb_vt[module][wind_ref][idx])
            i_qb_vt['QSim']['RPMPRESCRIBED']   = float(qb_vt['QSim']['RPMPRESCRIBED'][idx])
            i_qb_vt['QSim']['INITIAL_PITCH']   = float(qb_vt['QSim']['INITIAL_PITCH'][idx])
            i_qb_vt['QSim']['INITIAL_YAW']     = float(qb_vt['QSim']['INITIAL_YAW'][idx])
            i_qb_vt['QSim']['INITIAL_AZIMUTH'] = float(qb_vt['QSim']['INITIAL_AZIMUTH'][idx])

            if qb_vt['QBladeOcean']['override_wave']:
                i_qb_vt['QBladeOcean']['SIGHEIGHT']    = float(qb_vt['QBladeOcean']['SIGHEIGHT'][idx])
                i_qb_vt['QBladeOcean']['PEAKPERIOD']   = float(qb_vt['QBladeOcean']['PEAKPERIOD'][idx])
            
            if qb_vt['QSim']['WNDTYPE'] == 1:
                i_qb_vt['QTurbSim']['RandSeed1'] = int(qb_vt['QTurbSim']['RandSeed1'][idx])


            # add apendix based on wind speed to the file name
            QBLADE_namingOut_appendix = f'_{idx}'
            writer.qb_vt = i_qb_vt
            writer.QBLADE_runDirectory  = self.QBLADE_runDirectory
            writer.QBLADE_namingOut     = self.QBLADE_namingOut + QBLADE_namingOut_appendix

            writer.execute()

            if modopt['General']['qblade_configuration']['store_iterations']:
                writer.QBLADE_runDirectory = f"{self.QBLADE_runDirectory}/model_iterations"
                iteration = f"_it_{str(self.qb_inumber).zfill(3)}"
                
                if cases > 1:
                    case = f"_case_{idx}"
                else:
                    case = ""
                    
                writer.QBLADE_namingOut = f"{self.QBLADE_namingOut}{iteration}{case}"

                writer.execute()    

    def write_QBLADE_DLCGenerator(self, qb_vt, inputs, discrete_inputs,case_list,case_name):
        modopt = self.options['modeling_options']
        writer = InputWriter_QBlade()
        
        # For each case we want to generate a QBlade simulation - so we iterate through them
        i_qb_vt = copy.deepcopy(qb_vt) # create one instance of qb_vt per case to be simulated

        cases = len(case_name)
        
        for idx in range(cases):

            # hardcode for now
            i_qb_vt['QSim']['wave_flag']      = True
            # i_qb_vt['QBladeOcean']['WAVETYPE'] = 3
            i_qb_vt['QSim']['INITIAL_AZIMUTH'] = 0
                
            i_qb_vt['QTurbSim']['URef']         = case_list[idx][('QSim', 'MEANINF')]
            i_qb_vt['QSim']['MEANINF']          = 0
            i_qb_vt['QSim']['WNDTYPE']          = case_list[idx][('QSim', 'WNDTYPE')]
            i_qb_vt['QTurbSim']['TurbSimInp']   = case_list[idx][('QTurbSim', 'TurbSimInp')]
            i_qb_vt['QSim']['TMax']             = case_list[idx][('QSim', 'TMax')]
            i_qb_vt['QSim']['NUMTIMESTEPS']     = int(i_qb_vt['QSim']['TMax'] / i_qb_vt['QSim']['TIMESTEP'])
            i_qb_vt['QSim']['STOREFROM']        = case_list[idx][('QSim', 'STOREFROM')]
            
            i_qb_vt['QSim']['RPMPRESCRIBED']   = case_list[idx][('QSim', 'RPMPRESCRIBED')]
            i_qb_vt['QSim']['INITIAL_PITCH']   = case_list[idx][('QSim', 'INITIAL_PITCH')]
            i_qb_vt['QSim']['INITIAL_YAW']     = case_list[idx][('QSim', 'INITIAL_YAW')]
            # i_qb_vt['QSim']['INITIAL_AZIMUTH'] = 0

            i_qb_vt['QBladeOcean']['SIGHEIGHT']    = case_list[idx][('QBladeOcean', 'SIGHEIGHT')]
            i_qb_vt['QBladeOcean']['PEAKPERIOD']   = case_list[idx][('QBladeOcean', 'PEAKPERIOD')]
            # i_qb_vt['QBladeOcean']['DIRMEAN']      = case_list[idx][('QBladeOcean', 'DIRMEAN')]
            # i_qb_vt['QBladeOcean']['GAMMA']        = case_list[idx][('QBladeOcean', 'GAMMA')]
            i_qb_vt['QBladeOcean']['RANDSEED']     = case_list[idx][('QBladeOcean', 'RANDSEED')] % 65535 # 65535 is the maximum rand seed QBladeOcean allows

            # add apendix based on wind speed to the file name
            writer.qb_vt = i_qb_vt
            writer.QBLADE_runDirectory  = self.QBLADE_runDirectory
            writer.QBLADE_namingOut     = case_name[idx] # +'_U'+str(case_list[idx][('QSim', 'MEANINF')])+'_WindSeed'+str(case_list[idx][('QBladeOcean', 'RANDSEED')])+'_WaveSeed'+str(case_list[idx][('QBladeOcean', 'RANDSEED')])

            writer.execute()

            if modopt['General']['qblade_configuration']['store_iterations']:
                writer.QBLADE_runDirectory = f"{self.QBLADE_runDirectory}/model_iterations"
                iteration = f"_it_{str(self.qb_inumber).zfill(3)}"
                
                if cases > 1:
                    case = f"_case_{idx}"
                else:
                    case = ""
                    
                writer.QBLADE_namingOut = f"{self.QBLADE_namingOut}{iteration}{case}"

                writer.execute()    

    def init_QBlade_model(self):
        modopt = self.options['modeling_options']
        qb_vt = modopt['General']['qblade_configuration']['qb_vt']

        qb_vt['QSim']             = {}
        qb_vt['Main']             = {}
        qb_vt['Tower']            = {}
        qb_vt['Blade']            = {}
        qb_vt['Aero']             = {}
        qb_vt['Turbine']          = {}
        qb_vt['QBladeOcean']      = {}
        qb_vt['QTurbSim']         = {}
        qb_vt['Blade_6x6']        = {}

        qb_vt = self.load_QBlade_model_opts(qb_vt)
        return qb_vt

    def load_QBlade_model_opts(self,qb_vt,modeling_options={}):
        # Can provide own modeling options, used when we don't want to use default OpenFAST options
        if not modeling_options:
            modeling_options = self.options['modeling_options']

        if 'simulation' in modeling_options['Level4']:
            for key in modeling_options['Level4']['simulation']:
                qb_vt['QSim'][key] = modeling_options['Level4']['simulation'][key]
        if 'Main' in modeling_options['Level4']:
            for key in modeling_options['Level4']['Main']:
                qb_vt['Main'][key] = modeling_options['Level4']['Main'][key]
        if 'Tower' in modeling_options['Level4']:
            for key in modeling_options['Level4']['Tower']:
                qb_vt['Tower'][key] = modeling_options['Level4']['Tower'][key]
        if 'Blade' in modeling_options['Level4']:
            for key in modeling_options['Level4']['Blade']:
                qb_vt['Blade'][key] = modeling_options['Level4']['Blade'][key]
        if 'Aero' in modeling_options['Level4']:
            for key in modeling_options['Level4']['Aero']:
                qb_vt['Aero'][key] = modeling_options['Level4']['Aero'][key]
        if 'Turbine' in modeling_options['Level4']:
            for key in modeling_options['Level4']['Turbine']:
                qb_vt['Turbine'][key] = modeling_options['Level4']['Turbine'][key]
        if 'QBladeOcean' in modeling_options['Level4']:
            for key in modeling_options['Level4']['QBladeOcean']:
                qb_vt['QBladeOcean'][key] = modeling_options['Level4']['QBladeOcean'][key]
        if 'QTurbSim' in modeling_options['Level4']:
            for key in modeling_options['Level4']['QTurbSim']:
                qb_vt['QTurbSim'][key] = modeling_options['Level4']['QTurbSim'][key]
        return qb_vt

    def post_process(self, summary_stats, extreme_table, DELs, damage, chan_time, inputs, outputs, discrete_inputs, dlc_generator, discrete_outputs):
        # leaning heavily on equivalent funtion in "openmdao_openfast.py"
        # TODO do the post-processing acutally for DLCs and not only idealized cases
        modopt = self.options['modeling_options']
        
        if not self.qb_vt['Turbine']['NOSTRUCTURE']:
            if self.options['modeling_options']['flags']['blade']:
                outputs = self.get_blade_loading(summary_stats, extreme_table, inputs, outputs)
            if self.options['modeling_options']['flags']['tower']:
                outputs = self.get_tower_loading(summary_stats, extreme_table, inputs, outputs)
            if modopt['flags']['monopile']:
                outputs = self.get_monopile_loading(summary_stats, extreme_table, inputs, outputs)

            outputs = self.calculate_AEP(summary_stats, inputs, outputs, discrete_inputs, dlc_generator)

            outputs = self.get_weighted_DELs(DELs, damage, discrete_inputs, outputs)
            
            outputs = self.get_control_measures(summary_stats, chan_time, inputs, outputs)

            if modopt['flags']['floating']: # TODO: or (modopt['Level4']['from_qblade'] and self.qb_vt['Fst']['CompMooring']>0):
                outputs = self.get_floating_measures(summary_stats, chan_time, inputs, outputs)

            if any(summary_stats['qblade_failed']['max'] > 0):
                outputs['qblade_failed'] = 2

            # Save Data
            if modopt['General']['qblade_configuration']['save_timeseries']:
                self.save_timeseries(chan_time)

            if modopt['General']['qblade_configuration']['save_iterations']:
                self.save_iterations(summary_stats,DELs,discrete_outputs)

        else:
            outputs = self.calculate_AEP(summary_stats, inputs, outputs, discrete_inputs)

    def get_weighted_DELs(self, DELs, damage, discrete_inputs, outputs):
        modopt = self.options['modeling_options']
        if self.qb_vt['QSim']['WNDTYPE'] == 1 or self.qb_vt['QSim']['DLCGenerator']:
            U = self.qb_vt['QTurbSim']['URef']    
        else:
            U = self.qb_vt['QSim']['MEANINF']
      
        # Get wind distribution probabilities, make sure they are normalized
        pp = PowerProduction(discrete_inputs['turbine_class'])
        ws_prob = pp.prob_WindDist(U, disttype='pdf')
        ws_prob /= ws_prob.sum()

        # Scale all DELs and damage by probability and collapse over the various DLCs (inner dot product)
        # Also work around NaNs
        DELs = DELs.fillna(0.0).multiply(ws_prob, axis=0).sum()
        damage = damage.fillna(0.0).multiply(ws_prob, axis=0).sum()
        
        # Standard DELs for blade root and tower base
        outputs['DEL_RootMyb'] = np.max([DELs[f'Y_b RootBend. Mom. BLD {k+1}'] for k in range(self.n_blades)])
        outputs['DEL_TwrBsMyt'] = DELs['TwrBsM']
        outputs['DEL_TwrBsMyt_ratio'] = DELs['TwrBsM']/self.options['opt_options']['constraints']['control']['DEL_TwrBsMyt']['max']
            
        # Compute total fatigue damage in spar caps at blade root and trailing edge at max chord location
        if not modopt['Level4']['from_qblade']:
            for k in range(1,self.n_blades+1):
                for u in ['U','L']:
                    damage[f'BladeRootSpar{u}_Axial{k}'] = (damage[f'RootSpar{u}_Fzb{k}'] +
                                                        damage[f'RootSpar{u}_Mxb{k}'] +
                                                        damage[f'RootSpar{u}_Myb{k}'])
                    damage[f'BladeMaxcTE{u}_Axial{k}'] = (damage[f'Spn2te{u}_FLzb{k}'] +
                                                        damage[f'Spn2te{u}_MLxb{k}'] +
                                                        damage[f'Spn2te{u}_MLyb{k}'])

            # Compute total fatigue damage in low speed shaft, tower base, monopile base
            damage['LSSAxial'] = 0.0
            damage['LSSShear'] = 0.0
            damage['TowerBaseAxial'] = 0.0
            damage['TowerBaseShear'] = 0.0
            damage['MonopileBaseAxial'] = 0.0
            damage['MonopileBaseShear'] = 0.0
            
            for s in ['Ax','Sh']:
                sstr = 'Axial' if s=='Ax' else 'Shear'
                for ik, k in enumerate(['F','M']):
                    for ix, x in enumerate(['x','yz']):
                        damage[f'LSS{sstr}'] += damage[f'LSShft{s}{k}{x}a']

            for s in ['Ax','Sh']:
                sstr = 'Axial' if s=='Ax' else 'Shear'
                for ik, k in enumerate(['For','Mom']):
                    for ix, x in enumerate(['Z','XY']):
                        damage[f'TowerBase{sstr}'] += damage[f'TwrBs{s}{k}{x}t']
                        if modopt['flags']['monopile'] and modopt['Level3']['flag']:
                            damage[f'MonopileBase{sstr}'] += damage[f'M1N1{s}{k}K{x}e']
            
            # Assemble damages
            outputs['damage_blade_root_sparU'] = np.max([damage[f'BladeRootSparU_Axial{k+1}'] for k in range(self.n_blades)])
            outputs['damage_blade_root_sparL'] = np.max([damage[f'BladeRootSparL_Axial{k+1}'] for k in range(self.n_blades)])
            outputs['damage_blade_maxc_teU'] = np.max([damage[f'BladeMaxcTEU_Axial{k+1}'] for k in range(self.n_blades)])
            outputs['damage_blade_maxc_teL'] = np.max([damage[f'BladeMaxcTEL_Axial{k+1}'] for k in range(self.n_blades)])
            outputs['damage_lss'] = np.sqrt( damage['LSSAxial']**2 + damage['LSSShear']**2 )
            outputs['damage_tower_base'] = np.sqrt( damage['TowerBaseAxial']**2 + damage['TowerBaseShear']**2 )
            outputs['damage_monopile_base'] = np.sqrt( damage['MonopileBaseAxial']**2 + damage['MonopileBaseShear']**2 )

            # Log damages
            if self.options['opt_options']['constraints']['damage']['tower_base']['log']:
                outputs['damage_tower_base'] = np.log(outputs['damage_tower_base'])

        return outputs
    
    def calculate_AEP(self, sum_stats, inputs, outputs, discrete_inputs, dlc_generator):
        
        modopts = self.options['modeling_options']
        DLCs = [i_dlc['DLC'] for i_dlc in modopts['DLC_driver']['DLCs']]
        if 'AEP' in DLCs:
            DLC_label_for_AEP = 'AEP'
        else:
            DLC_label_for_AEP = '1.1'
            logger.warning('WARNING: DLC 1.1 is being used for AEP calculations.  Use the AEP DLC for more accurate wind modeling with constant TI.')

        if self.qb_vt['QSim']['DLCGenerator']:
            # idx_pwrcrv = []
            U = []
            for i_case in range(dlc_generator.n_cases):
                if dlc_generator.cases[i_case].label == DLC_label_for_AEP:
                    # idx_pwrcrv = np.append(idx_pwrcrv, i_case)
                    U = np.append(U, dlc_generator.cases[i_case].URef)
        elif self.qb_vt['QSim']['WNDTYPE'] == 1:
            U = self.qb_vt['QTurbSim']['URef']    
        else:
            U = self.qb_vt['QSim']['MEANINF']

        if len(U) > 1 and self.qb_vt['Turbine']['CONTROLLERTYPE'] > 0:
            #va gt
            if not self.options['modeling_options']['Floris']['flag']:
                pp = PowerProduction(discrete_inputs['turbine_class'])
            else:
                pp = PowerProduction_post(discrete_inputs['turbine_class'], inputs['site_weibull_Vmean'][0], inputs['site_weibull_shape_factor'][0])
                print('! estimating power production using site wind charactristics !')
            #va gt

            if not self.qb_vt['Turbine']['NOSTRUCTURE']:
                pwr_curve_vars_qb   = ['Gen. Elec. Power', 'Aero. Power Coefficient', 'Thrust Coefficient', 'Rotational Speed', 'Pitch Angle Blade 1']
            else:
                pwr_curve_vars_qb   = ['Aerodynamic Power', 'Power Coefficient', 'Thrust Coefficient', 'Rotational Speed', 'Pitch Angle Blade 1']
            
            pwr_curv_vars_of    = ["GenPwr", "RtFldCp", "RtFldCt", "RotSpeed", "BldPitch1"]
            rename_dict = dict(zip(pwr_curve_vars_qb, pwr_curv_vars_of))
            sum_stats_for_AEP   =  sum_stats.rename(columns=rename_dict)
            if not self.options['modeling_options']['Floris']['flag']:
                AEP, perf_data = pp.AEP(sum_stats_for_AEP, U, pwr_curv_vars_of)
            else:
                AEP, perf_data = pp.AEP_post(sum_stats_for_AEP, U, pwr_curv_vars_of)

            outputs['P_out'] = perf_data['GenPwr']['mean']
            outputs['Cp_out'] = perf_data['RtFldCp']['mean']
            outputs['Ct_out'] = perf_data['RtFldCt']['mean']
            outputs['Omega_out'] = perf_data['RotSpeed']['mean']
            outputs['pitch_out'] = perf_data['BldPitch1']['mean']
            outputs['AEP'] = AEP

        else:
            if not self.qb_vt['Turbine']['NOSTRUCTURE']:
                outputs['Cp_out']       = sum_stats['Aero. Power Coefficient']['mean'].mean()
                outputs['AEP']          = sum_stats['Gen. Elec. Power']['mean'].mean()
                outputs['P_out']        = sum_stats['Gen. Elec. Power']['mean'].iloc[0]
            else:
                outputs['Cp_out']       = sum_stats['Power Coefficient']['mean'].mean()
                outputs['AEP']          = sum_stats['Aerodynamic Power']['mean'].mean()
                outputs['P_out']        = sum_stats['Aerodynamic Power']['mean'].iloc[0]

            outputs['Ct_out']       = sum_stats['Thrust Coefficient']['mean'].mean()
            outputs['Omega_out']    = sum_stats['Rotational Speed']['mean'].mean()
            outputs['pitch_out']    = sum_stats['Pitch Angle Blade 1']['mean'].mean()                   
            logger.warning('WARNING: QBlade is not run using DLC 1.1/1.2. AEP cannot be estimated. Using average power instead.')

            outputs['V_out'] = sum_stats['X_g Inflow Vel. at Hub']['mean'].iloc[0]
        
        if len(U)>0:
            outputs['V_out'] = np.unique(U)
        else:
            outputs['V_out'] = sum_stats['X_g Inflow Vel. at Hub']['mean']

        return outputs
          	
    def get_blade_loading(self, sum_stats, extreme_table, inputs, outputs):
            """
            Find the spanwise loading along the blade span.

            Parameters
            ----------
            sum_stats : pd.DataFrame
            extreme_table : dict
            """

            # Determine maximum deflection magnitudes
            if self.n_blades == 2:
                defl_mag = [max(sum_stats['X_c Tip Trl.Def. (OOP) BLD 1']['max']), max(sum_stats['X_c Tip Trl.Def. (OOP) BLD 2']['max'])]
            else:
                defl_mag = [max(sum_stats['X_c Tip Trl.Def. (OOP) BLD 1']['max']), max(sum_stats['X_c Tip Trl.Def. (OOP) BLD 2']['max']), max(sum_stats['X_c Tip Trl.Def. (OOP) BLD 3']['max'])]
            # Get the maximum out of plane blade deflection
            outputs["max_TipDxc"] = np.max(defl_mag)

            # Return moments around x and y and axial force along blade span at instance of largest flapwise bending moment at each node
            My_chans = ["Y_b RootBend. Mom. BLD", "Y_l Mom. BLD_ pos 0.100", "Y_l Mom. BLD_ pos 0.200", "Y_l Mom. BLD_ pos 0.300", "Y_l Mom. BLD_ pos 0.400", "Y_l Mom. BLD_ pos 0.500", "Y_l Mom. BLD_ pos 0.600", "Y_l Mom. BLD_ pos 0.700", "Y_l Mom. BLD_ pos 0.800", "Y_l Mom. BLD_ pos 0.900"]
            Mx_chans = ["X_b RootBend. Mom. BLD", "X_l Mom. BLD_ pos 0.100", "X_l Mom. BLD_ pos 0.200", "X_l Mom. BLD_ pos 0.300", "X_l Mom. BLD_ pos 0.400", "X_l Mom. BLD_ pos 0.500", "X_l Mom. BLD_ pos 0.600", "X_l Mom. BLD_ pos 0.700", "X_l Mom. BLD_ pos 0.800", "X_l Mom. BLD_ pos 0.900"]
            Fz_chans = ["Z_b Root For. BLD", "Z_l For. BLD_ pos 0.100", "Z_l For. BLD_ pos 0.200", "Z_l For. BLD_ pos 0.300", "Z_l For. BLD_ pos 0.400", "Z_l For. BLD_ pos 0.500", "Z_l For. BLD_ pos 0.600", "Z_l For. BLD_ pos 0.700", "Z_l For. BLD_ pos 0.800", "Z_l For. BLD_ pos 0.900"]
                
            Fz = []
            Mx = []
            My = []
            for My_chan,Mx_chan,Fz_chan in zip(My_chans, Mx_chans, Fz_chans):
                if self.n_blades == 2:
                    if 'BLD_' in My_chan:
                        idx_BLD = My_chan.index('BLD_')
                        My_chan_bld1 = My_chan[:idx_BLD+4] + '1' + My_chan[idx_BLD+4:]
                        My_chan_bld2 = My_chan[:idx_BLD+4] + '2' + My_chan[idx_BLD+4:]
                    else:
                        idx_BLD = My_chan.index('BLD')
                        My_chan_bld1 = My_chan[:idx_BLD+3] + ' 1' + My_chan[idx_BLD+3:]
                        My_chan_bld2 = My_chan[:idx_BLD+3] + ' 2' + My_chan[idx_BLD+3:]
                    bld_idx_max = np.argmax([max(sum_stats[My_chan_bld1]['max']), max(sum_stats[My_chan_bld2]['max'])])
                else:
                    if 'BLD_' in My_chan:
                        idx_BLD = My_chan.index('BLD_')
                        My_chan_bld1 = My_chan[:idx_BLD+4] + '1' + My_chan[idx_BLD+4:]
                        My_chan_bld2 = My_chan[:idx_BLD+4] + '2' + My_chan[idx_BLD+4:]
                        My_chan_bld3 = My_chan[:idx_BLD+4] + '3' + My_chan[idx_BLD+4:]
                        bld_idx_max = np.argmax([max(sum_stats[My_chan_bld1]['max']), max(sum_stats[My_chan_bld2]['max']), max(sum_stats[My_chan_bld3]['max'])])
                        # TODO what about Mz here?
                        My_max_chan = My_chan[:idx_BLD+4] + str(bld_idx_max+1) + My_chan[idx_BLD+4:]
                        My.append(extreme_table[My_max_chan][np.argmax(sum_stats[My_max_chan]['max'])][My_chan[:idx_BLD+4] + str(bld_idx_max+1) + My_chan[idx_BLD+4:]])
                        Mx.append(extreme_table[My_max_chan][np.argmax(sum_stats[My_max_chan]['max'])][Mx_chan[:idx_BLD+4] + str(bld_idx_max+1) + Mx_chan[idx_BLD+4:]])
                        Fz.append(extreme_table[My_max_chan][np.argmax(sum_stats[My_max_chan]['max'])][Fz_chan[:idx_BLD+4] + str(bld_idx_max+1) + Fz_chan[idx_BLD+4:]])
                    else:
                        idx_BLD = My_chan.index('BLD')
                        My_chan_bld1 = My_chan[:idx_BLD+3] + ' 1' + My_chan[idx_BLD+3:]
                        My_chan_bld2 = My_chan[:idx_BLD+3] + ' 2' + My_chan[idx_BLD+3:]
                        My_chan_bld3 = My_chan[:idx_BLD+3] + ' 3' + My_chan[idx_BLD+3:]
                        bld_idx_max = np.argmax([max(sum_stats[My_chan_bld1]['max']), max(sum_stats[My_chan_bld2]['max']), max(sum_stats[My_chan_bld3]['max'])])
                        My_max_chan = My_chan + f" {bld_idx_max+1}"
                        My.append(extreme_table[My_max_chan][np.argmax(sum_stats[My_max_chan]['max'])][My_chan+ f" {bld_idx_max+1}"])
                        Mx.append(extreme_table[My_max_chan][np.argmax(sum_stats[My_max_chan]['max'])][Mx_chan+ f" {bld_idx_max+1}"])
                        Fz.append(extreme_table[My_max_chan][np.argmax(sum_stats[My_max_chan]['max'])][Fz_chan+ f" {bld_idx_max+1}"])


            if np.any(np.isnan(Fz)):
                logger.warning('WARNING: nans found in Fz extremes')
                Fz[np.isnan(Fz)] = 0.0
            if np.any(np.isnan(Mx)):
                logger.warning('WARNING: nans found in Mx extremes')
                Mx[np.isnan(Mx)] = 0.0
            if np.any(np.isnan(My)):
                logger.warning('WARNING: nans found in My extremes')
                My[np.isnan(My)] = 0.0
            
            blade_grid_local = np.linspace(0,0.9,10) * inputs['ref_axis_blade'][-1,2]
            blade_grid = np.hstack((blade_grid_local + inputs['Rhub'], inputs['Rtip']))
            spline_Fz = PchipInterpolator(blade_grid, np.hstack((Fz, 0.)))
            spline_Mx = PchipInterpolator(blade_grid, np.hstack((Mx, 0.)))
            spline_My = PchipInterpolator(blade_grid, np.hstack((My, 0.)))

            r = inputs['r']
            Fz_out = spline_Fz(r).flatten()
            Mx_out = spline_Mx(r).flatten()
            My_out = spline_My(r).flatten()

            outputs['blade_maxTD_Mx'] = Mx_out
            outputs['blade_maxTD_My'] = My_out
            outputs['blade_maxTD_Fz'] = Fz_out

            # Determine maximum root moment
            if self.n_blades == 2:
                blade_root_flap_moment = max([max(sum_stats['Y_b RootBend. Mom. BLD 1']['max']), max(sum_stats['Y_b RootBend. Mom. BLD 2']['max'])])
                blade_root_oop_moment  = max([max(sum_stats['Y_c RootBend. Mom. (OOP) BLD 1']['max']), max(sum_stats['Y_c RootBend. Mom. (OOP) BLD 2']['max'])])
                blade_root_tors_moment  = max([max(sum_stats['Z_b RootBend. Mom. BLD 1']['max']), max(sum_stats['Z_b RootBend. Mom. BLD 2']['max'])])
            else:
                blade_root_flap_moment = max([max(sum_stats['Y_b RootBend. Mom. BLD 1']['max']), max(sum_stats['Y_b RootBend. Mom. BLD 2']['max']), max(sum_stats['Y_b RootBend. Mom. BLD 3']['max'])])
                blade_root_oop_moment  = max([max(sum_stats['Y_c RootBend. Mom. (OOP) BLD 1']['max']), max(sum_stats['Y_c RootBend. Mom. (OOP) BLD 2']['max']), max(sum_stats['Y_c RootBend. Mom. (OOP) BLD 3']['max'])])
                blade_root_tors_moment  = max([max(sum_stats['Z_b RootBend. Mom. BLD 1']['max']), max(sum_stats['Z_b RootBend. Mom. BLD 2']['max']), max(sum_stats['Z_b RootBend. Mom. BLD 3']['max'])])
            
            outputs['max_RootMyb'] = blade_root_flap_moment
            outputs['max_RootMyc'] = blade_root_oop_moment
            outputs['max_RootMzb'] = blade_root_tors_moment

            ## Get hub moments and forces in the non-rotating frame
            outputs['hub_Fxyz'] = np.array([extreme_table['LSShftF'][np.argmax(sum_stats['LSShftF']['max'])]['X_s For. Shaft Const.'],
                                        extreme_table['LSShftF'][np.argmax(sum_stats['LSShftF']['max'])]['Y_s For. Shaft Const.'],
                                        extreme_table['LSShftF'][np.argmax(sum_stats['LSShftF']['max'])]['Z_s For. Shaft Const.']]) # TODO why the scalin if already in kN*1.e3
            outputs['hub_Mxyz'] = np.array([extreme_table['LSShftM'][np.argmax(sum_stats['LSShftM']['max'])]['X_s Mom. Shaft Const.'],
                                        extreme_table['LSShftM'][np.argmax(sum_stats['LSShftM']['max'])]['Y_s Mom. Shaft Const.'],
                                        extreme_table['LSShftM'][np.argmax(sum_stats['LSShftM']['max'])]['Z_s Mom. Shaft Const.']]) # TODO why the scalin if already in kN*1.e3

            ## Post process aerodynamic data
            # Angles of attack - max, std, mean
            blade1_chans_aoa = ["Angle of Attack BLD_1 pos 0.100", "Angle of Attack BLD_1 pos 0.200", "Angle of Attack BLD_1 pos 0.300", "Angle of Attack BLD_1 pos 0.400", "Angle of Attack BLD_1 pos 0.500", "Angle of Attack BLD_1 pos 0.600", "Angle of Attack BLD_1 pos 0.700", "Angle of Attack BLD_1 pos 0.800", "Angle of Attack BLD_1 pos 0.900"]
            blade2_chans_aoa = ["Angle of Attack BLD_2 pos 0.100", "Angle of Attack BLD_2 pos 0.200", "Angle of Attack BLD_2 pos 0.300", "Angle of Attack BLD_2 pos 0.400", "Angle of Attack BLD_2 pos 0.500", "Angle of Attack BLD_2 pos 0.600", "Angle of Attack BLD_2 pos 0.700", "Angle of Attack BLD_2 pos 0.800", "Angle of Attack BLD_2 pos 0.900"]
            
            aoa_max_B1  = [np.max(sum_stats[var]['max'])    for var in blade1_chans_aoa]
            aoa_mean_B1 = [np.mean(sum_stats[var]['mean'])  for var in blade1_chans_aoa]
            aoa_std_B1  = [np.mean(sum_stats[var]['std'])   for var in blade1_chans_aoa]
            aoa_max_B2  = [np.max(sum_stats[var]['max'])    for var in blade2_chans_aoa]
            aoa_mean_B2 = [np.mean(sum_stats[var]['mean'])  for var in blade2_chans_aoa]
            aoa_std_B2  = [np.mean(sum_stats[var]['std'])   for var in blade2_chans_aoa]
            if self.n_blades == 2:
                spline_aoa_max      = PchipInterpolator(self.R_out_AD, np.max([aoa_max_B1, aoa_max_B2], axis=0))
                spline_aoa_std      = PchipInterpolator(self.R_out_AD, np.mean([aoa_std_B1, aoa_std_B2], axis=0))
                spline_aoa_mean     = PchipInterpolator(self.R_out_AD, np.mean([aoa_mean_B1, aoa_mean_B2], axis=0))
            elif self.n_blades == 3:
                blade3_chans_aoa = ["Angle of Attack BLD_3 pos 0.100", "Angle of Attack BLD_3 pos 0.200", "Angle of Attack BLD_3 pos 0.300", "Angle of Attack BLD_3 pos 0.400", "Angle of Attack BLD_3 pos 0.500", "Angle of Attack BLD_3 pos 0.600", "Angle of Attack BLD_3 pos 0.700", "Angle of Attack BLD_3 pos 0.800", "Angle of Attack BLD_3 pos 0.900"]
                aoa_max_B3          = [np.max(sum_stats[var]['max'])    for var in blade3_chans_aoa]
                aoa_mean_B3         = [np.mean(sum_stats[var]['mean'])  for var in blade3_chans_aoa]
                aoa_std_B3          = [np.mean(sum_stats[var]['std'])   for var in blade3_chans_aoa]
                spline_aoa_max      = PchipInterpolator(self.R_out_AD, np.max([aoa_max_B1, aoa_max_B2, aoa_max_B3], axis=0))
                spline_aoa_std      = PchipInterpolator(self.R_out_AD, np.mean([aoa_max_B1, aoa_std_B2, aoa_std_B3], axis=0))
                spline_aoa_mean     = PchipInterpolator(self.R_out_AD, np.mean([aoa_mean_B1, aoa_mean_B2, aoa_mean_B3], axis=0))
            else:
                raise Exception('The calculations only support 2 or 3 bladed rotors')

            outputs['max_aoa']  = spline_aoa_max(r)
            outputs['std_aoa']  = spline_aoa_std(r)
            outputs['mean_aoa'] = spline_aoa_mean(r)

            return outputs

    def get_tower_loading(self, sum_stats, extreme_table, inputs, outputs):
        """
        Find the loading along the tower height.

        Parameters
        ----------
        sum_stats : pd.DataFrame
        extreme_table : dict
        """
        tower_chans_Fx = ["X_tb For. TWR Bot. Constr.", "X_l For. TWR pos 0.100", "X_l For. TWR pos 0.200", "X_l For. TWR pos 0.300", "X_l For. TWR pos 0.400", "X_l For. TWR pos 0.500", "X_l For. TWR pos 0.600", "X_l For. TWR pos 0.700", "X_l For. TWR pos 0.800", "X_l For. TWR pos 0.900", "X_tt For. TWR Top Constr."]
        tower_chans_Fy = ["Y_tb For. TWR Bot. Constr.", "Y_l For. TWR pos 0.100", "Y_l For. TWR pos 0.200", "Y_l For. TWR pos 0.300", "Y_l For. TWR pos 0.400", "Y_l For. TWR pos 0.500", "Y_l For. TWR pos 0.600", "Y_l For. TWR pos 0.700", "Y_l For. TWR pos 0.800", "Y_l For. TWR pos 0.900", "Y_tt For. TWR Top Constr."]
        tower_chans_Fz = ["Z_tb For. TWR Bot. Constr.", "Z_l For. TWR pos 0.100", "Z_l For. TWR pos 0.200", "Z_l For. TWR pos 0.300", "Z_l For. TWR pos 0.400", "Z_l For. TWR pos 0.500", "Z_l For. TWR pos 0.600", "Z_l For. TWR pos 0.700", "Z_l For. TWR pos 0.800", "Z_l For. TWR pos 0.900", "Z_tt For. TWR Top Constr."]
        tower_chans_Mx = ["X_tb Mom. TWR Bot. Constr.", "X_l Mom. TWR pos 0.100", "X_l Mom. TWR pos 0.200", "X_l Mom. TWR pos 0.300", "X_l Mom. TWR pos 0.400", "X_l Mom. TWR pos 0.500", "X_l Mom. TWR pos 0.600", "X_l Mom. TWR pos 0.700", "X_l Mom. TWR pos 0.800", "X_l Mom. TWR pos 0.900", "X_tt Mom. TWR Top Constr."]
        tower_chans_My = ["Y_tb Mom. TWR Bot. Constr.", "Y_l Mom. TWR pos 0.100", "Y_l Mom. TWR pos 0.200", "Y_l Mom. TWR pos 0.300", "Y_l Mom. TWR pos 0.400", "Y_l Mom. TWR pos 0.500", "Y_l Mom. TWR pos 0.600", "Y_l Mom. TWR pos 0.700", "Y_l Mom. TWR pos 0.800", "Y_l Mom. TWR pos 0.900", "Y_tt Mom. TWR Top Constr."]
        tower_chans_Mz = ["Z_tb Mom. TWR Bot. Constr.", "Z_l Mom. TWR pos 0.100", "Z_l Mom. TWR pos 0.200", "Z_l Mom. TWR pos 0.300", "Z_l Mom. TWR pos 0.400", "Z_l Mom. TWR pos 0.500", "Z_l Mom. TWR pos 0.600", "Z_l Mom. TWR pos 0.700", "Z_l Mom. TWR pos 0.800", "Z_l Mom. TWR pos 0.900", "Z_tt Mom. TWR Top Constr."]

        fatb_max_chan   = "Y_tb Mom. TWR Bot. Constr."

        # Get the maximum fore-aft moment at tower base, 
        # We use OF channel naming convention from here on out to be able to use the standard constraint convetnions
        outputs["max_TwrBsMyt"] = np.max(sum_stats[fatb_max_chan]['max'])
        outputs["max_TwrBsMyt_ratio"] = np.max(sum_stats[fatb_max_chan]['max'])/self.options['opt_options']['constraints']['control']['Max_TwrBsMyt']['max']
        # Return forces and moments along tower height at instance of largest fore-aft tower base moment
        Fx = [extreme_table[fatb_max_chan][np.argmax(sum_stats[fatb_max_chan]['max'])][var] for var in tower_chans_Fx]
        Fy = [extreme_table[fatb_max_chan][np.argmax(sum_stats[fatb_max_chan]['max'])][var] for var in tower_chans_Fy]
        Fz = [extreme_table[fatb_max_chan][np.argmax(sum_stats[fatb_max_chan]['max'])][var] for var in tower_chans_Fz]
        Mx = [extreme_table[fatb_max_chan][np.argmax(sum_stats[fatb_max_chan]['max'])][var] for var in tower_chans_Mx]
        My = [extreme_table[fatb_max_chan][np.argmax(sum_stats[fatb_max_chan]['max'])][var] for var in tower_chans_My]
        Mz = [extreme_table[fatb_max_chan][np.argmax(sum_stats[fatb_max_chan]['max'])][var] for var in tower_chans_Mz]

        # Spline results on tower basic grid
        tower_grid = np.linspace(0,1,11) # we require this spacing for WEIS/QBlade
        spline_Fx      = PchipInterpolator(tower_grid, Fx)
        spline_Fy      = PchipInterpolator(tower_grid, Fy)
        spline_Fz      = PchipInterpolator(tower_grid, Fz)
        spline_Mx      = PchipInterpolator(tower_grid, Mx)
        spline_My      = PchipInterpolator(tower_grid, My)
        spline_Mz      = PchipInterpolator(tower_grid, Mz)

        z_full = inputs['tower_z_full']
        z_sec, _ = util.nodal2sectional(z_full)
        z = (z_sec - z_sec[0]) / (z_sec[-1] - z_sec[0])

        outputs['tower_maxMy_Fx'] = spline_Fx(z)
        outputs['tower_maxMy_Fy'] = spline_Fy(z)
        outputs['tower_maxMy_Fz'] = spline_Fz(z)
        outputs['tower_maxMy_Mx'] = spline_Mx(z)
        outputs['tower_maxMy_My'] = spline_My(z)
        outputs['tower_maxMy_Mz'] = spline_Mz(z)
        
        return outputs

    def get_monopile_loading(self, sum_stats, extreme_table, inputs, outputs):
        """
        Find the loading along the monopile length.

        Parameters
        ----------
        sum_stats : pd.DataFrame
        extreme_table : dict
        """

        monopile_chans_Fx = []
        monopile_chans_Fy = []
        monopile_chans_Fz = []
        monopile_chans_Mx = []
        monopile_chans_My = []
        monopile_chans_Mz = []
        
        for idx, member in enumerate (self.qb_vt['QBladeOcean']['SUB_Sensors']):
            if idx == 8:
                rel_member_pos = 1
            else:
                rel_member_pos = 0
            monopile_chans_Fx += [f"X_l For. SUB_member_{member-1} pos {rel_member_pos:.3f}"]
            monopile_chans_Fy += [f"Y_l For. SUB_member_{member-1} pos {rel_member_pos:.3f}"]
            monopile_chans_Fz += [f"Z_l For. SUB_member_{member-1} pos {rel_member_pos:.3f}"]
            monopile_chans_Mx += [f"X_l Mom. SUB_member_{member-1} pos {rel_member_pos:.3f}"]
            monopile_chans_My += [f"Y_l Mom. SUB_member_{member-1} pos {rel_member_pos:.3f}"]
            monopile_chans_Mz += [f"Z_l Mom. SUB_member_{member-1} pos {rel_member_pos:.3f}"]

        max_chan   = "Y_l Mom. SUB_member_0 pos 0.000"

        # # Get the maximum of signal M1N1MKye
        outputs["max_M1N1MKye"] = np.max(sum_stats[max_chan]['max'])
        # # Return forces and moments along monopile at instance of largest fore-aft tower base moment
        Fx = [extreme_table[max_chan][np.argmax(sum_stats[max_chan]['max'])][var] for var in monopile_chans_Fx]
        Fy = [extreme_table[max_chan][np.argmax(sum_stats[max_chan]['max'])][var] for var in monopile_chans_Fy]
        Fz = [extreme_table[max_chan][np.argmax(sum_stats[max_chan]['max'])][var] for var in monopile_chans_Fz]
        Mx = [extreme_table[max_chan][np.argmax(sum_stats[max_chan]['max'])][var] for var in monopile_chans_Mx]
        My = [extreme_table[max_chan][np.argmax(sum_stats[max_chan]['max'])][var] for var in monopile_chans_My]
        Mz = [extreme_table[max_chan][np.argmax(sum_stats[max_chan]['max'])][var] for var in monopile_chans_Mz]

        # # Spline results on grid of channel locations along the monopile
        spline_Fx      = PchipInterpolator(self.Z_out_QBO_mpl, Fx)
        spline_Fy      = PchipInterpolator(self.Z_out_QBO_mpl, Fy)
        spline_Fz      = PchipInterpolator(self.Z_out_QBO_mpl, Fz)
        spline_Mx      = PchipInterpolator(self.Z_out_QBO_mpl, Mx)
        spline_My      = PchipInterpolator(self.Z_out_QBO_mpl, My)
        spline_Mz      = PchipInterpolator(self.Z_out_QBO_mpl, Mz)

        z_full = inputs['monopile_z_full']
        z_sec, _ = util.nodal2sectional(z_full)
        z = (z_sec - z_sec[0]) / (z_sec[-1] - z_sec[0])

        # QBladeOcean reports in N, but ElastoDyn and units here report in kN, so scale by 0.001
        outputs['monopile_maxMy_Fx'] = 1e-3*spline_Fx(z)
        outputs['monopile_maxMy_Fy'] = 1e-3*spline_Fy(z)
        outputs['monopile_maxMy_Fz'] = 1e-3*spline_Fz(z)
        outputs['monopile_maxMy_Mx'] = 1e-3*spline_Mx(z)
        outputs['monopile_maxMy_My'] = 1e-3*spline_My(z)
        outputs['monopile_maxMy_Mz'] = 1e-3*spline_Mz(z)

        return outputs
    
    def get_control_measures(self, sum_stats, chan_time, inputs, outputs):
        '''
        calculate control measures:
            - rotor_overspeed

        given:
            - sum_stats : pd.DataFrame
        '''

        # rotor overspeed
        outputs['rotor_overspeed'] = (np.max(sum_stats['HSS Rpm']['max']) * np.pi/30. / self.qb_vt['DISCON_in']['PC_RefSpd'] ) - 1.0

        # nacelle accelleration
        outputs['max_nac_accel'] = sum_stats['NcIMUTA']['max'].max()

        # Max pitch rate
        max_pitch_rates = np.r_[sum_stats['Pitch Vel. BLD 1']['max'],sum_stats['Pitch Vel. BLD 2']['max'],sum_stats['Pitch Vel. BLD 3']['max']]
        outputs['max_pitch_rate_sim'] = max(max_pitch_rates)  / np.rad2deg(self.qb_vt['DISCON_in']['PC_MaxRat'])        # normalize by ROSCO pitch rate

        # pitch travel and duty cycle
        if self.options['modeling_options']['General']['qblade_configuration']['keep_time']: # TODO keep time is a dummy variable in QBlade for now
            tot_time = 0
            tot_travel = 0
            num_dir_changes = 0
            for i_ts, ts in enumerate(chan_time):
                t_span = ts['Time'][-1] - ts['Time'][0]
                for i_blade in range(self.qb_vt['Main']['NUMBLD']):
                    ts[f'dBldPitch{i_blade+1}'] = np.r_[0,np.diff(ts[f'Pitch Angle Blade {i_blade+1}'])] / self.qb_vt['QSim']['TIMESTEP']

                    time_ind = ts['Time'] >= ts['Time'][0]

                    # total time
                    tot_time += t_span

                    # total pitch travel (\int |\dot{\frac{d\theta}{dt}| dt)
                    tot_travel += np.trapz(np.abs(ts[f'dBldPitch{i_blade+1}'])[time_ind], x=ts['Time'][time_ind])

                    # number of direction changes on each blade
                    num_dir_changes += np.sum(np.abs(np.diff(np.sign(ts[f'dBldPitch{i_blade+1}'][time_ind])))) / 2

            # Normalize by number of blades, total time
            avg_travel_per_sec = tot_travel / self.qb_vt['Main']['NUMBLD']  / tot_time
            outputs['avg_pitch_travel'] = avg_travel_per_sec

            dir_change_per_sec = num_dir_changes / self.qb_vt['Main']['NUMBLD']  / tot_time
            outputs['pitch_duty_cycle'] = dir_change_per_sec
        else:
            logger.warning('openmdao_qblade warning: avg_pitch_travel, and pitch_duty_cycle require keep_time = True')

        return outputs
    
    def get_floating_measures(self, sum_stats, chan_time, inputs, outputs):
        '''
        calculate floating measures:
            - Std_PtfmPitch (max over all dlcs if constraint, mean otheriwse)
            - Max_PtfmPitch

        given:
            - sum_stats : pd.DataFrame
        '''

        if self.options['opt_options']['constraints']['control']['Std_PtfmPitch']['flag']:
            outputs['Std_PtfmPitch'] = np.max(sum_stats['NP Pitch Y_l']['std'])
        else:
            outputs['Std_PtfmPitch'] = np.mean(sum_stats['NP Pitch Y_l']['std'])

        outputs['Max_PtfmPitch']  = np.max(sum_stats['NP Pitch Y_l']['max'])

        # Max platform offset        
        for timeseries in chan_time:
            max_offset_ts = np.sqrt(timeseries['NP Trans. X_g']**2 + timeseries['NP Trans. Y_g']**2).max()
            outputs['Max_Offset'] = np.r_[outputs['Max_Offset'],max_offset_ts].max()

        return outputs

    def calc_fractional_curved_length(self, control_points):
        # function returns the total curved blade length and the fractional length at each control point
        distances = np.diff(control_points, axis=0)
        segment_lengths = np.sqrt(np.sum(distances**2, axis=1))
        cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
        curved_length = cumulative_lengths[-1]
        fractional_length = cumulative_lengths/curved_length
        return curved_length, fractional_length
    
    def calculate_num_samples(self, distance):
        if distance <= 1.0:
            return 2
        elif distance <= 10.0:
            return int(np.ceil(distance / 1.0))
        elif distance <= 25.0:
            return int(np.ceil(distance / 2.0))
        elif distance <= 50.0:
            return int(np.ceil(distance / 5.0))
        else:
            return int(np.ceil(distance / 10.0))
        
    def validate_stations(self, station_array, component):
        required_stations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        missing_stations = [station for station in required_stations if station not in station_array]
        if missing_stations:
            raise ValueError(f"{component} is missing the following required stations: {missing_stations}, please modify the modeling file accordingly")
        
    def save_iterations(self,summ_stats,DELs,discrete_outputs):

        # Make iteration directory
        save_dir = os.path.join(self.QBLADE_runDirectory,'iteration_'+str(self.qb_inumber))
        os.makedirs(save_dir, exist_ok=True)

        # Save dataframes as pickles
        summ_stats.to_pickle(os.path.join(save_dir,'summary_stats.p'))
        DELs.to_pickle(os.path.join(save_dir,'DELs.p'))

        # Save qb_vt as pickle
        with open(os.path.join(save_dir,'qb_vt.p'), 'wb') as f:
            pickle.dump(self.qb_vt,f)

        discrete_outputs['ts_out_dir'] = save_dir

    def save_timeseries(self,chan_time):

        # Make iteration directory
        save_dir = os.path.join(self.QBLADE_runDirectory,'iteration_'+str(self.qb_inumber),'timeseries')
        os.makedirs(save_dir, exist_ok=True)

        # Save each timeseries as a pickled dataframe
        for i_ts, timeseries in enumerate(chan_time):
            output = OpenFASTOutput.from_dict(timeseries, self.QBLADE_namingOut)
            output.df.to_pickle(os.path.join(save_dir,self.QBLADE_namingOut + '_' + str(i_ts) + '.p'))

## va gt
class PowerProduction_post:
    """Class to generate power production estimates."""

    def __init__(self, turbine_class, weibull_Vmean=None, weibull_k=None, **kwargs):
        """
        Creates an instance of `PowerProduction`.

        Parameters
        ----------
        turbine_class : int
        """

        self.turbine_class = turbine_class
        self.weibull_Vmean=weibull_Vmean
        self.weibull_k=weibull_k

    def prob_WindDist_post(self, windspeed, disttype="pdf", weibull_Vmean=None, weibull_k=None):
        """
        Generates the probability of a windspeed given the cumulative
        distribution or probability density function of a Weibull distribution
        per IEC 61400, or for a given average windspeed and shape parameter

        NOTE: This uses the range of wind speeds simulated over, so if the
        simulated wind speed range is not indicative of operation range, using
        this cdf to calculate AEP is invalid

        Parameters
        ----------
        windspeed : float or list-like
            wind speed(s) to calculate probability of
        disttype : str, optional
            type of probability, currently supports CDF or PDF

        Returns
        -------
        p_bin : list
            list containing probabilities per wind speed bin
        """
        # Define parameters
        if not (weibull_Vmean is None or weibull_k is None):
            Vavg = weibull_Vmean
            k = weibull_k
            c=Vavg/(gamma(1+1/k))
            print(f'!! shape factor: k = {k:.3f}    mean wind speed: Vavg = {Vavg:.3f} m/s !!')
        else:
            if self.turbine_class in [1, "I"]:
                Vavg = 50 * 0.2
            elif self.turbine_class in [2, "II"]:
                Vavg = 42.5 * 0.2
            elif self.turbine_class in [3, "III"]:
                Vavg = 37.5 * 0.2
            
            k = 2  # Weibull shape parameter
            c = (2 * Vavg) / np.sqrt(np.pi)  # Weibull scale parameter

        if disttype.lower() == "cdf":
            # Calculate probability of wind speed based on WeibulCDF
            wind_prob = 1 - np.exp(-(windspeed / c) ** k)

        elif disttype.lower() == "pdf":
            # Calculate probability of wind speed based on WeibulPDF
            wind_prob = (
                (k / c)
                * (windspeed / c) ** (k - 1)
                * np.exp(-(windspeed / c) ** k)
            )

        else:
            raise ValueError(
                f"The {disttype} probability distribution type is invalid"
            )

        return wind_prob

    def AEP_post(self, stats, windspeeds, pwr_curve_vars):
        """
        Calculate AEP for simulation cases

        Parameters:
        ----------
        stats : pd.DataFrame
            DataFrame containing summary statistics of each DLC.
        windspeeds : list-like
            List of wind speed values corresponding to each power output in the stats input
            for a single dataset

        Returns:
        --------
        AEP : list
            List of annual energy productions.
        """

        assert len(stats) == len(windspeeds)

        pwr = stats.loc[:, ("GenPwr", "mean")].to_frame()

        # Group and average powers by wind speeds
        pwr["windspeeds"] = windspeeds
        pwr = pwr.groupby("windspeeds").mean()

        # Wind probability
        unique = list(np.unique(windspeeds))
        #va gt
        wind_prob = self.prob_WindDist_post(unique, disttype="pdf", weibull_Vmean=self.weibull_Vmean, weibull_k=self.weibull_k)
        #va gt

        # Calculate AEP
        AEP = np.trapz(pwr.T * wind_prob, unique) * 8760

        perf_data = {"U": unique}
        for var in pwr_curve_vars:
            try:
                perf_array = stats.loc[:, (var, "mean")].to_frame()
            except KeyError:
                print(var,"not found. . . continuing")
                continue
            perf_array["windspeed"] = windspeeds
            perf_array = perf_array.groupby("windspeed").mean()
            perf_data[var] = perf_array[var]

        return AEP, perf_data

#va gt