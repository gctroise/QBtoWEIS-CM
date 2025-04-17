"""Wake loss factor API"""

from weis.wakelossfact.wlf_calculation import calculationWLF as wlf

import openmdao.api as om

import numpy as np
import ruamel.yaml as ry
from pathlib import Path
import os

from scipy.special import gamma
from scipy.stats import weibull_min

import pandas as pd

from floris.flow_visualization import calculate_horizontal_plane_with_turbines, show, visualize_cut_plane
from floris.layout_visualization import plot_turbine_rotors, plot_turbine_labels

from floris import (
    FlorisModel,
    TimeSeries,
    WindRose,
    turbine_library
)
from floris.utilities import wrap_360

# import random
# import sys

# def load_yaml(fname_input : str) -> dict:
#     """
#     Reads and parses a YAML file in a safe mode using the ruamel.yaml library.

#     Args:
#         fname_input (str): Path to the YAML file to be loaded.

#     Returns:
#         dict: Parsed YAML content as a dictionary.
#     """
#     reader = ry.YAML(typ="safe", pure=True)
#     with open(fname_input, "r", encoding="utf-8") as f:
#         input_yaml = reader.load(f)
#     return input_yaml

# def append_to_yaml(file_path, data_to_append):
#     yaml = ry.YAML()
#     with open(file_path, 'wb') as f:
#         yaml.dump(data_to_append, f)

# def write_turb_file(Vout, Pout, Ctout, temp_floris_filename, temp_turbine_name):
#     yaml_floris=load_yaml(temp_floris_filename)
#     yaml_turbine=load_yaml(temp_turbine_name)

#     path_floris = Path(f'home/spl/QBtoWEIS-CM/weis/wakelossfact/temp_{np.random.randint(1, 100000):6d}')
#     path_floris.mkdir(parents=True, exist_ok=True)

#     path_turbine=Path(path_floris.name+"/turbine")
#     path_turbine.mkdir(parents=True, exist_ok=True)

#     yaml_floris["farm"]["turbine_type"]="!include turbine_files/turbine_input_data.yaml"

#     yaml_turbine["power_thrust_table"]["power"]=Pout
#     yaml_turbine["power_thrust_table"]["thrust_coefficient"]=Ctout
#     yaml_turbine["power_thrust_table"]["wind_speed"]=Vout

#     append_to_yaml(path_floris.name+"emgauss_floating_model.yaml", yaml_floris)
#     append_to_yaml(path_turbine.name+"turbine_input_data.yaml", yaml_turbine)

#     return path_floris, path_turbine

class wakelossfactor(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('wt_init')
        self.options.declare('modeling_options')        
        # pass

    def setup(self):
        self.add_input('rotor_diameter', val=250, units='m')
        self.add_input('hub_height', val=90, units='m')
        self.add_discrete_input('turbine_number', val=40)
        # self.add_discrete_input('nturbine_per_row', val=7)
        self.add_input('row_spacing', val=7)
        self.add_input('turbine_spacing', val=7)

        self.add_output('wake_loss_factor', val=0.15)

        if self.options['modeling_options']['Level4']['flag'] or self.options['modeling_options']['Level3']['flag']:
            n_ws = self.options['modeling_options']['DLC_driver']['n_cases']
        else:
            n_ws = self.options['modeling_options']['WISDEM']['RotorSE']['n_pc']
        self.add_input('V_out',   val=np.zeros(n_ws),   units='m/s',    desc='wind speed vector from the OF simulations')
        self.add_input('P_out',   val=np.zeros(n_ws),   units='W',      desc='rotor electrical power')
        self.add_input('Cp_out',  val=np.zeros(n_ws),                   desc='rotor aero power coefficient')
        self.add_input('Ct_out',  val=np.zeros(n_ws),                   desc='rotor aero thrust coefficient')

        self.add_input('tilt_vals', val=np.zeros(self.options['modeling_options']['DLC_driver']['n_cases']), units='deg',      desc='tilt values for wake loss calculation')
        # self.add_input('vel_tilt', val=np.zeros(n_ws), units='m/s',      desc='velocity values for tilt correction in wake loss calculation')
        
        self.add_input('farm_alignment_angle', val=0.0)

        self.wind_data_file=self.options['modeling_options']['Floris']["floris_wind_data_file"]
        self.floris_input=self.options['modeling_options']['Floris']["floris_config_file"]
        self.nturbine_per_row=self.options['modeling_options']['Floris']['turbine_per_row']
        self.override_layout=self.options['modeling_options']['Floris']['override_layout']
        self.farm_alignment_angle=self.options['modeling_options']['Floris']['alignment_angle'] 

        self.add_output('Cp_out_post', val=np.zeros(n_ws),                   desc='rotor aero power coefficient')
        self.add_output('Ct_out_post', val=np.zeros(n_ws),                   desc='rotor aero thrust coefficient')
        self.add_output('tilt_vals_post', val=np.zeros(self.options['modeling_options']['DLC_driver']['n_cases']),                   desc='tilt values for floris pitch correction')

        mydata= pd.read_table(self.wind_data_file, skiprows=[1])
        time_series = TimeSeries(
            wind_speeds=mydata["Vel"].to_numpy(), 
            wind_directions=mydata["dir"].to_numpy(),
            turbulence_intensities=0.07*np.ones(len(mydata["dir"])))
        time_series.assign_ti_using_IEC_method()
        wind_rose = time_series.to_WindRose(wd_step=22.5, ws_step=2)
        k_fit, loc, A_fit = weibull_min.fit(time_series.wind_speeds, floc=0)
        self.weibull_Vmean=A_fit*gamma(1+1/k_fit)
        self.weibull_k=k_fit        
        
        self.add_output('site_weibull_Vmean',        val=self.weibull_Vmean,    desc='site weibull V mean')
        self.add_output('site_weibull_shape_factor', val=self.weibull_k,        desc='site weibull shape factor')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        diam=inputs['rotor_diameter'][0]
        hub_height=inputs['hub_height'][0]
        turb_spacing=inputs['turbine_spacing'][0]
        row_spacing=inputs['row_spacing'][0]

        V_out=inputs["V_out"]
        P_out=inputs["P_out"]
        Cp_out=inputs["Cp_out"]
        Ct_out=inputs["Ct_out"]

        # path_floris, path_turbine = write_turb_file(inputs['V_out'], inputs['P_out'], inputs['Ct_out'], self.floris_input, self.floris_input+"/turbine/turbine_input_data.yaml")
        
        nturbine=discrete_inputs['turbine_number']
        # wlf_calc=wlf(diam, turb_spacing, row_spacing, self.nturbine_per_row, nturbine, self.wind_data_file, self.floris_input, self.override_layout)
        # wlf_calc=wlf(diam, turb_spacing, row_spacing, self.nturbine_per_row, nturbine, self.wind_data_file, path_floris, self.override_layout)
        if self.options['modeling_options']['Level1']:
            tilts=inputs["tilt_vals"]
            vel_tilt=self.options['modeling_options']['DLC_driver']['DLCs']['DLC'=='1.1']['wind_speed']
            
            wlf_calc=wlf(diam, hub_height, turb_spacing, row_spacing, self.nturbine_per_row, nturbine, self.wind_data_file, \
                      self.floris_input, self.override_layout, V_out, P_out, Cp_out, Ct_out, 
                      correct_by_tilt=True, tilt_val=tilts, vel_tilt=vel_tilt, farm_alignment_angle=self.farm_alignment_angle)
        else:    
            wlf_calc=wlf(diam, hub_height, turb_spacing, row_spacing, self.nturbine_per_row, nturbine, self.wind_data_file, \
                      self.floris_input, self.override_layout, V_out, P_out, Cp_out, Ct_out, farm_alignment_angle=self.farm_alignment_angle)
        wlf_calc.run()

        outputs['wake_loss_factor']=wlf_calc.wlf

        outputs['Cp_out_post']=Cp_out
        outputs['Ct_out_post']=Ct_out
        outputs['tilt_vals_post']=tilts

        outputs['site_weibull_Vmean']=wlf_calc.weibull_Vmean
        outputs['site_weibull_shape_factor']=wlf_calc.weibull_k

if __name__=='__main__':
    # fname_wt_input='../../qb_examples/02_iea15mw/IEA-15-240-RWT_VolturnUS-S.yaml'
    # fname_modeling_options='../../qb_examples/02_iea15mw/analysis_options_noopt.yaml'
    # fname_opt_options='../../qb_examples/02_iea15mw/modeling_options_dlc11.yaml'
    fname_wt_input='qb_examples/02_iea15mw/IEA-15-240-RWT_VolturnUS-S.yaml'
    fname_modeling_options='qb_examples/02_iea15mw/analysis_options_opt.yaml'
    fname_opt_options='qb_examples/02_iea15mw/modeling_options.yaml'
    
    from wisdem.glue_code.glue_code import WindPark as wisdemPark
    from weis.glue_code.gc_LoadInputs import WindTurbineOntologyPythonWEIS

    wt_initial = WindTurbineOntologyPythonWEIS(
        fname_wt_input,
        fname_modeling_options,
        fname_opt_options
        )
    wt_init, modeling_options, opt_options = wt_initial.get_input_data()

    modeling_options["Floris"]["floris_config_file"]=os.path.abspath('/home/gt/qbweis-CM/weis/wakelossfact/emgauss_floating_model.yaml')
    modeling_options["Floris"]["floris_wind_data_file"]='weis/wakelossfact/wind_data.txt'
    modeling_options['WISDEM']['RotorSE']['n_pc']=12
    # modeling_options["floris_input"]='../floris/examples/mytests/emgauss_floating_IEA15MW.yaml'
    # modeling_options["wind_data_file"]='../floris/examples/mytests/wind_data_test.txt'

    prob=om.Problem()
    prob.model.add_subsystem('wlf',wakelossfactor(wt_init=wt_init, modeling_options=modeling_options))
    prob.setup()
    prob.set_val('wlf.V_out',[4.0, 6.0, 8.0, 10.0, 11.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0])
    prob.set_val('wlf.P_out',[0.0, 3014.2502497558594, 6790.816645955404, 12490.687025960286, 13740.1106484375, 14729.633271972656, 15002.555152018229, 14981.06745719401, 14942.60398046875, 15003.131510253907, 14636.33189860026, 15012.239022786458])
    prob.set_val('wlf.Ct_out',[0.12494466940437754, 0.8350929547250271, 0.7911350168486436, 0.6974972585787376, 0.5818606461460392, 0.44259051660696663, 0.26198846389601627, 0.1693310992817084, 0.12387880702813467, 0.08931361770443619, 0.06522135047925015, 0.05374533084438493])
    prob.set_val('wlf.Cp_out',[0.0, 0.5304193029552698, 0.5031360725313425, 0.46074565894901753, 0.4070303560147683, 0.3365564095402757, 0.2144692376703024, 0.14093897951891024, 0.10362952494869629, 0.07351792490544419, 0.05363205176467697, 0.04309534049841265])
    prob.run_model()

    print('Wake loss factor = ', prob.get_val('wlf.wake_loss_factor')[0]*100, ' %')

    print(f'   Weibull_Vmean = {prob.get_val('wlf.site_weibull_Vmean')[0]:.3f} m/s')
    print(f'   Weibull_k     = {prob.get_val('wlf.site_weibull_shape_factor')[0]:.3f}')
