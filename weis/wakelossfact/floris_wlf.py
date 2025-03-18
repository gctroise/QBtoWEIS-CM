"""Wake loss factor API"""

from .wlf_calculation import calculationWLF as wlf

import openmdao.api as om

import numpy as np
import ruamel.yaml as ry
from pathlib import Path
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

        n_ws = self.options['modeling_options']['DLC_driver']['n_cases']
        self.add_input('V_out',   val=np.zeros(n_ws),   units='m/s',    desc='wind speed vector from the OF simulations')
        self.add_input('P_out',   val=np.zeros(n_ws),   units='W',      desc='rotor electrical power')
        self.add_input('Cp_out',  val=np.zeros(n_ws),                   desc='rotor aero power coefficient')
        self.add_input('Ct_out',  val=np.zeros(n_ws),                   desc='rotor aero thrust coefficient')

        self.wind_data_file=self.options['modeling_options']['Floris']["floris_wind_data_file"]
        self.floris_input=self.options['modeling_options']['Floris']["floris_config_file"]
        self.nturbine_per_row=self.options['modeling_options']['Floris']['turbine_per_row']
        self.override_layout=self.options['modeling_options']['Floris']['override_layout']

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
        wlf_calc=wlf(diam, hub_height, turb_spacing, row_spacing, self.nturbine_per_row, nturbine, self.wind_data_file, \
                      self.floris_input, self.override_layout, V_out, P_out, Cp_out, Ct_out)
        wlf_calc.run()

        outputs['wake_loss_factor']=wlf_calc.wlf

if __name__=='__main__':
    fname_wt_input='../../qb_examples/02_iea15mw/IEA-15-240-RWT_VolturnUS-S.yaml'
    fname_modeling_options='../../qb_examples/02_iea15mw/analysis_options_opt.yaml'
    fname_opt_options='../../qb_examples/02_iea15mw/modeling_options.yaml'
    # fname_wt_input='qb_examples/02_iea15mw/IEA-15-240-RWT_VolturnUS-S.yaml'
    # fname_modeling_options='qb_examples/02_iea15mw/analysis_options_opt.yaml'
    # fname_opt_options='qb_examples/02_iea15mw/modeling_options.yaml'
    
    from wisdem.glue_code.glue_code import WindPark as wisdemPark
    from weis.glue_code.gc_LoadInputs import WindTurbineOntologyPythonWEIS

    wt_initial = WindTurbineOntologyPythonWEIS(
        fname_wt_input,
        fname_modeling_options,
        fname_opt_options
        )
    wt_init, modeling_options, opt_options = wt_initial.get_input_data()

    modeling_options["floris_input"]='../../../floris/examples/mytests/emgauss_floating_IEA15MW.yaml'
    modeling_options["wind_data_file"]='../../../floris/examples/mytests/wind_data_test.txt'
    # modeling_options["floris_input"]='../floris/examples/mytests/emgauss_floating_IEA15MW.yaml'
    # modeling_options["wind_data_file"]='../floris/examples/mytests/wind_data_test.txt'

    prob=om.Problem()
    prob.model.add_subsystem('wlf',wakelossfactor(wt_init=wt_init, modeling_options=modeling_options))
    prob.setup()

    prob.run_model()

    print('Wake loss factor = ', prob.get_val('wlf.wake_loss_factor')[0]*100, ' %')
