"""Wake loss factor API"""

from .wlf_calculation import calculationWLF as wlf

import openmdao.api as om

class wakelossfactor(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('wt_init')
        self.options.declare('modeling_options')        
        # pass

    def setup(self):
        self.add_input('rotor_diameter', val=250, units='m')
        self.add_discrete_input('turbine_number', val=40)
        # self.add_discrete_input('nturbine_per_row', val=7)
        self.add_input('row_spacing', val=7)
        self.add_input('turbine_spacing', val=7)

        self.add_output('wake_loss_factor', val=0.15)

        self.wind_data_file=self.options['modeling_options']['Floris']["floris_wind_data_file"]
        self.floris_input=self.options['modeling_options']['Floris']["floris_config_file"]
        self.nturbine_per_row=self.options['modeling_options']['Floris']['turbine_per_row']
        self.override_layout=self.options['modeling_options']['Floris']['override_layout']

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        diam=inputs['rotor_diameter'][0]
        turb_spacing=inputs['turbine_spacing'][0]
        row_spacing=inputs['row_spacing'][0]

        nturbine=discrete_inputs['turbine_number']
        wlf_calc=wlf(diam, turb_spacing, row_spacing, self.nturbine_per_row, nturbine, self.wind_data_file, self.floris_input, self.override_layout)
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
