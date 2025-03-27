import numpy as np
import os, sys, time, json
import openmdao.api as om

# import wisdem.commonse.utilities as util

class myopex(om.ExplicitComponent):

    def initialize(self):
        pass
        self.options.declare('wt_init')
        # self.options.declare('modeling_options')
    
    def setup(self):
        wt_init = self.options['wt_init']

        self.add_input('rated_power', val=15000000, units="W")
        self.add_input('rotor_diameter', val=250.0, units='m')
        self.add_discrete_input('turbine_number', val=1)
        # self.add_input('plant_aep', val=1e9, units='kW*h')
        self.add_input('turbine_aep', val=50e6, units='kW*h')
        self.add_input('wake_loss_factor', val=0.15)
        # self.add_input('capacity_factor', val=0.4)
        
        self.add_input('distance_to_shore', val=100, units='km')
        self.add_input('plant_turbine_spacing', val=7)
        self.add_input('plant_row_spacing', val=7)
        
        self.add_output('capacity_factor_opex', val=0.5)
        self.add_output('power_density', val=5.0, units='MW/km**2')
        self.add_output('opex_cost', val=70.0, units='MUSD/year')
        self.add_output('opex_cost_kW', val=wt_init["costs"]["opex_per_kW"], units='USD/kW/year')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # wt_init = self.options['wt_init']
        # mod_opt = self.options['modeling_options']
        
        # self.set_val('myopex.rated_power', wt_init["assembly"]["rated_power"]/1000000) #in MW
        # self.set_val('myopex.turbine_number', wt_init["costs"]["turbine_number"]) #in km
        # self.set_val('myopex.capacity_factor', 0.4)
        # self.set_val('myopex.distance_to_shore', wt_init["bos"]["distance_to_site"])
        # power_dens=wt_init["assembly"]["rated_power"]/1e6/(wt_init["bos"]["plant_turbine_spacing"]*wt_init["bos"]["plant_row_spacing"]*wt_init["assembly"]["rotor_diameter"]**2/1e6)
        # self.set_val('myopex.power_density', power_dens)
        
        wlf=inputs["wake_loss_factor"]
        cf=inputs['turbine_aep']*discrete_inputs['turbine_number']*(1-wlf)/ \
                (inputs['rated_power']/1e3*discrete_inputs['turbine_number']*8760)
        # cf=inputs['capacity_factor']
        outputs['capacity_factor_opex']=cf
        pd=inputs['rated_power']/1e6 \
            /(inputs['plant_turbine_spacing']*inputs['plant_row_spacing']*inputs['rotor_diameter']**2/1e6) # in MW/km^2
        outputs['power_density']=pd
        
        opex=(inputs['rated_power']/1e6*discrete_inputs['turbine_number']) \
                                *7.224e-2*cf**0.84 \
                                *inputs['distance_to_shore']**0.19 \
                                *pd**0.22
        outputs['opex_cost'] = opex
        outputs['opex_cost_kW'] = opex*1e6 \
                /(inputs['rated_power']/1e3*discrete_inputs['turbine_number'])

if __name__=='__main__':
    fname_wt_input='../IEA-15-240-RWT_VolturnUS-S.yaml'
    fname_modeling_options='../analysis_options_opt.yaml'
    fname_opt_options='../modeling_options.yaml'
    
    from wisdem.glue_code.glue_code import WindPark as wisdemPark
    from weis.glue_code.gc_LoadInputs import WindTurbineOntologyPythonWEIS

    wt_initial = WindTurbineOntologyPythonWEIS(
        fname_wt_input,
        fname_modeling_options,
        fname_opt_options
        )
    wt_init, modeling_options, opt_options = wt_initial.get_input_data()

    prob=om.Problem()
    prob.model.add_subsystem('myopex',myopex(wt_init=wt_init, modeling_options=modeling_options))
    prob.setup()

    prob.run_model()

    print('Opex_cost = ', prob.get_val('myopex.opex_cost'), 'M€/year')
    print('Opex_cost_kw = ', prob.get_val('myopex.opex_cost_kW'), '€/kW/year')