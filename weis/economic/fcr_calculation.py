""" Fixed charge rate calculation """

import numpy as np
import os, sys, time, json
import openmdao.api as om

class fcr(om.ExplicitComponent):
    """
    Compute Fixed Charge Rate for the wind plant, formulas from: -

    Parameters
    ----------
    debt_fraction: float
        The debt ratio can be calculated by dividing total debt by total assets. A debt ratio of less than 1.0 indicates that a company has more assets than debt.
    depreciation_period: int
        The number of years over which an asset's value is gradually decreased due to wear and tear, obsolescence, or other factors, for accounting and tax purposes.
    depreciation_fraction: float
        The portion of an asset's total cost that is allocated as an expense for a specific period, typically a year.
    equity_risk_premium: float
        The extra return investors expect from investing in stocks compared to a risk-free investment.
    risk_free_rate: float
        The interest an investor would expect from an absolutely risk-free investment over a specified period of time.
    equity_beta: float
        The volatility of returns for a stock, taking into account the impact of the company's leverage from its capital structure.
    project_duration: int
        The total amount of time it takes to finish a project
    tax_rate: float
        The ratio at which a business or person is taxed
    credit_spread: float
        The difference in yield between two debt instruments of the same maturity but differing credit ratings. That's based on Italian bond spread.
    premium_spread: float
        A strategy involving both buying and selling options with different strike prices, resulting in a net premium payment from the trader.
    innovation_premium: float
        The added value a company receives from being recognized as an innovative leader, exceeding the value of its current business.
    
    Returns
    -------
    wacc: float
        The Weighted average cost of capital (WACC) is a company's average after-tax cost of capital.
    capital_recovery_factor: float
        The Capital Recovery factor is the ratio of a constant annuity to the present value of receiving that annuity for a given length of time.
    fixed_charge_rate: float
        The Fixed Charge Rate is the amount of revenue per dollar of investment required that must be collected annually from customers to pay the carrying charges on that investment.
    """
    def initialize(self):
        self.options.declare('wt_init')
        self.options.declare('modeling_options')
    
    def setup(self):
        wt_init = self.options['wt_init']

        self.add_input('debt_fraction', val=0.7, desc='Debt fraction (DF)')                                                         # DF
        self.add_discrete_input('depreciation_period', val=25, desc='Depreciation period (M)')                                      # M
        self.add_input('depreciation_fraction', val=0.02, desc='Depreciation fraction (df)')                                        # df
        self.add_input('equity_risk_premium', val=0.0619, desc='Equity risk premium (ERP)- Average values in Europe (InnoFund)')    # ERP
        self.add_input('risk_free_rate', val=0.0226, desc='Risk free rate (RFR)- Average values in Europe (InnoFund)')              # RFR
        self.add_input('equity_beta', val=0.83, desc='Coefficient for ROE calculation - value for Renewable Energy (InnoFund)')
        self.add_input('project_duration', val=25.0, desc='Project duration')
        self.add_input('tax_rate', val=0.278, desc='Corporate tax rate (Tr)')                                                       # Tr
        self.add_input('credit_spread', val=0.04, desc='Credit spread based on Italian bond spread')
        self.add_input('premium_spread', val=0.03, desc='Premium spread value defined by InnoFund')
        self.add_input('innovation_premium', val=0.03, desc='Innovation premium - reference value from InnoFund')
        
        self.debt_fraction=self.options['modeling_options']['FCR']['debt_fraction']
        self.depreciation_period=self.options['modeling_options']['FCR']['depreciation_period']
        self.depreciation_fraction=self.options['modeling_options']['FCR']['depreciation_fraction']
        self.equity_risk_premium=self.options['modeling_options']['FCR']['equity_risk_premium']
        self.risk_free_rate=self.options['modeling_options']['FCR']['risk_free_rate']
        self.equity_beta=self.options['modeling_options']['FCR']['equity_beta']
        # self.project_duration=self.options['modeling_options']['FCR']['project_duration']
        self.tax_rate=self.options['modeling_options']['FCR']['tax_rate']
        self.credit_spread=self.options['modeling_options']['FCR']['credit_spread']
        self.premium_spread=self.options['modeling_options']['FCR']['premium_spread']
        self.innovation_premium=self.options['modeling_options']['FCR']['innovation_premium']

        self.add_output('wacc', val=0.07, desc='Weighted average capital cost')
        self.add_output('capital_recovery_factor', val=0.0921, desc='Capital Recovery Factor')
        self.add_output('fixed_charge_rate', val=wt_init["costs"]["fixed_charge_rate"], desc='Fixed charge rate')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # DF = inputs['debt_fraction'][0]
        # df = discrete_inputs['depreciation_fraction'][0]
        # rfr = inputs['risk_free_rate'][0]
        # erp = inputs['equity_risk_premium'][0]
        # equity_b = inputs['equity_beta'][0]
        # cs = inputs['credit_spread'][0]
        # ps = inputs['premium_spread'][0]
        # ip = inputs['innovation_premium'][0]
        # M = inputs['depreciation_period'][0]
        t = inputs['project_duration'][0]
        # Tr = inputs['tax_rate'][0]

        DF = self.debt_fraction
        df = self.depreciation_fraction
        rfr = self.risk_free_rate
        erp = self.equity_risk_premium
        equity_b = self.equity_beta
        cs = self.credit_spread
        ps = self.premium_spread
        ip = self.innovation_premium
        M = self.depreciation_period
        # t = self.project_duration
        Tr = self.tax_rate
        
        
        EF = 1 - DF # Equity fraction
        
        # rfr=np.where(country='Italy')
        # erp=np.where(country='Italy')
        # equity_b=np.where(type_industry='Green & Renewable Energy')
        
        roe = rfr + equity_b * erp + ip                 # Return on equity or cost of equity
        crod = rfr + cs + ps                            # cost of debt
        
        # Tr=np.where(country='Italy')
        
        WACC = EF * roe + DF * crod * ( 1 - Tr )        # Weighted average capital cost
        outputs['wacc'] = WACC
        
        crf = WACC / ( 1 - 1 / ( 1 + WACC )**t )        # Capital recovery factor
        outputs['capital_recovery_factor'] = crf

        d = WACC                                        # Discount rate of depretiation
        
        PVD = df * d / ( 1 - 1 / ( 1 + d )**M )         # Present value of depretiation
        
        Profinfact = ( 1 - Tr * PVD ) / ( 1 - Tr )
        
        fcr = crf * Profinfact                          # Fixed charge rate
        outputs['fixed_charge_rate'] = fcr

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
    prob.model.add_subsystem('fcr',fcr(wt_init = wt_init, modeling_options = modeling_options))
    prob.setup()

    prob.run_model()

    print('WACC = ', prob.get_val('fcr.wacc'))
    print('Capital recovery factor = ', prob.get_val('fcr.capital_recovery_factor'))
    print('Fixed charge rate = ', prob.get_val('fcr.fixed_charge_rate'))
    