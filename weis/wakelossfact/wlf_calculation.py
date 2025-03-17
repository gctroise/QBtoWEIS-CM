"""Wake loss factor calculation"""

from floris.flow_visualization import calculate_horizontal_plane_with_turbines, show, visualize_cut_plane
from floris.layout_visualization import plot_turbine_rotors, plot_turbine_labels

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from floris import (
    FlorisModel,
    TimeSeries,
    WindRose,
)
from floris.utilities import wrap_360

import pandas as pd

from datetime import datetime

class calculationWLF:
    def __init__(self,diam, turb_spacing, row_spacing, nturb_per_row, nturbine, wind_data_file, floris_input, override_layout):
        self.diam=diam
        self.turb_spacing=turb_spacing
        self.row_spacing=row_spacing
        self.nturb_per_row=nturb_per_row
        self.nturbine=nturbine
        self.wind_data_file = wind_data_file
        self.floris_input = floris_input
        self.override_layout=override_layout

        self.wlf=0.0
        
    def run(self):
        #%% import wind data
        mydata= pd.read_table(self.wind_data_file, skiprows=[1])
        time_series = TimeSeries(
            wind_speeds=mydata["Vel"].to_numpy(), 
            wind_directions=mydata["dir"].to_numpy(),
            turbulence_intensities=0.07*np.ones(len(mydata["dir"])))
        time_series.assign_ti_using_IEC_method()
        wind_rose = time_series.to_WindRose(wd_step=22.5, ws_step=2)
        # # plot wind data
        # fig00= plt.figure()
        # date_format = "%d/%m/%Y %H:%M"
        # tt = [datetime.strptime(date, date_format) for date in mydata["date"].values]
        # plt.plot(tt,mydata["Vel"])
        # plt.ylabel("Vel (m/s)")
        # wind_rose.plot()
        
        # create floris model 
        fmodel = FlorisModel(self.floris_input)
        fmodel.set(
            wind_data=wind_rose
        )
        #%% set farm layout
        # turbine_number=10
        # nturb_per_row=7
        # row_spacing=7
        # turb_spacing=7
        # nrows=int(np.flor(turbine_number/nturb_per_row))
        # diam=fmodel.core.farm.rotor_diameters[0]

        if self.override_layout:
            xx_turb=[self.turb_spacing*self.diam*(i%self.nturb_per_row) for i in range(self.nturbine)]
            yy_turb=[self.row_spacing*self.diam*(i//(self.nturb_per_row)) for i in range(self.nturbine)]
            fmodel.set(layout_x=xx_turb, layout_y=yy_turb)

        ax_layout=plot_turbine_rotors(fmodel)
        plot_turbine_labels(fmodel,ax_layout)

        #%% calculate enery loss 
        
        # run the model without wake losses
        fmodel.run_no_wake()
        farm_power_no_wake = fmodel.get_farm_power()
        farm_aep_no_wake = fmodel.get_farm_AEP()

        # Run the model and collect the outputs with wake losses
        fmodel.run()
        # Get the power outputs
        turbine_powers = fmodel.get_turbine_powers()
        farm_power = fmodel.get_farm_power()
        expected_farm_power = fmodel.get_expected_farm_power()
        farm_aep = fmodel.get_farm_AEP()
        self.wlf=np.abs((farm_aep-farm_aep_no_wake)/farm_aep_no_wake)

        #%% Display results
        np.set_printoptions(linewidth=200)
        print(f"Turbine power have shape {turbine_powers.shape} and are \n{turbine_powers}")
        print(f"Farm power has shape {farm_power.shape} and is {farm_power}")
        print(f"Expected farm power has shape {expected_farm_power.shape} and is {expected_farm_power}")
        print(f"Farm AEP is {farm_aep/1e9:.2f} GWh with wake losses")
        print(f"Farm AEP is {farm_aep_no_wake/1e9:.2f} GWh without wake losses")
        print(f"Farm overall wake loss is {(farm_aep-farm_aep_no_wake)/farm_aep_no_wake:.3f}%")
        # fig1 =plt.subplot()
        # horizontal_plane = \
        #     calculate_horizontal_plane_with_turbines(fmodel, \
        #         x_resolution=200, y_resolution=100,findex_for_viz=3)
        # visualize_cut_plane(horizontal_plane, fig1)
        # plt.show()