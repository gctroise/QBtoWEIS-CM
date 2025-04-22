"""Wake loss factor calculation"""

from floris.flow_visualization import calculate_horizontal_plane_with_turbines, show, visualize_cut_plane
from floris.layout_visualization import plot_turbine_rotors, plot_turbine_labels

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from scipy.optimize import curve_fit
from scipy.special import gamma
from scipy.stats import weibull_min

from floris import (
    FlorisModel,
    TimeSeries,
    WindRose,
    turbine_library
)
from floris.utilities import wrap_360

import pandas as pd

from datetime import datetime

def weibull_func(U, A, k):
    return (k / A) * (U / A) ** (k - 1) * np.exp(-((U / A) ** k))


def estimate_weibull(wind_data_file):
    # Normalize the frequency
    mydata= pd.read_table(wind_data_file, skiprows=[1])
    time_series = TimeSeries(
        wind_speeds=mydata["Vel"].to_numpy(), 
        wind_directions=mydata["dir"].to_numpy(),
        turbulence_intensities=0.07*np.ones(len(mydata["dir"])))
    time_series.assign_ti_using_IEC_method()
    wind_rose = time_series.to_WindRose(wd_step=22.5, ws_step=2)
    # Fit the Weibull distribution
    k_fit, loc, A_fit = weibull_min.fit(time_series.wind_speeds, floc=0)
    weibull_Vmean=A_fit*gamma(1+1/k_fit)

    return A_fit, k_fit, weibull_Vmean

class calculationWLF:
    def __init__(self,diam, hub_height, turb_spacing, row_spacing, nturb_per_row, nturbine, wind_data_file, floris_input, override_layout, 
                 V_out, P_out, Cp_out, Ct_out, correct_by_tilt=False, tilt_val=None, vel_tilt=None, farm_alignment_angle=0.0):
        self.diam=diam
        self.hub_height=hub_height
        self.turb_spacing=turb_spacing
        self.row_spacing=row_spacing
        self.nturb_per_row=nturb_per_row
        self.nturbine=nturbine
        self.wind_data_file = wind_data_file
        self.floris_input = floris_input
        self.override_layout=override_layout
        self.V_out=V_out
        self.P_out=P_out
        self.Cp_out=Cp_out
        self.Ct_out=Ct_out

        self.wlf=0.0

        self.correct_by_tilt=correct_by_tilt
        self.tilt_val=tilt_val
        self.vel_tilt=vel_tilt
        self.farm_alignment_angle=farm_alignment_angle

    def run(self):
        #%% import wind data
        mydata= pd.read_table(self.wind_data_file, skiprows=[1])
        time_series = TimeSeries(
            wind_speeds=mydata["Vel"].to_numpy(), 
            wind_directions=mydata["dir"].to_numpy(),
            turbulence_intensities=0.07*np.ones(len(mydata["dir"])))
        time_series.assign_ti_using_IEC_method()
        wind_rose = time_series.to_WindRose(wd_step=22.5, ws_step=2)

        # A_fit, k_fit = estimate_weibull(wind_rose.wind_speeds, wind_rose.freq_table.sum(axis=0))
        k_fit, loc, A_fit = weibull_min.fit(time_series.wind_speeds, floc=0)
        self.weibull_Vmean=A_fit*gamma(1+1/k_fit)
        self.weibull_k=k_fit

        # # plot wind data
        # fig00= plt.figure()
        # date_format = "%d/%m/%Y %H:%M"
        # tt = [datetime.strptime(date, date_format) for date in mydata["date"].values]
        # plt.plot(tt,mydata["Vel"])
        # plt.ylabel("Vel (m/s)")
        # wind_rose.plot()
        
        # create turbine info dictionary
        turbine_data_dict={
            "wind_speed": list(self.V_out),
            "power_coefficient": list(self.Cp_out),
            "thrust_coefficient": list(self.Ct_out),
        }
        turbine_dict = turbine_library.build_cosine_loss_turbine_dict(
            turbine_data_dict,
            "example_turbine",
            file_name=None,
            generator_efficiency=1,
            hub_height=self.hub_height,
            cosine_loss_exponent_yaw=1.88,
            cosine_loss_exponent_tilt=1.88,
            rotor_diameter=self.diam,
            TSR=8,
            ref_air_density=1.225,
            ref_tilt=0,
        )

        
        # # create floris model 
        # fmodel = FlorisModel(self.floris_input)
        # fmodel.set(
        #     wind_data=wind_rose,
        #     turbine_type=[turbine_dict],
        #     reference_wind_height=self.hub_height
        # )

        #%% create floris model 
        fmodel = FlorisModel(self.floris_input)
        fmodel.set(
            wind_data=wind_rose,
            turbine_type=[turbine_dict],
            reference_wind_height=self.hub_height
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
            rotmat=np.array([[np.cos(np.deg2rad(self.farm_alignment_angle)), -np.sin(np.deg2rad(self.farm_alignment_angle))],
                             [np.sin(np.deg2rad(self.farm_alignment_angle)),  np.cos(np.deg2rad(self.farm_alignment_angle))]])
            for ii in range(len(xx_turb)):
                rr=rotmat.dot(np.array([[xx_turb[ii]], [yy_turb[ii]]]))
                xx_turb[ii]=rr[0][0]
                yy_turb[ii]=rr[1][0]

            fmodel.set(layout_x=xx_turb, layout_y=yy_turb)

        if self.correct_by_tilt:
            # fmodel.core.farm.correct_cp_ct_for_tilt = np.full((1,len(fmodel.layout_x)), True)
            # turbine_dict["correct_cp_ct_for_tilt"] = np.full((1,len(fmodel.layout_x)), True)
            turbine_dict["correct_cp_ct_for_tilt"]=True
            turbine_dict["floating_tilt_table"] = {}
            turbine_dict["floating_tilt_table"]["wind_speed"]=self.vel_tilt
            turbine_dict["floating_tilt_table"]["tilt"]=self.tilt_val
            
            fmodel.set(
                wind_data=wind_rose,
                turbine_type=[turbine_dict],
                reference_wind_height=self.hub_height
                )

        ax_layout=plot_turbine_rotors(fmodel)
        plot_turbine_labels(fmodel,ax_layout)

        #%% calculate energy loss 
        
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
        print(f"Farm overall wake loss is {(farm_aep-farm_aep_no_wake)/farm_aep_no_wake:.3f}")
        # fig1 =plt.subplot()
        # horizontal_plane = \
        #     calculate_horizontal_plane_with_turbines(fmodel, \
        #         x_resolution=200, y_resolution=100,findex_for_viz=3)
        # visualize_cut_plane(horizontal_plane, fig1)
        # plt.show()
        
        plt.figure()
        ax=plt.plot(np.linspace(0,time_series.wind_speeds.max()/60,len(time_series.wind_speeds)),time_series.wind_speeds)
        print(time_series.wind_speeds.max())
        print(time_series.wind_speeds.min())
        plt.ylim((0,20))


        plt.figure()
        ws_freq=wind_rose.freq_table.sum(axis=0)
        ws_freq=ws_freq/ws_freq.sum()
        ws_vel=wind_rose.wind_speeds
        ws_bin_width=np.diff(ws_vel).mean()
        # pdf_weib=weibull_func(ws_vel, self.weibull_Vmean*gamma(1+1/self.weibull_k), self.weibull_k)
        pdf_weib=weibull_min.pdf(ws_vel, self.weibull_k, 0.0, self.weibull_Vmean/gamma(1+1/self.weibull_k))
        plt.bar(ws_vel,ws_freq)
        plt.plot(ws_vel,pdf_weib*ws_bin_width)
        plt.show()
        print(f'wlf: weibull shape factor={self.weibull_k}    weibull Vmean={self.weibull_Vmean} m/s')