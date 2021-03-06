# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 09:12:21 2021

@author: johan
"""


import os, gzip   

from scipy.interpolate import interp1d, interp2d
from scipy.optimize import fsolve, fmin
from scipy import signal
import pandas as pd
import numpy as np
import gsw

from tqdm import tqdm



#Usefull functions
def date2float(d, epoch=pd.to_datetime(0, utc=True, origin='unix', cache='False')):
    return (d - epoch).dt.total_seconds()

        
def fillGaps(x,y):
    f = interp1d(x[np.isfinite(x+y)],y[np.isfinite(x+y)], bounds_error=False, fill_value=np.NaN)
    return(f(x))


#New Steady State model (With adcp constraint)
class SteadyState_2_Model(object):
    def __init__(self,time,sal, temp_ext, temp_int, pres_ext, pres_int, lon,lat,ballast,pitch,profile,navresource,tau,adcp_speed,**param):
        
        self.timestamp = time
        self.time = date2float(self.timestamp)
        
        self.external_pressure = pres_ext
        self.internal_pressure = pres_int
        self.external_temperature = temp_ext
        self.internal_temperature = temp_int
        
        self.longitude = lon
        self.latitude = lat
        self.profile = profile
        
        self.salinity = sal
        
        self.ballast = ballast/1000000 # m^3
        self.pitch = np.deg2rad(pitch) # rad
        
        self.navresource=navresource
        
        self.tau=tau
        self.adcp_speed=adcp_speed
        
        
  
        """
            'mass': 60.772, # Vehicle mass in kg
            'vol0': 0.05945962561711446, # Reference volume in m**3, with ballast at 0 (with -500 to 500 range), at surface pressure and 20 degrees C
            'area_w': 0.24, # Wing surface area, m**2
            
            'Cd_0': 0.0465087374050051, #
            'Cd_1': 1.27273918009592, # 
            
            'Cd': 0.05, #Cd is nly used to determine the Cd(alpha) curve
            
            'Cl': 1.4827080303263165, # Negative because wrong convention on theta
            
            'alpha_stall': 10, #
            'alpha_linear': 10, #
              
            'comp_p': 5.212342277484937e-06, # Pressure dependent hull compression factor
            'comp_t': 6.563961197364007e-05, # Temperature dependent hull compression factor
            
            'SSStau': 18.5 #characteristic response time of the glider in sec
        """
        
        
        #For the first set of data :
        self.param_reference = dict({
            'mass': 60.772, # Vehicle mass in kg
            'vol0': 0.05944919244909694, # Reference volume in m**3, with ballast at 0 (with -500 to 500 range), at surface pressure and 20 degrees C
            'area_w': 0.24, # Wing surface area, m**2
            
            'Cd_0': 0.046601172003459146, #
            'Cd_1': 2.2846652475518, # 
            
            'Cd': 0.05, #Cd is nly used to determine the Cd(alpha) curve
            
            'Cl': 1.9551684905403062, # Negative because wrong convention on theta
            
            'alpha_stall': 10, #
            'alpha_linear': 10, #
              
            'comp_p': 4.702895408506633e-06, #1.023279317627415e-06, #Pressure dependent hull compression factor
            'comp_t': 8.898950889070907e-05, #1.5665248101730484e-04, # Temperature dependent hull compression factor
            
            'SSStau': 18.5 #characteristic response time of the glider in sec
        })
        
        
        
        """
        
        #For the second set of data :
        self.param_reference = dict({
            'mass': 60.772, # Vehicle mass in kg
            'vol0': 0.059516188313651484, # Reference volume in m**3, with ballast at 0 (with -500 to 500 range), at surface pressure and 20 degrees C
            'area_w': 0.24, # Wing surface area, m**2
            
            'Cd_0': 0.07386659009159255, #
            'Cd_1': 0.34783408917512537, # 
            
            'Cd': 0.05, #Cd is only used to determine the Cd(alpha) curve
            
            'Cl': 1.8805894442146223, 
            
            'alpha_stall': 10, #
            'alpha_linear': 10, #
              
            'comp_p': 4.052789561765741e-06, # Pressure dependent hull compression factor
            'comp_t': 4.753268947701921e-05, # Temperature dependent hull compression factor
            
            'SSStau': 18.5 #characteristic response time of the glider in sec
        })
        """
        
        

        self.param = self.param_reference.copy()
        for k,v in param.items():
            self.param[k] = v
        self.param_initial = self.param
                
        def RM(x,N):
            big = np.full([N,len(x)+N-1],np.nan)
            for n in np.arange(N):
                if n == N-1:
                    big[n, n : ] = x
                else:
                    big[n, n : -N+n+1] = x
            return np.nanmedian(big[:,int(np.floor(N/2)):-int(np.floor(N/2))],axis=0)

        def smooth(x,N):
            return np.convolve(x, np.ones(N)/N, mode='same')
                
        #self.pressure = fillGaps(self.time, self.raw_pressure)
        #self.temperature = fillGaps(self.time, self.raw_temperature)
        #self.salinity = fillGaps(self.time, self.raw_salinity)

        #self.pressure = smooth(RM(self.pressure,5),20)
        #self.temperature = smooth(RM(self.temperature,5),20)
        #self.salinity = smooth(RM(self.salinity,5),20)
        
        self.depth = gsw.z_from_p(self.external_pressure,self.latitude) # m . Note depth (Z) is negative, so diving is negative dZdt
        self.dZdt = np.gradient(self.depth,self.time) # m.s-1
        self.dZdt = smooth(self.dZdt, 40) #Smooth noisy value --------------------------------------- !!!!!!!------!!!!!!
        

        self.g = gsw.grav(self.latitude,self.external_pressure)
        
        self.SA = gsw.SA_from_SP(self.salinity, self.external_pressure, self.longitude, self.latitude)
        self.CT = gsw.CT_from_t(self.SA, self.external_temperature, self.external_pressure)
        self.rho = gsw.rho(self.SA, self.CT, self.external_pressure)

        ### Basic model
        # Relies on steady state assumption that buoyancy, weight, drag and lift cancel out when not accelerating.
        # F_B - cos(glide_angle)*F_L - sin(glide_angle)*F_D - F_g = 0 
        # cos(glide_angle)*F_L + sin(glide_angle)*F_D = 0

        # Begin with an initial computation of angle of attack and speed through water:
        self.model_function()
        self.Apply_lowpass_filter()
        
        ### Get good datapoints to regress over
        self._valid = np.full(np.shape(self.pitch),True)
        self._valid[self.external_pressure < 5] = False
        self._valid[np.abs(self.pitch) < 0.2] = False   # TODO change back to 15
        self._valid[np.abs(self.pitch) > 0.6] = False   # TODO change back to 15
        self._valid[np.abs(np.gradient(self.dZdt,self.time)) > 0.0005] = False # Accelerations
        self._valid[np.gradient(self.pitch,self.time)==0] = False # Rotation
        self._valid = self._valid & ((navresource == 100) | (navresource == 117) ) #100=glider going down & 117=glider going up (=> not at surface or inflecting)
        
        #self._valid[np.abs(self.pitch) < 0.55] = False   
        #self._valid[np.abs(self.pitch) > 0.57] = False   
        #self._valid[self.vert_dir*np.sign(self.pitch)< 0] = False   #Stall
        
        
        # Do first pass regression on vol parameters, or volume and hydro?
        #self.regression_parameters = ('vol0','Cd_0','Cd_1','Cl','comp_p','comp_t') 
        self.regression_parameters = ('vol0',) 
        #self.regression_parameters = ('SSStau',) 
        #('vol0','comp_p','comp_t','Cd_0','Cd_1','Cl_h','Cl_w') # Has to be a tuple, requires a trailing comma if single element




    ### Principal forces
    @property
    def F_B(self):
        return self.g * self.rho * (self.ballast + self.vol0 * (1 - self.comp_p*(self.external_pressure-0.8) + self.comp_t*(self.external_temperature-10)))
        #return self.g * self.rho * (self.ballast + self.vol0 * (1 - self.comp_p*(self.external_pressure) + self.comp_t*(self.internal_temperature-10)))

    @property
    def F_g(self):
        return self.mass * self.g
    
    @property
    def vert_dir(self):
        return np.sign(self.F_B-self.F_g) #Positive is buoyancy force up & negative is buoyancy force down

    @property
    def F_L(self):
        return self.dynamic_pressure * (self.Cl) * self.alpha

    @property
    def F_D(self):
        return self.dynamic_pressure * (   self.Cd_0 + self.Cd_1*(self.alpha)**2   )
        #return self.dynamic_pressure * (   self.Cd   )

    @property
    def Pa(self):
        return self.external_pressure * 10000 # Pa

    ### Important variables
    @property
    def glide_angle(self):
        return self.pitch + self.alpha
    
    @property
    def dynamic_pressure(self):
        return self.rho * self.area_w * self.speed**2 / 2

    @property
    def w_H2O(self):
        # Water column upwelling
        return self.dZdt - self.speed_vert
    
    @property
    def U_H2O(self):
        return self.adcp_speed - self.speed   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    @property
    def R1(self):
        return np.sqrt(np.nanmean(   (1-self.tau)*self.w_H2O[self._valid]**2   +   self.tau*self.U_H2O[self._valid]**2   ))
        # np.sqrt(np.nanmean(   (1-self.tau)*((self.w_H2O[219631:279060])[self._valid[219631:279060]])**2   +   ((self.tau*self.U_H2O[219631:279060])[self._valid[219631:279060]])**2   ))
    

    ### Basic equations
    """
    def _solve_alpha(self):
        _pitch_range = np.linspace( np.deg2rad(0), np.deg2rad(90) , 100)
        _alpha_range = np.zeros_like(_pitch_range)
        
        #R??solution vol normal
        for _istep, _pitch in enumerate(_pitch_range):
            _tmp = fsolve(self._equation_alpha, 0.001, args=(_pitch), full_output=True)
            _alpha_range[_istep] = _tmp[0]
        
        #R??solution 
        for _istep, _pitch in enumerate(_pitch_range):
            _tmp = fsolve(self._equation_alpha, 0.001, args=(_pitch), full_output=True)
            _alpha_range[_istep] = _tmp[0]
            
        _interp_fn = interp1d(_pitch_range,_alpha_range)
        return _interp_fn(np.abs(self.pitch)) * np.sign(self.pitch)    
    """
    
    
    def _stall_factor(self, _alpha):
        d_alpha = np.abs(_alpha) - self.alpha_linear
        tau_alpha = self.alpha_stall - self.alpha_linear
        res=1.0
        if (np.abs(_alpha)>self.alpha_linear) :
            res=np.exp(-d_alpha/tau_alpha)
        return res
    
    def stall_factor(self, alpha):
        d_alpha = np.abs(alpha) - self.alpha_linear
        tau_alpha = self.alpha_stall - self.alpha_linear
        res=alpha
        res[np.abs(alpha)<self.alpha_linear]=1.0
        res[np.abs(alpha)>self.alpha_linear]=np.exp(-d_alpha[np.abs(alpha)>self.alpha_linear]/tau_alpha)
        return res
    
    
    def _solve_alpha(self):
        _pitch_range = np.linspace( np.deg2rad(0), np.deg2rad(90) , 100)
        _alpha_range1 = np.zeros_like(_pitch_range)
        _alpha_range2 = np.zeros_like(_pitch_range)      
        #R??solution vol normal
        for _istep, _pitch in enumerate(_pitch_range):
            _tmp = fsolve(self._equation_alpha, 0.001, args=(_pitch), full_output=True)
            _alpha_range1[_istep] = _tmp[0]
        
        #R??solution d??crochage
        for _istep, _pitch in enumerate(_pitch_range):
            if (np.sign(_pitch)>0) :
                _tmp = fsolve(self._equation_alpha, (-np.pi/2 -_pitch + 0.0001), args=(_pitch), full_output=True)
                _alpha_range2[_istep] = _tmp[0]
            else :
                _tmp = fsolve(self._equation_alpha, (np.pi/2 -_pitch - 0.0001), args=(_pitch), full_output=True)
                _alpha_range2[_istep] = _tmp[0]
            
        _interp_fn1 = interp1d(_pitch_range,_alpha_range1)
        _interp_fn2 = interp1d(_pitch_range,_alpha_range2)
        
        Res=_interp_fn1(np.abs(self.pitch)) * np.sign(self.pitch) #R??solution noramle
        Res[self.vert_dir*np.sign(self.pitch)<0]=(_interp_fn2(np.abs(self.pitch)) * np.sign(self.pitch))[self.vert_dir*np.sign(self.pitch)<0] #R??solution d??crochage
        return Res
    

    def _equation_alpha(self, _alpha, _pitch):
        return (   self.Cd_0 + self.Cd_1 *(_alpha)**2   ) / (   (self.Cl) * np.tan(_alpha + _pitch)   ) - _alpha
        #return (   self.Cd   ) / (   (self.Cl) * np.tan(_alpha + _pitch)   ) - _alpha

    def _solve_speed(self):
        _dynamic_pressure = (self.F_B - self.F_g) * np.sin(self.glide_angle) / (   self.Cd_0 + self.Cd_1 * (self.alpha)**2   )
        #_dynamic_pressure = (self.F_B - self.F_g) * np.sin(self.glide_angle) / (   self.Cd   )
        return np.sqrt(2 * _dynamic_pressure / self.rho / self.area_w)

    def model_function(self):
        self.alpha = self._solve_alpha()
        self.speed = self._solve_speed()
        self.speed [ (self.external_pressure < 5) & ((self.navresource == 115)|(self.navresource == 116)) ] = 0 #Set speed to be zero at the surface
        self.speed_vert = np.sin(self.glide_angle)*self.speed
        self.speed_horz = np.cos(self.glide_angle)*self.speed 
        #self.Apply_lowpass_filter()#------------------------------------------------------!!!!!!!!!!!!!!!!!!!!

    def cost_function(self,x_initial):        
        for _istep, _key in enumerate(self.regression_parameters):
            self.param[_key] = x_initial[_istep] * self.param_reference[_key]
        self.model_function()
        return self.R1 
    
    def regress(self):
        x_initial = [self.param[_key] / self.param_reference[_key] for _istep,_key in enumerate(self.regression_parameters)]
        print('Initial parameters: ', self.param)
        print('Non-optimised score: '+str(self.cost_function(x_initial)) )
        print('Regressing...')

        R = fmin(self.cost_function, x_initial, disp=True, full_output=True, maxiter=500)
        for _istep,_key in enumerate(self.regression_parameters):
            self.param[_key] = R[0][_istep] * self.param_reference[_key]
        
        x_initial = [self.param[_key] / self.param_reference[_key] for _istep,_key in enumerate(self.regression_parameters)]
        print('Optimised parameters: ', self.param)
        print('Final Optimised score: '+str(self.cost_function(x_initial)) )
        self.model_function()
        
    def Apply_lowpass_filter(self):
        speed=fillGaps(self.time,self.speed)
        speed=np.nan_to_num(speed, copy=True, nan=0.0, posinf=None, neginf=None)
        alpha=fillGaps(self.time,self.alpha)
        alpha=np.nan_to_num(alpha, copy=True, nan=0.0, posinf=None, neginf=None)
        sos = signal.butter(1, 1/(2*np.pi*self.SSStau), 'lowpass', fs=1, output='sos') #order, cutoff frequency, "lowpass", sampling frequency,
        self.speed_filtered = signal.sosfilt(sos, speed )
        self.alpha_filtered=signal.sosfilt(sos, alpha )
        glide_angle=self.alpha_filtered+self.pitch
        self.speed_vert_filtered = np.sin(glide_angle)*self.speed_filtered
        self.speed_horz_filtered = np.cos(glide_angle)*self.speed_filtered
        
        
        

    ### Coefficients
    @property        
    def mass(self):
        return self.param['mass']
    
    @property        
    def vol0(self):
        return self.param['vol0']
    
    @property        
    def comp_p(self):
        return self.param['comp_p']
    
    @property        
    def comp_t(self):
        return self.param['comp_t']
    
    @property        
    def area_w(self):
        return self.param['area_w']
    
    @property        
    def Cd_0(self):
        return self.param['Cd_0']
    
    @property        
    def Cd_1(self):
        return self.param['Cd_1']
    
    @property        
    def Cd(self):
        return self.param['Cd']
    
    @property        
    def Cl(self):
        return self.param['Cl']
    
    @property        
    def alpha_stall(self):
        return self.param['alpha_stall']
    
    @property        
    def alpha_linear(self):
        return self.param['alpha_linear']
      
    @property        
    def SSStau(self):
        return self.param['SSStau']