# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:18:05 2021

@author: johan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, fmin
import gsw
from tqdm import tqdm


#Define general model   
#drone_params = {"mg" : 60, "m11" : 10, "m22"}
#u=np.zeros_like

"""
class SeaExplorer(object):
    def __init__(self, drone_params):
        self.mg = 60
        self.AR = 7
        self.eOsborne = 0.8
        self.Cd1_hull = 2.1
        self.Omega = 0.75
        
    def compute_FB_and_Fg(self, pressure, rho, Vbp, mg=None, Vg=None):
        
    def compute_lift_and_drag(self,alpha, U, rho, Cd0=None):

    def compute_inverted_mass_matrix():
"""

def date2float(d, epoch=pd.to_datetime(0, utc=True, origin='unix', cache='False')):
    return (d - epoch).dt.total_seconds()   
        
class DynamicFlightModel(object):
    def __init__(self,time,sal,internal_temp,external_temp,pres,lon,lat,ballast,pitch, profile, navresource, adcp_speed, **param):
        #Data acquired by the glider during flight
        #Fill gaps between data(interpolate)
        def fillGaps(x,y):
            f = interp1d(x[np.isfinite(x+y)],y[np.isfinite(x+y)], bounds_error=False, fill_value=np.NaN)
            return(f(x))


        #Time of data sampling
        self.timestamp = time
        self.time = date2float(self.timestamp)
        self.pressure = fillGaps(self.time, pres)
        self.longitude = fillGaps(self.time, lon) 
        self.latitude = fillGaps(self.time, lat) 
        self.profile = fillGaps(self.time, lat) 
        self.internal_temperature = fillGaps(self.time, internal_temp) 
        self.external_temperature = fillGaps(self.time, external_temp) 
        self.salinity = fillGaps(self.time, sal) 
        self.ballast = fillGaps(self.time, ballast/1000000) # m^3
        self.pitch =fillGaps(self.time, np.deg2rad(pitch)) # rad
        #calculation of some useful variables        
        self.depth = gsw.z_from_p(self.pressure,self.latitude) # m  !!!!! Note depth (Z) is negative, so diving is negative dZdt
        self.dZdt = np.gradient(self.depth,self.time) # m.s-1
        self.g = gsw.grav(self.latitude,self.pressure)       
        self.SA = gsw.SA_from_SP(self.salinity, self.pressure, self.longitude, self.latitude)
        self.CT = gsw.CT_from_t(self.SA, self.external_temperature, self.pressure)
        self.rho = gsw.rho(self.SA, self.CT, self.pressure)
        
        self.adcp_speed = fillGaps(self.time, adcp_speed)
        
        
        #First and last index to compute flight from
        def indice_debut(sal,external_temp,pres,lon,lat,ballast,pitch,profile,navresource):
            i=0
            while (np.isfinite(sal[i]+external_temp[i]+pres[i]+lon[i]+lat[i]+ballast[i]+pitch[i])==False):
                i+=1
            return i
        self.first_index=indice_debut(sal,external_temp,pres,lon,lat,ballast,pitch,profile,navresource)
        
        def indice_fin(sal,external_temp,pres,lon,lat,ballast,pitch,profile,navresource):
            i=-1
            while (np.isfinite(sal[i]+external_temp[i]+pres[i]+lon[i]+lat[i]+ballast[i]+pitch[i])==False):
                i=i-1
            return i
        self.last_index=indice_fin(sal,external_temp,pres,lon,lat,ballast,pitch,profile,navresource)
        
        
        #Time of solving
        self.dt = 0.1
        #self.t=np.arange(self.time[self.first_index],self.time[self.last_index],self.dt)
        #self.t=np.arange(1598460380, 1598478740,self.dt)
        self.t=np.arange(1598558200, 1598560030, self.dt)
        self.u = np.zeros_like(self.t)
        self.w = np.zeros_like(self.t)
        

        
        #Drone parameters/coefficients ------------------------------------------------------------------------------------
        """
        self.mg=60.772
        self.Vg=0.05947062400832888
        self.comp_p=5.4459542481259674e-06
        self.comp_t=-2.8025987632281052e-05
        self.Cd0=0.1360775344330183
        self.Cd1=3.477573934000217
        self.S=0.09
        """
        
        
        self.param_reference = dict({
            'mg': 60.772, # Vehicle mass in kg
            'm11':2.405,
            'm22':62.59,
            'Vg': 0.05944919244909694, # Reference volume in m**3, with ballast at 0 (with -500 to 500 range), at surface pressure and 20 degrees C
            'S': 0.24, # Wing surface area, m**2
            'Cd0': 0.046601172003459146, #
            'Cd1': 2.2846652475518, #
            'Cl': 1.9551684905403062, #
            'alpha_stall': 10, #
            'alpha_linear': 10, #
            'comp_p': 4.702895408506633e-06, # Pressure dependent hull compression factor
            'comp_t': 8.898950889070907e-05 # Temperature dependent hull compression factor
        })

        self.param = self.param_reference.copy()
        for k,v in param.items():
            self.param[k] = v
        self.param_initial = self.param
        self.regression_parameters = ('m22',)
        
    ###---------------------------------------------- END OF INIT --------------------------------------------- 
    
    
    
        
    ### Regression coefficients --> if "self.param" is modified, they are modified too
    @property        
    def mg(self):
        return self.param['mg']

    @property        
    def m11(self):
        return self.param['m11']
    
    @property        
    def m22(self):
        return self.param['m22']
 
    @property        
    def Vg(self):
        return self.param['Vg']
    
    @property        
    def comp_p(self):
        return self.param['comp_p']
    
    @property        
    def comp_t(self):
        return self.param['comp_t']
    
    @property        
    def S(self):
        return self.param['S']
            
    @property        
    def Cd0(self):
        return self.param['Cd0']
    
    @property        
    def Cd1(self):
        return self.param['Cd1']
                          
    @property        
    def Cl(self):
        return self.param['Cl']
    
    @property        
    def alpha_stall(self):
        return self.param['alpha_stall']
    
    @property        
    def alpha_linear(self):
        return self.param['alpha_linear']
         
            
    #Vertical water velocity to be determined
    @property
    def w_H2O(self):
        # Water column up speed
        return np.interp(self.t, self.time, self.dZdt) - self.w
    
    @property
    def U_H2O(self):
        # Water column total speed
        return np.interp(self.t, self.time, self.adcp_speed) - (self.w**2+self.u**2)
    
   #Cost functions    
    @property
    def R_0(self):
        # Water column total speed
        return np.sqrt(np.nanmean(   self.w_H2O**2   )) 
    
    @property
    def R_1(self):
        # Water column total speed
        return np.sqrt(np.nanmean(   self.U_H2O**2   ))
    
    @property
    def R_2(self):
        # Water column total speed
        return np.sqrt(np.nanmean(   0.5*self.w_H2O**2 + 0.5*self.U_H2O**2   ))



    #Solving process -------------------------------------------------------------------------------------------------

    def _stall_factor(self, _alpha):
        d_alpha = np.abs(_alpha) - self.alpha_linear
        tau_alpha = self.alpha_stall - self.alpha_linear
        res=1.0
        if (np.abs(_alpha)>self.alpha_linear) :
            res=np.exp(-d_alpha/tau_alpha)
        return res
        
    def compute_inverted_mass_matrix(self, pitch):
        C2 = np.cos(pitch)**2
        CS = np.cos(pitch)*np.sin(pitch)
        denom = self.mg**2 + (self.m22+self.m11)*self.mg + self.m11*self.m22
        M11 = ((self.m22-self.m11)*C2 + self.m11 + self.mg) / denom
        M12 = ((self.m22-self.m11)*CS) / denom
        M21 = M12
        M22 = (-(self.m22-self.m11)*C2 + self.m22 + self.mg) / denom
        return M11, M12, M21, M22
    
    def compute_FB_and_Fg(self, g, rho, pressure, ballast, temperature):
        Fb=g*rho*(ballast + self.Vg * (1 - self.comp_p*(pressure-0.8) + self.comp_t*(temperature-10)))
        Fg=self.mg*g
        return Fb, Fg
    
    def compute_lift_and_drag(self, pitch, rho, u, w):
        U = np.sqrt(u**2 + w**2)
        alpha = np.arctan2(w,u) - pitch
        q = 0.5 * rho * self.S * U**2
        L = q * (self.Cl)*(alpha)
        D = q * (   self.Cd0 + self.Cd1*(alpha)**2   )
        return L, D
            
    def F(self, t, u, w):
        pitch=np.interp(t, self.time, self.pitch)
        pressure=np.interp(t, self.time, self.pressure)
        ballast=np.interp(t, self.time, self.ballast)
        rho=np.interp(t, self.time, self.rho)
        g=np.interp(t, self.time, self.g)
        internal_temperature=np.interp(t, self.time, self.internal_temperature)
        external_temperature=np.interp(t, self.time, self.external_temperature)
        
        
        M11, M12, M21, M22 = self.compute_inverted_mass_matrix(pitch)
        Fb, Fg = self.compute_FB_and_Fg(g, rho, pressure, ballast, external_temperature)
        L, D=self.compute_lift_and_drag(pitch, rho, u, w)
        alpha = np.arctan2(w,u) - pitch
        
        Fx=np.sin(pitch + alpha)*L-np.cos(pitch + alpha)*D
        Fy=Fb - Fg -np.cos(pitch + alpha)*L -np.sin(pitch + alpha)*D
        return ( M11*Fx + M12*Fy )
    
    def G(self, t, u, w):
        pitch=np.interp(t, self.time, self.pitch)
        pressure=np.interp(t, self.time, self.pressure)
        ballast=np.interp(t, self.time, self.ballast)
        rho=np.interp(t, self.time, self.rho)
        g=np.interp(t, self.time, self.g)
        internal_temperature=np.interp(t, self.time, self.internal_temperature)
        external_temperature=np.interp(t, self.time, self.external_temperature)
        
        M11, M12, M21, M22 = self.compute_inverted_mass_matrix(pitch)
        Fb, Fg = self.compute_FB_and_Fg(g, rho, pressure, ballast, external_temperature)
        L, D=self.compute_lift_and_drag(pitch, rho, u, w)
        alpha = np.arctan2(w,u) - pitch
        
        Fx=np.sin(pitch + alpha)*L-np.cos(pitch + alpha)*D
        Fy=Fb - Fg -np.cos(pitch + alpha)*L -np.sin(pitch + alpha)*D
        return ( M21*Fx + M22*Fy )
  
    
    def set_initial_conditions(self, u0=None, w0=None):
        if u0 is None :
            self.u[0]=0
        else :
            self.u[0]=u0
        if u0 is None :
            self.w[0]=0
        else :
            self.w[0]=w0
            
    def solveRK4(self):
        dt=self.dt
        t=self.t
        nt=len(t)
        for k in tqdm(range(nt-1)):
            u=self.u[k]
            w=self.w[k]
            
            k1_u=dt*self.F(t[k], u, w)
            k1_w=dt*self.G(t[k], u, w)
            
            k2_u=dt*self.F(t[k]+dt/2, u+k1_u/2, w+k1_w/2)
            k2_w=dt*self.G(t[k]+dt/2, u+k1_u/2, w+k1_w/2)
            
            k3_u=dt*self.F(t[k]+dt/2, u+k2_u/2, w+k2_w/2)
            k3_w=dt*self.G(t[k]+dt/2, u+k2_u/2, w+k2_w/2)
            
            k4_u=dt*self.F(t[k]+dt, u+k3_u, w+k3_w) 
            k4_w=dt*self.G(t[k]+dt, u+k3_u, w+k3_w) 
            
            du=(k1_u +2*k2_u + 2*k3_u + k4_u)/6 
            self.u[k+1]=self.u[k]+du
            
            dw=(k1_w +2*k2_w + 2*k3_w + k4_w)/6
            self.w[k+1]=self.w[k]+dw
            
    #------------------------ Regression ------------------------
    
    def cost_function(self,x_initial):        
        for _istep, _key in enumerate(self.regression_parameters):
            self.param[_key] = x_initial[_istep] * self.param_reference[_key]
        self.solveRK4()
        return self.R_2
    
    def regress(self):
        x_initial = [self.param[_key] / self.param_reference[_key] for _istep,_key in enumerate(self.regression_parameters)]
        print('Initial parameters: ', self.param)
        print('Non-optimised score: '+str(self.cost_function(x_initial)) )
        print('Regressing...')

        R = fmin(self.cost_function, x_initial, disp=True, full_output=True, maxiter=800)
        for _istep,_key in enumerate(self.regression_parameters):
            self.param[_key] = R[0][_istep] * self.param_reference[_key]
        
        x_initial = [self.param[_key] / self.param_reference[_key] for _istep,_key in enumerate(self.regression_parameters)]
        print('Optimised parameters: ', self.param)
        print('Final Optimised score: '+str(self.cost_function(x_initial)) )
        self.solveRK4()
            