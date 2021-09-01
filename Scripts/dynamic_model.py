# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:18:05 2021

@author: johan
"""

import numpy as np
import panda as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import gsw


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
    def __init__(self,time,sal,temp,pres,lon,lat,ballast,pitch,profile,navresource):
        #Data acquired by the glider durong flight
        self.timestamp = time
        self.time = date2float(self.timestamp)
        self.pressure = pres
        self.longitude = lon
        self.latitude = lat
        self.profile = profile
        self.temperature = temp
        self.salinity = sal
        self.ballast = ballast/1000000 # m^3
        self.pitch = np.deg2rad(pitch) # rad
        
        #Correstion to some data        
        self.depth = gsw.z_from_p(self.pressure,self.latitude) # m . Note depth (Z) is negative, so diving is negative dZdt
        self.dZdt = np.gradient(self.depth,self.time) # m.s-1
        self.g = gsw.grav(self.latitude,self.pressure)       
        self.SA = gsw.SA_from_SP(self.salinity, self.pressure, self.longitude, self.latitude)
        self.CT = gsw.CT_from_t(self.SA, self.temperature, self.pressure)
        self.rho = gsw.rho(self.SA, self.CT, self.pressure)
        
        #Drone parameters
        self.AR = 7
        self.eOsborne = 0.8
        self.Cd1_hull = 2.1
        self.Omega = 0.75
        self.mg=60.772
        self.m11=0.2*60.772
        self.m22=0.92*60.772
        self.Vg=59.015 / 1000
        self.comp_p=4.5e-06
        self.comp_t=-6.5e-05
        self.Cd_0=0.11781
        self.Cd_1=2.94683
        self.aw= 3.82807
        self.ah= 3.41939
        self.S=0.09

        #Drone speed to be determined
        self.dt = 1
        self.u = np.zeros_like(np.arrange(self.time[0],self.time[-1],dt))
        self.w = np.zeros_like(np.arrange(self.time[0],self.time[-1],dt))
        
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
        Fb=g*rho*(ballast + self.Vg * (1 - self.comp_p*pressure - self.comp_t*(temperature-10)))
        Fg=self.mg*g
        return Fb, Fg
    
    def compute_lift_and_drag(self, pitch, rho, u, w):
        U = np.sqrt(u**2 + w**2)
        alpha = np.arctan2(w,u) - pitch
        q = 0.5 * rho * self.S * U**2
        L = q * (self.aw + self.ah)*alpha
        D = q * (self.Cd0 + self.Cd1*alpha**2)
        return L, D
            
    def f(self, t, u, w):
        pitch=np.interp(t, self.time, self.pitch)
        pressure=np.interp(t, self.time, self.pressure)
        ballast=np.interp(t, self.time, self.ballast)
        rho=np.interp(t, self.time, self.rho)
        g=np.interp(t, self.time, self.g)
        T=np.interp(t, self.time, self.T)
        
        M11, M12, M21, M22 = self.compute_inverted_mass_matrix(pitch)
        Fb, Fg = self.compute_FB_and_Fg(g, rho, pressure, ballast, temperature)
        L, D=self.compute_lift_and_drag(pitch, rho, u, w)
        alpha = np.arctan2(w,u) - pitch
        
        Fx=np.sin(pitch + alpha)*L-cos(pitch + alpha)*D
        Fy=Fb - Fg -cos(pitch + alpha)*L -sin(pitch + alpha)*D
        
        return ( M11*Fx + M12*Fy )
    
    def g(self, t, u, w):
        pitch=np.interp(t, self.time, self.pitch)
        pressure=np.interp(t, self.time, self.pressure)
        ballast=np.interp(t, self.time, self.ballast)
        rho=np.interp(t, self.time, self.rho)
        g=np.interp(t, self.time, self.g)
        T=np.interp(t, self.time, self.T)
        
        M11, M12, M21, M22 = self.compute_inverted_mass_matrix(pitch)
        Fb, Fg = self.compute_FB_and_Fg(g, rho, pressure, ballast, temperature)
        L, D=self.compute_lift_and_drag(pitch, rho, u, w)
        alpha = np.arctan2(w,u) - pitch
        
        Fx=np.sin(pitch + alpha)*L-cos(pitch + alpha)*D
        Fy=Fb - Fg -cos(pitch + alpha)*L -sin(pitch + alpha)*D
        
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
        t=np.arrange(self.time[0],self.time[-1],dt) #????
        nt=t.size        
        for k in range(nt-1):
            k1_u=dt*self.f(t[k], u[k], w[k])
            k1_w=dt*self.g(t[k], u[k], w[k])
            
            k2_u=dt*self.f(t[k]+dt/2, u[k]+k1_u/2, w[k]+k1_w/2)
            k2_w=dt*self.g(t[k]+dt/2, u[k]+k1_u/2, w[k]+k1_w/2)
            
            k3_u=dt*self.f(t[k]+dt/2, u[k]+k2_u/2, w[k]+k2_w/2)
            k3_w=dt*self.g(t[k]+dt/2, u[k]+k2_u/2, w[k]+k2_w/2)
            
            k4_u=dt*self.f(t[k]+dt, u[k]+k3_u, w[k]+k3_w) 
            k4_w=dt*self.g(t[k]+dt, u[k]+k3_u, w[k]+k3_w) 
            
            du=(k1_u +2*k2_u + 2*k3_u + k4_u)/6
            u[k+1]=u[k]+du
            
            dw=(k1_w +2*k2_w + 2*k3_w + k4_w)/6
            w[k+1]=w[k]+dw
            
            
        
        