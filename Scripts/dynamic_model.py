# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:18:05 2021

@author: johan
"""

import numpy as np
import pandas as pd
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
        #Fill gaps between data(interpolate)
        def fillGaps(x,y):
            f = interp1d(x[np.isfinite(x+y)],y[np.isfinite(x+y)], bounds_error=False, fill_value=np.NaN)
            return(f(x))
        
        self.timestamp = time
        self.time = date2float(self.timestamp)
        self.pressure = fillGaps(self.time, pres)
        self.longitude = fillGaps(self.time, lon)
        self.latitude = fillGaps(self.time, lat)
        self.profile = fillGaps(self.time, lat)
        self.temperature = fillGaps(self.time, temp)
        self.salinity = fillGaps(self.time, sal)
        self.ballast = fillGaps(self.time, ballast/1000000) # m^3
        self.pitch = fillGaps(self.time, np.deg2rad(pitch)) # rad
        
        #calculation of some data        
        self.depth = gsw.z_from_p(self.pressure,self.latitude) # m . Note depth (Z) is negative, so diving is negative dZdt
        self.dZdt = np.gradient(self.depth,self.time) # m.s-1
        self.g = gsw.grav(self.latitude,self.pressure)       
        self.SA = gsw.SA_from_SP(self.salinity, self.pressure, self.longitude, self.latitude)
        self.CT = gsw.CT_from_t(self.SA, self.temperature, self.pressure)
        self.rho = gsw.rho(self.SA, self.CT, self.pressure)
        
        #First and last index to fompute flight from
        def indice_debut(sal,temp,pres,lon,lat,ballast,pitch,profile,navresource):
            i=0
            while (np.isfinite(sal[i]+temp[i]+pres[i]+lon[i]+lat[i]+ballast[i]+pitch[i])==False):
                i+=1
            return i
        self.first_index=indice_debut(sal,temp,pres,lon,lat,ballast,pitch,profile,navresource)
        
        def indice_fin(sal,temp,pres,lon,lat,ballast,pitch,profile,navresource):
            i=-1
            while (np.isfinite(sal[i]+temp[i]+pres[i]+lon[i]+lat[i]+ballast[i]+pitch[i])==False):
                i=i-1
            return i
        self.last_index=indice_fin(sal,temp,pres,lon,lat,ballast,pitch,profile,navresource)
        
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
        self.Cd0=0.11781
        self.Cd1=2.94683
        self.aw= 3.82807
        self.ah= 3.41939
        self.S=0.09

        #Drone speed to be determined
        self.dt = 1
        #self.t=np.arange(self.time[self.first_index],self.time[self.last_index],self.dt)
        self.t=np.arange(self.time[self.first_index],self.time[self.first_index]+10000,self.dt)
        self.u = np.zeros_like(self.t)
        self.w = np.zeros_like(self.t)
        
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
        if u>1:
            u=1
        if w>1:
            w=1
        U = np.sqrt(u**2 + w**2)
        alpha = np.arctan2(w,u) - pitch
        q = 0.5 * rho * self.S * U**2
        L = q * (self.aw + self.ah)*alpha
        D = q * (self.Cd0 + self.Cd1*alpha**2)
        return L, D
            
    def F(self, t, u, w):
        pitch=np.interp(t, self.time, self.pitch)
        pressure=np.interp(t, self.time, self.pressure)
        ballast=np.interp(t, self.time, self.ballast)
        rho=np.interp(t, self.time, self.rho)
        g=np.interp(t, self.time, self.g)
        temperature=np.interp(t, self.time, self.temperature)
        
        M11, M12, M21, M22 = self.compute_inverted_mass_matrix(pitch)
        Fb, Fg = self.compute_FB_and_Fg(g, rho, pressure, ballast, temperature)
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
        temperature=np.interp(t, self.time, self.temperature)
        
        M11, M12, M21, M22 = self.compute_inverted_mass_matrix(pitch)
        Fb, Fg = self.compute_FB_and_Fg(g, rho, pressure, ballast, temperature)
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
        for k in range(nt-1):
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