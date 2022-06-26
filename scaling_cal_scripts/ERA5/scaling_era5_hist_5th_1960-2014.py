#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 16:19:14 2021

@author: huangzq
"""
import xarray as xr
from numba import float64
from numba import guvectorize
import numpy as np
import pandas as pd
    
gravity = 9.80665
gas_constant = 287.04
kappa = 2.0/7.0
cp = gas_constant/kappa
cp_v = 1870
gas_constant_v = 461.50 
latent_heat_v = 2.5e6

@guvectorize([(float64[:],float64[:],float64[:],float64[:],float64[:],float64[:])], '(n),(n)->(n),(n),(n),(n)',target='parallel',nopython=True)
def saturation_thermodynamics(temp, plev, es, qs, rs, L):
    
    Rd = gas_constant
    Rv = gas_constant_v
    gc_ratio = Rd/Rv  # ratio of gas constants for dry air and water vapor
    
    es0       = 611.21  # saturation vapor pressure at T0 (Pa)
    T0        = 273.16  # (K)
    Ti        = T0 - 23 # (K)

    a3l       = 17.502  # liquid water (Buck 1981)
    a4l       = 32.19   # (K)

    a3i       = 22.587  # ice (Alduchov and Eskridge 1996)
    a4i       = -0.7    # (K)

    
    nc_shape = temp.shape
    for i in range(nc_shape[0]):

        esl       = es0 * np.exp(a3l * (temp[i] - T0)/(temp[i] - a4l))
        esi       = es0 * np.exp(a3i * (temp[i] - T0)/(temp[i] - a4i))
        Ls0       = 2.834e6  # latent heat of sublimation  (J / kg) [+- 0.01 error for 173 K < T < 273 K]
        Ls        = Ls0 * np.ones_like(temp[i])
        
        Lv0       = 2.501e6  # latent heat of vaporization at triple point (J / kg)
        cpl       = 4190     # heat capacity of liquid water (J / kg / K)
        cpv       = cp_v     # heat capacity of water vapor (J / kg / K)
        Lv        = Lv0 - (cpl - cpv) * (temp[i] - T0)
        
        iice      = temp[i] <= Ti
        iliquid   = temp[i] >= T0
        imixed    = (temp[i] > Ti) * (temp[i] < T0)
        
        ## indexing didn't work
        L1 = Ls * iice
        L2 = Lv * iliquid
        a = ((temp[i] - Ti)/(T0 - Ti))**2
        L3 = ((1-a) * Ls + a * Lv) * imixed
        L[i] = L1 + L2 + L3
        
        es1 = esi * iice
        es2 = esl * iliquid
        es3 = ((1-a) * esi + a * esl) * imixed
        es[i] = es1 + es2 + es3

        rs[i]      =  gc_ratio * es[i] / (plev[i] - es[i])
        qs[i]      =  rs[i] / (1 + rs[i])


def sat_deriv(temp, plev): #, dqsat_dp, dqsat_dT, dln_esat_dT
    # Calculates derivatives of the saturation specific humidity wrt
    # temperature and pressure
    # use finite difference approximation
    dp = 0.1
    dT = 0.01
 
    es_p_plus, qs_p_plus,_,_ = saturation_thermodynamics(temp, plev+dp)
    es_p_minus, qs_p_minus,_,_ = saturation_thermodynamics(temp, plev-dp)
    es_T_plus, qs_T_plus,_,_ = saturation_thermodynamics(temp+dT, plev)
    es_T_minus, qs_T_minus,_,_ = saturation_thermodynamics(temp-dT, plev)
    
    dqsat_dp   = (qs_p_plus-qs_p_minus)/(2.0*dp)
    dqsat_dT   = (qs_T_plus-qs_T_minus)/(2.0*dT)
    dln_esat_dT = (np.log(es_T_plus)-np.log(es_T_minus))/2/dT

    return dqsat_dp,dqsat_dT,dln_esat_dT




def moist_adiabatic_lapse_rate(temp, plev):
    # MOIST_ADIABATIC_LAPSE_RATE Returns saturated moist-adiabatic lapse rate.
    # Units are K / m.

    g               = gravity
    cpd             = cp
    cpv             = cp_v
    Rd              = gas_constant
    Rv              = gas_constant_v
    gc_ratio        = Rd / Rv

    es, qs, rs, L = saturation_thermodynamics(temp, plev)
    
    lapse_rate      = g/cpd * (1 + rs) / (1 + cpv/cpd*rs) * (1 + L*rs / Rd / temp) / (1 + L**2 * rs * (1 + rs/gc_ratio)/(Rv * temp**2 * (cpd + rs*cpv)))

    return lapse_rate

@guvectorize(['void(float64[:,:,:,:],float64[:,:,:,:],float64[:,:,:])'], '(m,n,o,p),(m,n,o,p)->(m,n,o)',target='parallel',nopython=True) 
def integrate(neg_dqsat_dp_total_omega, plev_level, F):
    nc_shape = plev_level.shape
    for i in range(nc_shape[0]):
        for j in range(nc_shape[1]):
            for k in range(nc_shape[2]):
                plev_level_len = len(plev_level[0,0,0,:])
                dx1 = np.zeros(plev_level_len)
                for m in range(plev_level_len):
                    if m == 0:
                        dx1[m] = (plev_level[0,0,0,m+1] - plev_level[0,0,0,m])/2
                    elif m == plev_level_len -1:
                        dx1[m] = (plev_level[0,0,0,m] - plev_level[0,0,0,m-1])/2
                    else:
                        dx1[m] = (plev_level[0,0,0,m+1] - plev_level[0,0,0,m-1])/2

                F[i,j,k] = np.sum(neg_dqsat_dp_total_omega[i,j,k,:] * dx1)

@guvectorize(['void(float64[:,:,:,:],float64[:,:,:,:],float64[:,:,:,:])'], '(m,n,o,p),(m,n,o,p)->(m,n,o,p)',target='parallel',nopython=True) 
def dqsat_dp_total_change(lapse_rate_env,dqsat_dp_total,dqsat_dp_total_change):
    crit_lapse_rate = 0.002 # (k/m) for tropopause
    nc_shape = lapse_rate_env.shape
    for i in range(nc_shape[0]):
        for j in range(nc_shape[1]):
            for k in range(nc_shape[2]):
                itrop = np.where(lapse_rate_env[i,j,k,:] > crit_lapse_rate)[0]
                dqsat_dp_total_change[i,j,k,:]  = dqsat_dp_total[i,j,k,:] 
                if itrop.size!=0:
                    if np.max(itrop)+1<len(lapse_rate_env[0,0,0,:]):
                        dqsat_dp_total_change[i,j,k,np.max(itrop)+1:]=0

def get_omega_aveTime():
    ## firstly, extract precipitation extreme, and calculate wap multiyear mean conditioned on occurrence of precipitation extreme
    
    wap_alltime_cond = np.zeros((yearE[-1]-yearS[0]+1,8,145,288))
    
    for i in range(len(yearS)):
        
        prec_fp_i = getFilepath_pr(yearE[i])
        prec_f_i  = xr.open_dataset(prec_fp_i)
        
        wap_fp_i = getFilepath1('wap', yearS[i], yearE[i])
        wap_f_i = xr.open_dataset(wap_fp_i)
        
        prec_f_i = prec_f_i.sel(time=prec_f_i.time.dt.year==yearS[i])
        
        if(len(prec_f_i['time']) % 365 != 0): ### dealing with leap year
            prec_f_i = prec_f_i.sel(time=~((prec_f_i.time.dt.month == 2) & (prec_f_i.time.dt.day == 29)))
            wap_f_i = wap_f_i.sel(time=~((wap_f_i.time.dt.month == 2) & (wap_f_i.time.dt.day == 29)))
        
        prec_data_i = prec_f_i['tp']
        wap_data_i  = wap_f_i['w']
        wap_data_i  = wap_data_i.values
        
        prec_yMax_idx = np.argmax(prec_data_i.values,axis=0)
        prec_yMax_idx = np.moveaxis(np.ones([145,288,365]) * range(365), -1, 0) == prec_yMax_idx
        
        wap_data_i  = np.moveaxis(wap_data_i,0,1)
        wap_cond_i  = wap_data_i * (prec_yMax_idx)
        wap_cond_i  = np.nansum(wap_cond_i,axis=1,keepdims=True)
        
        wap_alltime_cond[i,:,:,:] = wap_cond_i[:,0,:,:]
    
    myvar_wap_alltime_cond = xr.DataArray(data = wap_alltime_cond, dims=['time','level','latitude','longitude'], name = 'vertical_velocity_cond',
                                      coords=dict(time = pd.date_range(start=str(yearS[0]),end = str(yearE[-1]+1),freq='Y'),level=wap_f_i.coords['level'],latitude=wap_f_i.coords['latitude'],longitude=wap_f_i.coords['longitude']),
                                      attrs = dict(description = 'vertical velocity conditioned on the occurrence of precipitation extreme',units = 'Pa s-1'))
    wap_alltime_cond_wp      = 'procData/era5/'+'wap_alltime_cond_era5_'+str(yearS[0])+'-'+str(yearE[-1])+'.nc'
    myvar_wap_alltime_cond.to_netcdf(wap_alltime_cond_wp)
    
    wap_avgtime_cond = np.nanmean(wap_alltime_cond,axis=0)
    myvar_wap_avgtime_cond = xr.DataArray(data = wap_avgtime_cond, dims=['level','latitude','longitude'], name = 'average_vertical_velocity_cond',
                                          coords=dict(level=wap_f_i.coords['level'],latitude=wap_f_i.coords['latitude'],longitude=wap_f_i.coords['longitude']),
                                          attrs = dict(description = 'multiyear average vertical velocity conditioned on the occurrence of precipitation extreme',units = 'Pa s-1'))
    wap_avgtime_cond_wp      = 'procData/era5/'+'wap_avgtime_cond_era5_'+str(yearS[0])+'-'+str(yearE[-1])+'.nc'
    myvar_wap_avgtime_cond.to_netcdf(wap_avgtime_cond_wp)
    
    return wap_avgtime_cond
        

def scaling(omega, temp, plev, ps):
        
    # criterion for identifying tropopaus
    plev_mask = 0.05e5 # (Pa) exclude levels above this as a fail-safe
    dqsat_dp, dqsat_dT,_ = sat_deriv(temp,plev)  ## 4min for a loop
    es, qsat, rsat, latent_heat = saturation_thermodynamics(temp, plev)  ## 1min for a loop
    lapse_rate = moist_adiabatic_lapse_rate(temp, plev) ## 1min for a loop
    # virtual temperature
    temp_virtual = temp*(1.0+qsat*(gas_constant_v/gas_constant-1.0))

    # density
    rho = plev/gas_constant/temp_virtual
    dT_dp = lapse_rate/gravity/rho

    # find derivative of saturation specific humidity with respect to pressure along 
    # a moist adiabat at the given temperature and pressure for each level
    dqsat_dp_total = dqsat_dp+dqsat_dT*dT_dp
    # mask above tropopause using simple lapse rate criterion
    dT_dp_env = np.gradient(temp, plev[0,0,0,:], axis=3)
    lapse_rate_env = dT_dp_env*rho*gravity

    dqsat_dp_total = dqsat_dp_total_change(lapse_rate_env,dqsat_dp_total)

    # mask above certain level as fail safe
    dqsat_dp_total[plev<plev_mask]=0
    dqsat_dp_total_omega = dqsat_dp_total*omega
    
    omega_aveTime = omega_filled_aveTime
    dqsat_dp_total_omega_aveTime = dqsat_dp_total*omega_aveTime

    # replaces nans with zeros as subsurface values should not contribute
    # to the column integral
    dqsat_dp_total_omega[np.isnan(dqsat_dp_total_omega)]=0
    dqsat_dp_total_omega_aveTime[np.isnan(dqsat_dp_total_omega_aveTime)]=0

    # also use surface pressure to zero subsurface value
    ps = ps[:,:,:,np.newaxis]
    kbot = plev <= ps
    dqsat_dp_total_omega = dqsat_dp_total_omega * kbot
    dqsat_dp_total_omega_aveTime = dqsat_dp_total_omega_aveTime * kbot

    # integrate in the vertical
    precip = -integrate(-dqsat_dp_total_omega,plev)/gravity
    precip_thermo = -integrate(-dqsat_dp_total_omega_aveTime,plev)/gravity

    return precip,precip_thermo


#############################  
def month2daily(month_file,ys,ye):
    month_data =  xr.open_dataset(month_file)
    month_data = month_data['sp'].sel(time=slice(str(ys),str(ye)))
    days       = [31,28,31,30,31,30,31,31,30,31,30,31] * (ye - ys + 1)
    latlen     = month_data.shape[1]
    lonlen     = month_data.shape[2]
    dayly_data = np.zeros(((ye -ys +1)*365,latlen,lonlen))
    iter_time  = 0
    for i in range(len(days)):
        for j in range(days[i]):
            dayly_data[iter_time,:,:] = month_data[i,:,:]
            iter_time += 1
    return dayly_data


var_name = dict(ta = 'ta', wap = 'wap', ps = 'ps',tas = 'tas')
var_name2 = dict(scaling_cond = 'scaling', scaling_thermo_cond = 'scaling_thermo', 
    scaling_dynamic_cond = 'scaling_dynamic', sat_thermo_cond = 'sat_thermo', wap_mean_cond = 'wap_mean',
    prec_cond = 'prec', tas_cond = 'tas')

freq     = dict(ta = 'day', wap = 'day', ps = 'Amon')
src_id   = 'era5'
yearS    = [i for i in range(1960,2015)]
yearE    = [i for i in range(1960,2015)]
dataTime = [str(ys)+'0101-'+str(ye)+'1231' for ys,ye in zip(yearS,yearE)]

def getFilepath1(var, ys, ye):
    filePath = 'rawData/era5/daily/' + var_name[var]+'_daily_era5_reanalysis_'+ str(ys)+ '-' + str(ye) + '.nc'
    return filePath

def getFilepath2(var):
    filePath =  'rawData/era5/' + var + '_monthly_era5_reanalysis_' + '1960' +'-'+ '2014' +'.nc'
    return filePath

def writeFilepath(var,ys,ye):
    filePath = 'procData/era5/' + var_name2[var]+'_'+'cond'+'_era5_'+str(ys)+'-'+str(ye)+'.nc'
    return filePath

def writeFilepath2(var,ys,ye):
    filePath = 'procData/era5/' + var +'_era5_'+str(ys)+'-'+str(ye)+'.nc'
    return filePath

def extrt_yMax(varArray):
    nyear = int(varArray.shape[0] / 365)
    varArray_yMax = np.zeros((nyear,varArray.shape[1],varArray.shape[2]))
    for i in range(nyear):
        varArray_yMax[i,:,:] = varArray[365*i:365*(i+1),:,:].max(axis = 0)
    return varArray_yMax

def getFilepath_pr(i):
    filePath = 'rawData/era5/daily/prec_daily_era5_reanalysis_'+str(i)+'-'+str(i)+'.nc'
    return filePath

if __name__ == "__main__":
    
    ps_File = getFilepath2('ps')
    ps_f = xr.open_dataset(ps_File)
    ps_data = ps_f['sp']
    
    omega_filled_aveTime = get_omega_aveTime()
    omega_filled_aveTime = np.moveaxis(omega_filled_aveTime,0,-1)[np.newaxis,:,:,::-1]
    scaling_allyear = np.zeros((yearE[-1]-yearS[0]+1,145,288))
    scaling_thermo_allyear = scaling_allyear.copy()
    scaling_budget_allyear = scaling_allyear.copy()
    sat_thermo_allyear = scaling_allyear.copy()
    wap_mean_allyear   = scaling_allyear.copy()
    prec_cond_allyear  = scaling_allyear.copy()
    tas_cond_allyear   = scaling_allyear.copy()

    for i in range(len(yearS)):
        
        prec_fp_i = getFilepath_pr(yearE[i])
        prec_f_i  = xr.open_dataset(prec_fp_i)
        prec_f_i = prec_f_i.sel(time=prec_f_i.time.dt.year==yearS[i])
        
        wap_fp_i = getFilepath1('wap', yearS[i], yearE[i])
        wap_f_i = xr.open_dataset(wap_fp_i)
        
        ta_fp_i = getFilepath1('ta', yearS[i], yearE[i])
        ta_f_i = xr.open_dataset(ta_fp_i)
        
        tas_fp_i = getFilepath1('tas', yearS[i], yearE[i])
        tas_f_i = xr.open_dataset(tas_fp_i)
        
        if(len(prec_f_i['time']) % 365 != 0): ### dealing with leap year
            prec_f_i = prec_f_i.sel(time=~((prec_f_i.time.dt.month == 2) & (prec_f_i.time.dt.day == 29)))
            wap_f_i = wap_f_i.sel(time=~((wap_f_i.time.dt.month == 2) & (wap_f_i.time.dt.day == 29)))
            ta_f_i = ta_f_i.sel(time=~((ta_f_i.time.dt.month == 2) & (ta_f_i.time.dt.day == 29)))
            tas_f_i = tas_f_i.sel(time=~((tas_f_i.time.dt.month == 2) & (tas_f_i.time.dt.day == 29)))
        
        prec_data_i = prec_f_i['tp']
        prec_yMax_i = prec_data_i.max(axis=0)
        wap_data_i  = wap_f_i['w']
        ta_data_i = ta_f_i['t']
        tas_data_i = tas_f_i['t2m']
        
        # prec_yMax_idx = prec_data_i == prec_yMax_i
        prec_yMax_idx = np.argmax(prec_data_i.values,axis=0)
        prec_yMax_idx = np.moveaxis(np.ones([145,288,365]) * range(365), -1, 0) == prec_yMax_idx
        
        # prec_yMax_idx['time'] = wap_data_i.time
        wap_data_i  = np.moveaxis(wap_data_i.values, 1,0)
        wap_cond_i  = wap_data_i * prec_yMax_idx
        omega_filled = wap_cond_i
        omega_filled = omega_filled.sum(axis=1,keepdims=True)
        omega_filled = np.moveaxis(omega_filled,0,-1)[:,:,:,::-1]
        
        # prec_yMax_idx['time'] = ta_data_i.time
        ta_data_i = np.moveaxis(ta_data_i.values,1,0)
        ta_cond_i = ta_data_i * (prec_yMax_idx)
        ta_filled = ta_cond_i
        ta_filled = ta_filled.sum(axis=1,keepdims=True)
        ta_filled = np.moveaxis(ta_filled, 0, -1)[:,:,:,::-1]
        
        # prec_yMax_idx['time'] = tas_data_i.time
        tas_cond_i = np.squeeze(tas_data_i).values * (prec_yMax_idx)
        tas_filled = tas_cond_i
        tas_filled = tas_filled.sum(axis=0,keepdims=True)
        
        plev_test = ta_f_i['level'][::-1]* 100
        plev_test = plev_test.expand_dims(['time','latitude','longitude'])
        plev_test = np.ones(ta_filled.shape) * plev_test.values
        
        ps_daily = month2daily(ps_File, yearS[i], yearE[i])
        ps_cond  = ps_daily * (prec_yMax_idx)
        ps_cond  = ps_cond.sum(axis=0, keepdims=True)
        
        precip, precip_thermo = scaling(omega_filled,ta_filled,plev_test,ps_cond)
        _, sat_thermo, _, _ = saturation_thermodynamics(ta_filled, plev_test)
        sat_thermo = np.nansum(sat_thermo,axis = 3)
        wap_mean = np.nanmean(omega_filled,axis = 3)
        
        scaling_allyear[i,:,:] = precip
        scaling_thermo_allyear[i,:,:] = precip_thermo
        scaling_budget_allyear[i,:,:] = precip - precip_thermo
        sat_thermo_allyear[i,:,:] = sat_thermo
        wap_mean_allyear[i,:,:] = wap_mean
        prec_cond_allyear[i,:,:] = prec_yMax_i
        tas_cond_allyear[i,:,:] = tas_filled
        
    myvar_precip = xr.DataArray(data = scaling_allyear,dims=['time','latitude','longitude'], name = 'scaling',
                                coords=dict(time = pd.date_range(start=str(yearS[0]),end = str(yearE[-1]+1),freq='Y'),latitude  = prec_data_i.coords['latitude'],longitude  = prec_data_i.coords['longitude']),
                                attrs = dict(description = 'full precipitation scaling conditioned on occurrence of precipitation extreme',units = 'kg m-2 s-1'))
    
    myvar_precip_thermo = xr.DataArray(data = scaling_thermo_allyear,dims=['time','latitude','longitude'], name = 'scaling_thermo',
                                       coords=dict(time = pd.date_range(start=str(yearS[0]),end = str(yearE[-1]+1),freq='Y'),latitude  = prec_data_i.coords['latitude'],longitude  = prec_data_i.coords['longitude']),
                                       attrs = dict(description = 'thermodynamic precipitation scaling conditioned on occurrence of precipitation extreme',units = 'kg m-2 s-1'))
    
    myvar_precip_dynamic = xr.DataArray(data = scaling_budget_allyear,dims=['time','latitude','longitude'], name = 'scaling_budget',
                                       coords=dict(time = pd.date_range(start=str(yearS[0]),end = str(yearE[-1]+1),freq='Y'),latitude  = prec_data_i.coords['latitude'],longitude  = prec_data_i.coords['longitude']),
                                       attrs = dict(description = 'dynamic precipitation scaling conditioned on occurrence of precipitation extreme, calculated as the difference between full scaling and thermodynamic scaling',units = 'kg m-2 s-1'))
    
    myvar_sat_thermo = xr.DataArray(data = sat_thermo_allyear, dims=['time','latitude','longitude'], name = 'sat_thermo',
                                    coords=dict(time = pd.date_range(start=str(yearS[0]),end = str(yearE[-1]+1),freq='Y'),latitude  = prec_data_i.coords['latitude'],longitude  = prec_data_i.coords['longitude']),
                                    attrs = dict(description = 'integrated saturation moisture conditioned on occurrence of precipitation extreme',units = 'undefined'))
    
    myvar_wap_mean = xr.DataArray(data = wap_mean_allyear, dims=['time','latitude','longitude'], name = 'wap_mean',
                                  coords=dict(time = pd.date_range(start=str(yearS[0]),end = str(yearE[-1]+1),freq='Y'),latitude  = prec_data_i.coords['latitude'],longitude  = prec_data_i.coords['longitude']),
                                  attrs = dict(description = 'vertically mean wap conditioned on occurrence of precipitation extreme',units = 'Pa s-1'))
    
    myvar_prec = xr.DataArray(data = prec_cond_allyear, dims=['time','latitude','longitude'], name = 'prec_cond',
                              coords=dict(time = pd.date_range(start=str(yearS[0]),end = str(yearE[-1]+1),freq='Y'),latitude  = prec_data_i.coords['latitude'],longitude  = prec_data_i.coords['longitude']),
                              attrs = dict(description = '1day precipitation extreme',units = 'm'))
    
    myvar_tas = xr.DataArray(data = tas_cond_allyear, dims=['time','latitude','longitude'], name = 'tas_cond',
                             coords=dict(time = pd.date_range(start=str(yearS[0]),end = str(yearE[-1]+1),freq='Y'),latitude  = prec_data_i.coords['latitude'],longitude  = prec_data_i.coords['longitude']),
                             attrs = dict(description = 'surface temperature conditioned on occurrence of precipitation extreme',units = 'K'))

    scaling_wp = writeFilepath('scaling_cond',yearS[0], yearE[-1])
    scaling_thermo_wp = writeFilepath('scaling_thermo_cond', yearS[0], yearE[-1])
    scaling_dynamic_wp = writeFilepath('scaling_dynamic_cond', yearS[0], yearE[-1])
    sat_thermo_wp = writeFilepath('sat_thermo_cond', yearS[0], yearE[-1])
    wap_mean_wp = writeFilepath('wap_mean_cond', yearS[0], yearE[-1])
    prec_cond_wp = writeFilepath('prec_cond', yearS[0], yearE[-1])
    tas_cond_wp  = writeFilepath('tas_cond', yearS[0], yearE[-1])
    
    myvar_precip.to_netcdf(scaling_wp)
    myvar_precip_thermo.to_netcdf(scaling_thermo_wp)
    myvar_precip_dynamic.to_netcdf(scaling_dynamic_wp)
    myvar_sat_thermo.to_netcdf(sat_thermo_wp)
    myvar_wap_mean.to_netcdf(wap_mean_wp)
    myvar_prec.to_netcdf(prec_cond_wp)
    myvar_tas.to_netcdf(tas_cond_wp)
