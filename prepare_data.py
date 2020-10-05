import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammainc as incomplete_gamma

from constants import *
if not remote:
    import kepler_io


cut = 3 #for normalisation
sigma_correction = 0.9865783925581086 # = np.sqrt(scipy.special.gammainc(1.5, 0.5* cut**2)/scipy.special.gammainc(0.5, 0.5* cut**2)), computed for cut = 3, it is chosen such that: STD[{N(x, sigma=1) | x < cut}]



def kepler(KEPLER_ID):
    """downloads light curve with KEPLER_ID, iterates to get sigma of data"""
    KEPLER_DATA_DIR = "../kepler/"
    file_names = kepler_io.kepler_filenames(KEPLER_DATA_DIR, KEPLER_ID, long_cadence=long_cadence)
    all_time, all_flux = kepler_io.read_kepler_light_curve(file_names)
    Identificator = []
    for i in range(len(all_time)):
        #all_flux[i], sigma = normalization(all_flux[i])
        all_flux[i] = (all_flux[i]/np.median(all_flux[i]))-1
        Identificator += [i for num in range(len(all_flux[i]))]
        #print(np.average(all_flux[i]))
    Time = np.concatenate(all_time)
    Time -= 131.512439459#Time[0]
    Flux = np.concatenate(all_flux)
    #Flux -= np.average(Flux)
    #Flux, sigma = normalization(Flux)
    Flux /= sigma_in_star_flux_units #it is neccesary to first eliminate spline to get a good sigma. This number was obtained from normalization on a flux with eliminated spline
    d = {'Flux': Flux, 'Time': Time, 'Identificator': Identificator}
    df = pd.DataFrame(data=d)
    if long_cadence:
        df.to_csv('data/kepler' + str(KEPLER_ID) + '.txt', index=False)
    else:
        df.to_csv('data/kepler_short' + str(KEPLER_ID) + '.txt', index=False)



def discard_planet(Time, Flux, period, phase, half_time_transit):
    """eliminates flux where there was a planet. That is:
        Flux[k*period + start: k*period + end] = 0
        Variance[k*period + start: k*period + end] = inf
        for k natural number"""
    ne_prehodna_tocka = np.array([(math.fmod(t - phase, period) < period - half_time_transit) and (math.fmod(t-phase, period)>half_time_transit) for t in Time]) #true if a point is not during a time of transit
    return np.copy(Time[ne_prehodna_tocka]), np.copy(Flux[ne_prehodna_tocka])


def normalization_kepler(KEPLER_ID):
    file_names = kepler_io.kepler_filenames("../../kepler/", KEPLER_ID, long_cadence=long_cadence)
    all_time, all_flux = kepler_io.read_kepler_light_curve(file_names)
    for i in range(len(all_time)):
        all_flux[i] = (all_flux[i] / np.median(all_flux[i])) -1
    Time = np.concatenate(all_time)
    #Time -= 131.512439459 #for Kepler90 short cadence to make it compatible with long cadence

    Flux = np.concatenate(all_flux)
    flux_without_spline = np.concatenate([all_flux[i] - all_spline[i] for i in range(len(all_flux))])

    Time, Flux = discard_planet(Time, Flux, T, phase, time_half_transit)

    #average, sigma = normalization(Flux)
    average, sigma = normalization(flux_without_spline)

    
    Flux = (Flux - average) / sigma

    #plot distribution of flux and flux as a function of time to check by eye that everything is ok.
    plt.subplot(1, 2, 1)
    maska = (Flux > -15) & (Flux < 15)
    p_cel, x = np.histogram(Flux[maska], bins=50)
    p = p_cel / ((x[1:] - x[:-1]) * len(Flux))
    x = 0.5 * (x[1:] + x[:-1])
    plt.plot(x, p, '.', color='black', label='noise distribution')

    plt.subplot(1, 2, 2)
    plt.plot(Time, Flux, '.')
    plt.ylim(-10, 10)
    plt.savefig('C:/Users/USER/Documents/Physics/kepler/non_planets_candidates/including_star_plots/'+ str(KEPLER_ID) + '.png')
    plt.close()
    #plt.show()

    df = pd.DataFrame(data= {'Flux': Flux, 'Time': Time})
    df.to_csv('C:/Users/USER/Documents/Physics/kepler/non_planets_candidates/including_star/' + str(KEPLER_ID) + '.txt', index=False)


def mask_outliers(flux, limit, region):
    """returns mask that identifies all outliers := points that are more than limit away from median"""
    avg = np.median(flux)
    
    if region == 'upper':
        mask = flux- avg < limit
    elif region == 'lower':
        mask = flux- avg > limit
    elif region == 'both':
        mask = np.abs(flux- avg) > limit
    else:
        print('parameter region should be on of the following: "upper", "lower" ali "both" ')
    return np.array(mask)

def read_kepler(export_path, KEPLER_ID):
    KEPLER_DATA_DIR = "../kepler/"
    file_names = kepler_io.kepler_filenames(KEPLER_DATA_DIR, KEPLER_ID)
    all_time, all_flux = kepler_io.read_kepler_light_curve(file_names)
    
    for f in all_flux:
        f /= np.median(f)
    Time_direktni = np.concatenate(all_time)
    Time_direktni -= Time_direktni[0]
    Flux_direktni = np.concatenate(all_flux)
    Flux_direktni -= np.average(Flux_direktni)

    d = {'Flux': Flux_direktni, 'Time': Time_direktni}
    df = pd.DataFrame(data=d)
    # plt.plot(df["Time"], df["Flux"], '.')
    # plt.show()
    df.to_csv(export_path, index=False)

def normalization(Flux):
    """iterates to get the right sigma of Flux and normalizes Flux to sigma = 1"""
    sigma = np.std(Flux)
    average= np.median(Flux)
    for i in range(20):
        flux_no_outliers = np.copy(Flux[ np.abs(Flux-average) < cut * sigma])  # ~ negates mask_outliers
        average = np.median(flux_no_outliers)
        sigma = np.std(flux_no_outliers) / sigma_correction
    return average, sigma


def normalize_identify_holes(KEPLER_ID):
    file_names = kepler_io.kepler_filenames("../../kepler/", KEPLER_ID, long_cadence=long_cadence)
    all_time, all_flux = kepler_io.read_kepler_light_curve(file_names)
    for i in range(len(all_time)):
        all_flux[i] = (all_flux[i] / np.median(all_flux[i])) - 1

    Time = np.concatenate(all_time)
    Flux = np.concatenate(all_flux)

    average, sigma = normalization(Flux)
    Flux = (Flux - average) / sigma

    Time -= Time[0]

    return Time, Flux, sigma
    #df = pd.DataFrame(data={'Flux': Flux, 'Time': Time})
    #df.to_csv(home + 'data/time_flux_holes_' + str(id) + '.txt', index=False)

def zeros(time, Flux, export_path):
    """Finds parts of signal where there are missing measurements and adds 0 flux and infinite variance there. 
    assumes variance si 1 if we have a measurement"""
    # reads data
    # df = pd.read_csv(import_path)
    # Flux = np.array(df["Flux"])
    # time = np.array(df["Time"])

    #normalizes times and finds where there is missing data
    difference_Time = (time[1:] - time[:-1])
    difference_Time /= dt  # dimensionless t[i+1]-t[i]
    difference_Time = np.round(difference_Time, 0).astype(int)
    mask = difference_Time > 1
    indexes = (np.arange(len(difference_Time), dtype = int)+1)[mask]
    add = difference_Time[mask] - 1
    Variance = np.ones(len(Flux))

    #fills with zeros where there is no data and fixes Variance there to infinity
    indexes_all_repeated = []
    for i in range(len(indexes)):
        indexes_all_repeated = indexes_all_repeated + [indexes[i] for j in range(add[i])]
    indexes_all_repeated = indexes_all_repeated + [len(Flux) for i in range(300)] #zero paddling
    Flux = np.insert(Flux, indexes_all_repeated, np.zeros(len(indexes_all_repeated)))
    Variance = np.insert(Variance, indexes_all_repeated, np.inf*np.ones(len(indexes_all_repeated)))
    Time = time[0] + dt * np.arange(len(Flux))

    #prints in a file
    d = {'Flux': Flux, 'Time': Time, 'Variance': Variance}
    df_out = pd.DataFrame(data=d)
    df_out.to_csv(export_path, index=False)
    return Time, Flux, Variance


# usage of stellar_params
# properties = ['mass', 'radius', 'teff', 'logg', 'feh']
# for property in properties:
#     value, errp, errm = stellar_params[property][0], stellar_params[property+'_err_high'][0], stellar_params[property+'_err_low'][0]
#     print(property+ ' = {0} + {1} - {2}'.format(value, errp, -errm))


def logg_feh_Teff_lattice(logg0, feh0, Teff0):
    """for predicting limb darkening parameters, not used in this article"""
    # logg
    logg = round(logg0)
    if logg > 5:
        print('Warning: logg for this star is out of the computed range for limb darkening parameters')
        logg = 5
    elif logg < 0:
        print('Warning: logg for this star is out of the computed range for limb darkening parameters')
        logg = 0

    # Teff
    if Teff0 < 3250:
        print('Warning: Teff for this star is out of the computed range for limb darkening parameters')
        Teff = 3500
    elif Teff0 < 10250:
        Teff = round(Teff0 / 250.0) * 250
    elif Teff0 < 13500:
        Teff = round(Teff0 / 500.0) * 250
    else:
        Teff = 13000
        print(
            'Warning: Teff for this star is out of the prepared range for limb darkening parameters but the the range can easily be extended')

    # feh
    if feh0 < -5.5:
        feh = -5
        print('Warning: feh for this star is out of the computed range for limb darkening parameters')
    elif feh0 < -0.4:
        feh = round(feh0 / 0.5) * 0.5
    elif feh0 < 0.4:
        feh = round(feh0 / 0.1) * 0.1
    elif feh0 < 1.5:
        feh = round(feh0 / 0.5) * 0.5
    else:
        feh = 1
        print('Warning: feh for this star is out of the computed range for limb darkening parameters')

    return logg, feh, Teff


def u1u2_from_stellar_properties(df, stellar_parameters):
    """df = pandas_limb_darkening_table for shorter notation"""
    logg, feh, Teff = stellar_parameters['logg'][0], stellar_parameters['feh'][0], stellar_parameters['teff'][0]
    loggL, fehL, TeffL = logg_feh_Teff_lattice(logg, feh, Teff)

    df = df.loc[(df['Filt'] == 'Kp') & (df['Mod'] == 'ATLAS') & (df['Met'] == 'L') & (df['Teff'] == TeffL) & (df['logg'] == loggL) & (df['feh'] == fehL)]

    return (df[['u1', 'u2']]).to_numpy()[0]

    # to see ranges of logg, feh, Teff
    # df = df.loc[(df['Filt'] == 'Kp') & (df['Mod'] == 'ATLAS') ]
    # Teff1 = np.array(df['Teff'])
    # feh1 = np.array(df['feh'])
    # logg1 = np.array(df['logg'])
    # plt.figure(figsize = (15, 15))
    # plt.subplot(2, 2, 1)
    # plt.plot(Teff1, logg1, '.')
    # plt.subplot(2, 2, 2)
    # plt.plot(Teff1, feh1, '.')
    # plt.subplot(2, 1, 2)
    # plt.plot(logg1, feh1, '.')
    # plt.show()



if __name__ == "__main__":
    non_planets = np.loadtxt("C:/Users/USER/Documents/Physics/kepler/non_planets_candidates/non_planets_ids.txt", dtype=int)
    print(len(non_planets))
    data_stars = pd.read_csv("C:/Users/USER/Documents/Physics/kepler/download_scripts/namings.csv", index_col = False)
    for id in non_planets :
        params = (data_stars[data_stars['kepid'] == id]).values[0]
        T = params[3]
        phase = params[4]
        time_half_transit = (params[5] *0.5 /24.0)
        normalization_kepler(id, T, phase, time_half_transit*1.1)
        print(id)

    # id = eight_planet_id
    # kepler(id)
    #
    # normalization_spline_kepler(id)
    # name_datoteke_spline = 'data/kepler_short' + str(id) + '.txt'
    # name_export_spline = 'data/zeros_kepler_short' + str(id) + '.txt'
    # if long_cadence:
    #     name_datoteke_spline = 'data/spline_kepler' + str(id) + '.txt'
    #     name_export_spline = 'data/zeros_spline_kepler' + str(id) + '.txt'
    # zeros('data/just_spline_kepler' + str(id) + '.txt', 'data/zeros_just_spline_kepler' + str(id) + '.txt')
    # zeros('data/just_spline_kepler' + str(id) + '.txt', 'data/zeros_just_spline_kepler' + str(id) + '.txt')


        
