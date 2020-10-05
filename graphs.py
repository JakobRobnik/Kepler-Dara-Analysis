import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", palette="muted")

from noise_analysis import  noise
from celerite_comparisson import celerite_modeling
import fitting_planets_functions as func
from constants import *

### ploting graphs for the article. imports data as npy files or using pandas

file = 'zeros_kepler' +str(eight_planet_id)
df = pd.read_csv('data/'+file+'.txt')
Flux = np.array(df["Flux"])
Time = np.array(df["Time"])
Variance = np.array(df["Variance"])
#
# file = 'zeros_spline_kepler' +str(eight_planet_id)
# dfS = pd.read_csv('data/'+file+'.txt')
# FluxS = np.array(dfS["Flux"])
# TimeS = np.array(dfS["Time"])
# VarianceS = np.array(dfS["Variance"])
#spline_planets = np.ndarray.tolist(np.load('spline_planets.npy')) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




def hyperprior():
    plt.figure(figsize=(15, 10))
    plt.rcParams['xtick.labelsize']=15
    plt.rcParams['ytick.labelsize']=15
    kji = np.fft.rfftfreq(lenTime, d=dt)
    number_of_fourier_nodes = fourier_nodes(2.0)
    kji = kji[1:1 + number_of_fourier_nodes+1 // 2]
    bins = 20 + 80*np.log(1+kji/0.3)

    params_transitions = np.ndarray.tolist(np.load("params_planets_semi_kill.npy"))
    splines = func.prepare_splines(r)
    flux_transitions = func.flux_transitions(params_transitions, Time, splines, r) + out_of_phase_flux

    star_data = Flux - flux_transitions
    num = 25
    freq, power_data = noise.Power_spectrum(star_data, num)
    err_power_data = np.sqrt(2.0/num) * power_data
    
    Pk = np.load('Pk_semi_kill.npy')
    power_simulation = np.load('sim_semi_kill.npy')
    power_nodes = np.load('power_nodes_semi_kill.npy')
    err_Pk = np.sqrt(2.0/bins) * Pk
    freqsim, power_sim = noise.make_bins(kji, power_simulation, num)
    err_power_sim = np.sqrt(2.0 / num) * power_sim
    
    b, a = signal.butter(5, 0.1, 'low')
    
    #plt.plot(kji, power_nodes[1:len(Pk)+1], '.', label = r'$<P_{data}>$', color = 'skyblue')
    plt.plot(kji, signal.filtfilt(b, a, func.individual_band_smooth(kji, power_nodes)), label = 'star`s flux', color = 'goldenrod')
    plt.errorbar(freqsim, power_sim, yerr = err_power_sim, fmt = '.', capsize =2, label = 'simulation', color = 'forestgreen')
    plt.plot(kji, Pk + power_noise, label = 'hyperprior + noise', color = 'darkorange')

    bound_upper = Pk + err_Pk + power_noise
    bound_lower = Pk - err_Pk + power_noise
    #plotting the confidence intervals
    plt.fill_between(kji, bound_lower, bound_upper, color='darkorange', alpha=0.15)

    plt.errorbar(freq, power_data, yerr = err_power_data, fmt = '.', capsize = 2, label = 'signal - planets', color = 'darkred')
    #plt.plot(kji, Pk_corrected, label = 'bias corrected hyperprior', color = 'green')

    plt.plot([kji[0], kji[-1]], [power_noise, power_noise], color = 'black', label = 'noise')
    
    plt.yscale('log')
    plt.xlabel(r'$\nu [day^{-1}]$', fontsize = 17)
    plt.ylabel(r'$P(\nu)$', fontsize = 17)
    plt.grid(False)
    plt.legend(fontsize = 17)
    plt.xlim(0, 2)
    plt.savefig('hyperprior.pdf')
    plt.show()

def posterior():
    plt.rcParams['xtick.labelsize'] = 17
    plt.rcParams['ytick.labelsize'] = 17
    plt.figure(figsize=(20, 10))

    #planet
    plt.subplot(1, 2, 1)
    plt.title('hypothetical new planet', fontsize = 20)
    array = np.loadtxt('Kepler90j_posterior.txt', delimiter=',')
    p = np.flip(np.exp((array[:, 0] - array[len(array[0] // 2), 0]) / (-2.0)))
    amplitude = -np.flip(array[:, 1])
    mask = amplitude <= 0.0
    radius = func.amplitude_to_radius(amplitude[mask])
    p = p[mask]
    delta_radius = radius[1:] - radius[:-1]
    delta_radius = np.insert(delta_radius, [0, ], [delta_radius[0]])
    p /= np.sum(delta_radius * p)
    radius, p = func.eliminate_zigzag(radius, p, 10)
    plt.plot(radius, p)
    plt.plot([1.2, 1.2], [0, 1.9], ':', color = 'black')
    plt.ylim(0, 1.75)
    plt.xlim(0, 1.5)
    plt.text(1.05, 1.23, r'$\frac{p(1.2 R_{\oplus})}{p(0)} = 0.00012$', fontsize=18, rotation = 90)
    plt.xlabel(r'$r [R_{\oplus}]$', fontsize = 18)
    plt.ylabel('posterior probability', fontsize = 18, labelpad= 20)

    #moon
    plt.subplot(1, 2, 2)
    plt.title('hypothetical new moon', fontsize = 20)
    array = np.loadtxt('moon_posterior.txt', delimiter=',')
    p = np.flip(np.exp((array[:, 0] - array[len(array[0]//2), 0])/(-2.0)))
    amplitude = -np.flip(array[:, 1])
    mask = amplitude <= 0.0
    radius = func.amplitude_to_radius(amplitude[mask])
    p = p[mask]
    delta_radius = radius[1:] - radius[:-1]
    delta_radius = np.insert(delta_radius, [0, ], [delta_radius[0]])
    p /= np.sum(delta_radius*p)
    mask_divide = radius < 0.45
    mask_one = p[mask_divide] > 1.25
    radius1 = radius[mask_divide][mask_one]
    p1 = p[mask_divide][mask_one]
    # p[1:] = func.constant_band_smooth(p[1:], 3)
    #radius1, p1 = func.eliminate_zigzag(radius[mask_divide], p[mask_divide ], 0)
    radius2, p2 = func.eliminate_zigzag(radius[~mask_divide], p[~mask_divide], 28)
    radius = np.concatenate((radius1, radius2))
    p = np.concatenate((p1, p2))
    radius, p = func.eliminate_zigzag(radius, p, 5)
    plt.plot(radius, p, color =  'darkorange')
    plt.text(0.95, 1.23, r'$\frac{p(0.9 R_{\oplus})}{p(0)} = 0.17$', fontsize = 18, rotation = 90)
    plt.plot([0.9, 0.9], [0, 1.9], ':', color = 'black')
    plt.ylim(0, 1.75)
    plt.xlim(0, 1.5)
    plt.xlabel(r'$r [R_{\oplus}]$', fontsize = 18)
    #plt.ylabel('posterior probability', fontsize = 18)
    plt.savefig('posterior_planet_moon')
    plt.show()


def spline_vs_joint():
    r = [0.006992828328479127, 0.0062988835325231834, 0.007046208697398815, 0.015373546248870142, 0.014252558501556694, 0.01542692661778983]
    r_g, r_h = 0.04339823993170634, 0.06042657761708681
    spline_g = func.prepare_shape(r_g)
    spline_h = func.prepare_shape(r_h)
    out_of_phase_flux = func.planet_out_of_phase(Time, Flux, Variance, np.load('out_of_phases_semi_kill.npy'), [spline_g, spline_h], [r_g, r_h], ['TTV', 'TTV'], direct_return=True)

    plt.rcParams['xtick.labelsize'] = 17
    plt.rcParams['ytick.labelsize'] = 17
    plt.figure(figsize=(15, 10))

    #parameters
    params_transitions = np.ndarray.tolist(np.load("params_planets_semi_kill.npy"))
    params_transitions_spline = [7.0081855522622565, 6.1622038831690391, 0.077296118959435806, -0.53175663061265244, 8.7197589925145262, 8.0100179478329618, 0.082184166529983199, -0.72167288621304249, 14.449008660338034, 0.75443806105365352, 0.088868503560561696, -0.23500786726638878, 59.736791476312689, 27.445741864635057, 0.1619122110404517, -2.49383787750737, 91.939973838316732, 2.7849264915101295, 0.18274167147732515, -2.2302058125525033, 124.91050198235823, 123.21742271463862, 0.21459795439089063, -2.6283054452949655]
    nodes = np.load("params_nodes_semi_kill.npy")
    splines = func.prepare_splines(r)
    Pk = np.load('Pk_semi_kill.npy')

    #fluxes
    planets_joint = func.flux_transitions(params_transitions, Time, splines) + out_of_phase_flux
    np.save('search_flux2.npy', (Flux - planets_joint)/Variance)
    planets_spline = func.flux_transitions(params_transitions_spline, Time, splines)
    flux_nodes = func.flux_nodes(nodes, lenTime)
    flux_spline = np.array(pd.read_csv('data/zeros_just_spline_kepler' + str(eight_planet_id) + '.txt')["Flux"])
    flux_celerite, planets_celerite, power_celerite_analytic = celerite_modeling.celerite_results(Time, Flux, splines)
    print(noise.autocorr(flux_celerite)[0])
    #powers
    bin_width = 50
    freq_planets, power_planets, err_planets = noise.Power_spectrum(planets_joint, bin_width, errors = True)[:len(Pk)]
    freq_data, power_data, err_data = noise.Power_spectrum(Flux, bin_width, errors = True)[:len(Pk)]
    freq_spline, power_spline, err_spline = noise.Power_spectrum(flux_spline, bin_width, errors = True)[:len(Pk)]
    freq_celerite, power_celerite, err_celerite = noise.Power_spectrum(flux_celerite, bin_width, errors=True)[:len(Pk)]
    freq_joint, power_joint, err_joint = noise.Power_spectrum(flux_nodes, bin_width, errors = True)[:len(Pk)]

    ########ploting#########
    # fourier domain
    plt.subplot(2, 1, 1)

    plt.plot([0, 2], [len(Time), len(Time)], ':', color = 'brown', label = 'noise')
    plt.errorbar(freq_planets, power_planets, yerr = err_planets, capsize = 2, fmt = '.', color = 'green', label = 'planets')
    plt.errorbar(freq_data, power_data, yerr = err_data, capsize = 2, fmt = '.', color = 'black', label = 'data')
    plt.errorbar(freq_spline, power_spline, yerr=err_spline, capsize=2, fmt='.', color = 'red', label='star variation (spline)')
    plt.plot(np.linspace(0, 2, len(power_celerite_analytic)), power_celerite_analytic, color= 'darkorange', label='star variation (celerite)')
    #plt.errorbar(freq_celerite, power_celerite, yerr=err_celerite, capsize=2, fmt='.', color= 'darkorange', label='star variation (celerite)')
    plt.errorbar(freq_joint, power_joint, yerr=err_joint, capsize=2, fmt='.', color = 'blue', label='star variation (Fourier GP)')
    plt.yscale('log')
    plt.xlim(0, 2)
    plt.ylim(1e2, 1e7)
    plt.xlabel(r'$\nu$ [1/days]', fontsize = 17)
    plt.ylabel(r'$P(\nu)$', fontsize = 17)
    plt.legend(loc = 3, fontsize = 14, ncol = 2)

    # time domain
    plt.subplot(2, 1, 2)

    mask_measured = Variance<2.0
    plt.plot(Time[mask_measured], Flux[mask_measured], '.', color = 'black', label = 'data')
    plt.plot(Time[mask_measured], flux_spline[mask_measured]+planets_spline[mask_measured], color = 'red', label = 'spline')
    plt.plot(Time[mask_measured], (flux_celerite+planets_celerite)[mask_measured], color='darkorange', label='celerite')
    plt.plot(Time[mask_measured], flux_nodes[mask_measured]+planets_joint[mask_measured], color = 'blue', label = 'Fourier GP')

    # mark smaller planets (in phase)
    letters = ['b', 'c', 'i', 'd', 'e', 'f']
    for planet in range(6):
        tji = [params_transitions[4 * planet] * k + params_transitions[4 * planet + 1] for k in range(int(Time[-1] / params_transitions[4 * planet]))]
        for t in tji:
            plt.arrow(t, 2.5, 0, -0.7, facecolor='black', color='black', shape='full', head_width=0.07)
            plt.text(t-dt, 3, letters[planet], fontsize=15)

    plt.xlim(80, 88)
    plt.ylim(-5, 5)
    plt.ylabel(r'flux [$\sigma$]', fontsize = 17)
    plt.xlabel('t [days]', fontsize = 17)
    plt.legend(loc = 2, fontsize = 14)
    plt.savefig('comparisson.pdf')
    plt.show()

def fourier_nodes(max):
    return int(max * dt * lenTime) + 1



def amplitude_comparisson():
    plt.rcParams['xtick.labelsize'] = 17
    plt.rcParams['ytick.labelsize'] = 17
    plt.figure(figsize=(20, 10))
    ymin, ymax = 0.63, 1.18
    #gs = gridspec.GridSpec(2, 2)
    periods = {'b': 7, 'c': 8.7, 'i': 14.5, 'd': 59.7, 'e': 91.9, 'f': 124.9, 'g': 210.6, 'h': 331.6}

    plt.subplot(1, 3, 1)
    planet_name = 'c'
    plt.title('Kepler 90'+planet_name + ' (period = {0} days)'.format(periods[planet_name]), fontsize=20)
    As = np.loadtxt('amplitude_tests/amplitude_test_planet_' + planet_name + '_spline_joint.txt')
    Aj = np.loadtxt('amplitude_tests/amplitude_test_planet_' + planet_name + '_joint.txt')
    SNR = As[:, 4]
    Am = np.loadtxt('amplitude_tests/SNRtest_'+planet_name+'.txt')
    Ac = np.loadtxt('amplitude_tests/amplitude_test_planet_'+planet_name+'_celerite.txt')

    plt.errorbar(Am[:, 2], Am[:, 0], Am[:, 1], capsize=2, fmt='.', color='green', label='matched filter')
    plt.errorbar(SNR, Aj[:, 0], Aj[:, 1], capsize=2, fmt='.', color='blue', label='Fourier GP')
    plt.errorbar(Ac[:, 2], Ac[:, 0], Ac[:, 1], capsize=2, fmt='.', color='darkorange', label='celerite')
    plt.errorbar(SNR, As[:, 0], As[:, 1], capsize=2, fmt='.', color='red', label='spline')
    plt.plot([SNR[0], SNR[-1]], [1, 1], ':', color='black')
    plt.xlabel(r'$SNR_0$', fontsize=18)
    plt.ylabel(r'$SNR/SNR_0$', fontsize=18)
    plt.ylim(ymin, ymax)
    plt.legend(loc=4, fontsize=18)

    plt.subplot(1, 3, 2)
    planet_name = 'i'
    plt.title('Kepler 90' + planet_name + ' (period = {0} days)'.format(periods[planet_name]), fontsize=20)
    As = np.loadtxt('amplitude_tests/amplitude_test_planet_' + planet_name + '_spline_joint.txt')
    Aj = np.loadtxt('amplitude_tests/amplitude_test_planet_' + planet_name + '_joint.txt')
    SNR = As[:, 4]
    Am = np.loadtxt('amplitude_tests/SNRtest_' + planet_name + '.txt')
    Ac = np.loadtxt('amplitude_tests/amplitude_test_planet_' + planet_name + '_celerite.txt')

    plt.errorbar(Am[:, 2], Am[:, 0], Am[:, 1], capsize=2, fmt='.', color='green', label='matched filter')
    plt.errorbar(SNR, Aj[:, 0], Aj[:, 1], capsize=2, fmt='.', color='blue', label='Fourier GP')
    plt.errorbar(Ac[:, 2], Ac[:, 0], Ac[:, 1], capsize=2, fmt='.', color='darkorange', label='celerite')
    plt.errorbar(SNR, As[:, 0], As[:, 1], capsize=2, fmt='.', color='red', label='spline')
    plt.plot([SNR[0], SNR[-1]], [1, 1], ':', color='black')
    plt.xlabel(r'$SNR_0$', fontsize=18)
    #plt.ylabel(r'$A/A_0$', fontsize=18)
    plt.ylim(ymin, ymax)
    plt.legend(loc=4, fontsize=18)

    plt.subplot(1, 3, 3)
    planet_name = 'g'
    plt.title('Kepler 90' + planet_name + ' (period = {0} days)'.format(periods[planet_name]), fontsize=20)
    As = np.loadtxt('amplitude_tests/amplitude_test_planet_' + planet_name + '_spline_joint.txt')
    Aj = np.loadtxt('amplitude_tests/amplitude_test_planet_' + planet_name + '_joint.txt')
    SNR = As[:, 4]
    Am = np.loadtxt('amplitude_tests/SNRtest_' + planet_name + '.txt')
    Ac = np.loadtxt('amplitude_tests/amplitude_test_planet_' + planet_name + '_celerite.txt')

    plt.errorbar(Am[:, 2], Am[:, 0], Am[:, 1], capsize=2, fmt='.', color='green', label='matched filter')
    plt.errorbar(SNR, Aj[:, 0], Aj[:, 1], capsize=2, fmt='.', color='blue', label='Fourier GP')
    plt.errorbar(Ac[:, 2], Ac[:, 0], Ac[:, 1], capsize=2, fmt='.', color='darkorange', label='celerite')
    plt.errorbar(SNR, As[:, 0], As[:, 1], capsize=2, fmt='.', color='red', label='spline')
    plt.plot([SNR[0], SNR[-1]], [1, 1], ':', color='black')
    plt.ylim(ymin, ymax)
    plt.xlabel(r'$SNR_0$', fontsize=18)
    #plt.ylabel(r'$A/A_0$', fontsize=18)
    plt.legend(loc = 4, fontsize=18)
    plt.savefig('../matched_filter/amplitude_test_matched.pdf')
    plt.show()


#spline_vs_joint()
#hyperprior()
amplitude_comparisson()
