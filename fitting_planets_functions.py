import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
from scipy import signal
from scipy import interpolate
from scipy import optimize
from scipy import stats
from datetime import datetime

#my imports
from constants import *
from noise_analysis import noise
from matched_filter import transit_search as ts
from joint_fit_and_preprocessing import prepare_data
import emcee


if not long_cadence:
    file = 'kepler_short' + str(eight_planet_id)
    Identificator = np.array(pd.read_csv('C:/Users/USER/Documents/Physics/machine/joint_fit_and_preprocessing/data/'+file+'.txt')["Identificator"])

def find_index(time, array):
    """returns first index i at which time Time[i] > time.
    Time must be an increasing sequence"""
    i_min = 0
    i_max= len(array)-1
    while i_max - i_min > 1:
        i = (i_min + i_max)//2
        if array[i] > time:
            i_max = i
        else:
            i_min = i
    if array[i] > time:
        return i
    else:
        return i+1

def eliminate_zigzag(x_given, y_given, num_remove):
    x = np.copy(x_given)
    y = np.copy(y_given)
    for i in range(num_remove):
        neighbours1 = np.sqrt(np.square(x[1:]-x[:-1]) + np.square(y[1:]-y[:-1])) #neighbour1[i] = distance from point i+1 to point i+2
        neighbours2 = np.sqrt(np.square(x[2:] - x[:-2]) + np.square(y[2:] - y[:-2])) #neighbour1[i] = distance from point i+1 to point i+3
        delta = neighbours1[:-1] + neighbours1[1:] - neighbours2 #lenght of zigzag - lenght of zigzag if (i+1)th point is removed
        remove_index = np.argmax(delta)
        x = np.delete(x, [remove_index+1, ])
        y = np.delete(y, [remove_index+1, ])
    return (x, y)


def check_gradient(FgradF, test_point, *args):
    """FgradF returns (F, gradF). check_gradient checks if these two match, that is, if gradF is indeed gradient of F on a test_point
    *args are extra parameters to pass to the function FgradF"""
    f, grad = FgradF(test_point, *args)
    index = np.arange(len(grad))
    def ff(x):
        return FgradF(x, *args)[0]
    approx_grad = optimize.approx_fprime(test_point, ff, 1e-6)
    #plt.plot(index, approx_grad, 'o', label = 'numerical', color = 'red')
    plt.plot(index, (grad/approx_grad), 'o', color = 'blue')
    plt.title('Gradient test: gradient is correct if given/numerical is close to 1 for all arguments')
    plt.ylabel('given/numerically computed')
    plt.show()

def amplitude_to_radius(A, u1, u2, sigma_in_star_flux_units, R_star):
    return np.sqrt((1-(u1/3.0)-(u2/6.0))*(-A)*sigma_in_star_flux_units) * R_star

def err_amplitude_to_err_radius(A, err_A, u1, u2, sigma_in_star_flux_units, R_star):
    """converts gaussian error on amplitude of a planet to an error on radius of a planet"""
    return err_A * np.sqrt((1-(u1/3.0)-(u2/6.0))*sigma_in_star_flux_units/(-A)) * R_star * 0.5


def intensity(relative_distance, u1, u2):
    """relative tangential distance is in units of radius of star, normed so that depth of transit is 1""" 
    mi = np.sqrt(1 - np.square(relative_distance)) # cosinus of angle between observer, center of a star and a point on a surface
    return (1 - u1 * (1 - mi) - u2 * np.square(1 - mi)) #1 at the centre of transit (relative distance = 0)

def limb_darkening(relative_time, r, u1, u2):
    """relative_time = abs(time-phase)/time_half_transit
       r = R_planet/R_star"""
    if relative_time > (1+r):
        return 0.0
    elif relative_time > (1-r):
        return integrate.quad( lambda d: 2 * math.acos((-r**2 + relative_time**2 + d**2) / (2 * relative_time * d)) * d * intensity(d, u1, u2), relative_time - r, 1)[0]/(math.pi*r**2)
    elif relative_time > 0.5:
        return integrate.quad(lambda d: 2 * math.acos((-r**2 + relative_time**2 + d**2) / (2 * relative_time * d)) * d * intensity(d, u1, u2), relative_time - r, relative_time + r)[0]/(math.pi*r**2) #ker je A ze minimum, upostevajoc polno ploscino
    else: #it is useless to compute integral as a difference is 1.5 promile
        return intensity(relative_time, u1, u2)

def integrated_limb_darkening(t, delta_t, r):
    intervals = np.linspace(t-delta_t/2, t+delta_t/2, 10)
    return integrate.simps([limb_darkening(x, r) for x in intervals], intervals)/delta_t

def prepare_shape(r, u1 = u1_Kepler90, u2 = u2_Kepler90):
    """"fits profile with spline before fitting so that profile is computed faster when fitting."""
    relative_times = np.linspace((-1-r)*1.1, (1+r)*1.1, 500)
    fluxes = [limb_darkening(abs(t), r, u1, u2) for t in relative_times]
    spline = interpolate.splrep(relative_times, fluxes)
    return spline



def prepare_splines(r):
    splines = []
    for planet_number in range(len(r)):
        splines.append(prepare_shape(r[planet_number]))
    return splines

def planet_fit_loglikelihood(x0_planet, Time, Flux, spline, r, distribution_parameters, Variance):
    """optimizes parameters of transition (x0_planet is an initial guess) to find minimal -2 log likelihood"""
    deltaP, deltaFi, deltaTau = dt, 3 * dt, 2 * dt  # 0.01, 0.04, dt
    bound = [(x0_planet[0] - deltaP, x0_planet[1] + deltaP), (x0_planet[1] - deltaFi, x0_planet[1] + deltaFi),
             (x0_planet[2] - deltaTau, x0_planet[2] + deltaTau), (-np.inf, 0.0)]

    def minimize(parameters_transition):
        return negative_2loglikelihood(Flux - profile_with_spline(Time, *parameters_transition, spline, r), distribution_parameters, Variance)

    #borders = np.concatenate((np.expand_dims(x0_planet*0.6, 1), np.expand_dims(x0_planet*1.4, 1)), axis = 1)
    params = optimize.fmin_l_bfgs_b(minimize, x0=x0_planet, bounds=bound, approx_grad=True, epsilon=1e-7)[0]
    return params

def profile_with_spline(T, period, phase_given, time_half_transit, amplitude, spline):
    """T are times at which we want to get profile it must be a collection of equally spaced time intervals starting with zero: [0, dt, 2dt, ...].
     Profile is calculated with use of precomputed 'spline'."""
    F = np.zeros(len(T)) #flux impact of a planet
    if period < 0.0:
        print('Invalid period in profile_with_spline')
        return F
    elif time_half_transit < 0.0:
        print('Invalid time_half_transit in profile_with_spline')
        return F
    phase = np.remainder(phase_given, period)
    number_of_transits = 1+ int((T[-1]-phase-time_half_transit)/period)

    for k in range((1 if phase < time_half_transit else 0), number_of_transits): #1 because we want to avoid having only part of the transit
        for i in range(int((k*period+phase-time_half_transit-T[0])/dt)-1, int((k*period+phase+time_half_transit-T[0])/dt)+2): #+1 because we want to include index after the end of the transit and +1 because of the indexing in python
            F[i] = amplitude * interpolate.splint((T[i]-k*period-phase-dt*0.5)/time_half_transit, (T[i]-k*period-phase+dt*0.5)/time_half_transit, spline)*time_half_transit/dt
    return F



def Jacobian_Hessian_profile_with_spline(T, period, phase_given, time_half_transit, spline):
    """T are times at which we want to get profile it must be a collection of equally spaced time intervals starting with zero: [0, dt, 2dt, ...].
     Profile is calculated with use of precomputed 'spline'."""
    J = np.zeros((3, len(T)))
    H = np.zeros((6, len(T))) #folded like: H11, H12, H13, H22, H23, H33
    if period < 0.0:
        print('Invalid period in Jacobian_profile_with_spline')
        return J, H
    elif time_half_transit < 0.0:
        print('Invalid time_half_transit in Jacobian_profile_with_spline')
        return J, H
    phase = np.remainder(phase_given, period)
    number_of_transits = int((T[-1])/period)
    if phase+time_half_transit < math.fmod(T[-1], period): #in case remainder contains a full transit
        number_of_transits += 1
    for k in range((1 if phase < time_half_transit else 0), number_of_transits): #1 because we want to avoid having only part of the transit
        for i in range(int((k*period+phase-time_half_transit-T[0])/dt)-1, int((k*period+phase+time_half_transit-T[0])/dt)+2): #+1 because we want to include index after the end of the transit and +1 because of the indexing in python
            tplus, tminus = (T[i] - k * period - phase + dt * 0.5) / time_half_transit, (T[i]-k*period-phase-dt*0.5)/time_half_transit
            Uint = -interpolate.splint(tminus, tplus, spline)
            Uplus, Uminus = -interpolate.splev(tplus, spline), -interpolate.splev(tminus, spline)
            DUplus, DUminus = -interpolate.splev(tplus, spline, der = 1), -interpolate.splev(tminus, spline, der = 1)
            J[1, i] = -(Uplus - Uminus)/dt
            J[0, i] = J[1, i]*k
            J[2, i] = (-(tplus*Uplus-tminus*Uminus) + Uint) / dt
            H[3, i] = (DUplus-DUminus)/(time_half_transit*dt) #H22
            H[0, i] = H[3, i] * k**2 #H11
            H[1, i] = H[3, i] * k #H12
            H[2, i] = (tplus*DUplus-tminus*DUminus)/(time_half_transit*dt) #H13
            H[4, i] = H[2, i] * k #H23
            H[5, i] = (tplus*tplus*DUplus-tminus*tminus*DUminus)/(time_half_transit*dt) #H33

    return J, H



def ttv_profile_with_spline(T, phase_arr, time_half_transit_arr, amplitude_arr, spline):
    F = np.zeros(len(T)) #flux impact of a planet
    for num_event in range(len(phase_arr)):
        phase, time_half_transit, amplitude = phase_arr[num_event], time_half_transit_arr[num_event], amplitude_arr[num_event]
        integer_time_half_transit = int(time_half_transit / dt)
        index_min, index_max = int((phase - time_half_transit)/dt) -2, int((phase + time_half_transit)/dt) +3
        for i in range(index_min, index_max):
            F[i] = amplitude * interpolate.splint((T[i]-phase-dt*0.5)/time_half_transit, (T[i]-phase+dt*0.5)/time_half_transit, spline)*time_half_transit/dt
    return F


#normal
def gradient_and_profile_with_spline(T, F, period, phase_given, time_half_transit, amplitude, spline):
    """a variant of profile wtih spline it also returns gradient"""
    if period < 0.0:
        print(("Invalid period in gradient_and_profile_with_spline"))
        raise ValueError
    elif time_half_transit < 0.0:
        print(('Invalid time_half_transit in gradient_and_profile_with_spline'))
        raise ValueError
    phase = np.remainder(phase_given, period)

    number_of_transits = int((T[-1]-T[0])/period)
    gradF = np.zeros(shape=(4, len(T)))
    if phase+time_half_transit < math.fmod(T[-1], period): #in case remainder contains a full transit
        number_of_transits += 1

    for k in range((1 if phase < time_half_transit else 0), number_of_transits): #1 because we do not want a transit over the edge
        for i in range(int((k*period+phase-time_half_transit-T[0])/dt), int((k*period+phase+time_half_transit-T[0])/dt)+2): #+1 for the for loop and +1 to include a point after the transit
            t_minus = (T[i]-k*period-phase-dt*0.5)/time_half_transit
            t_plus = (T[i]-k*period-phase+dt*0.5)/time_half_transit
            osnovna_vrednost = interpolate.splint(t_minus, t_plus, spline) * time_half_transit / dt
            base_value_of_derivative = (-amplitude/dt) * (interpolate.splev(t_plus, spline) - interpolate.splev(t_minus, spline))
            F[i] += amplitude * osnovna_vrednost
            gradF[1, i] = base_value_of_derivative
            gradF[0, i] = k* base_value_of_derivative
            gradF[2, i] = (F[i]/time_half_transit) + (-amplitude / dt)*(interpolate.splev(t_plus, spline)*t_plus - interpolate.splev(t_minus, spline) * t_minus)
            gradF[3, i] = osnovna_vrednost

    return gradF

def gradient_logp_and_logp(parameters, Time, flux, Variance, spline, distribution_parametrs):
    F = np.zeros(len(Time))
    point_planet_grad = gradient_and_profile_with_spline(Time, F, *parameters, spline)
    logp, point_pdf_grad = noise.derivative_logp(flux- F, Variance, *distribution_parametrs)
    return logp, np.array([-np.dot(point_pdf_grad, point_planet_grad[i, :]) for i in range(len(parameters))])

#TTV
def gradient_TTV(T, F, phases, time_half_transit, amplitude, spline):
    """like gradient_and_profile_with_spline but also accounting for TTV of a planet"""
    if time_half_transit < 0.0:
        raise Exception('Invalid time_half_transit in gradient_and_profile_with_spline')

    gradF = np.zeros(shape=(len(phases) + 2, len(T)))

    for phase_num in range(len(phases)):
        for i in range(int((phases[phase_num]-time_half_transit)/dt), int((phases[phase_num]+time_half_transit)/dt)+2): #+1 for the for loop and +1 to include a point after the transit
            t_minus = (T[i]-phases[phase_num]-dt*0.5)/time_half_transit
            t_plus = (T[i]-phases[phase_num]+dt*0.5)/time_half_transit
            osnovna_vrednost = interpolate.splint(t_minus, t_plus, spline) * time_half_transit / dt
            base_value_of_derivative = (-amplitude/dt) * (interpolate.splev(t_plus, spline) - interpolate.splev(t_minus, spline))
            F[i] += amplitude * osnovna_vrednost
            gradF[phase_num, i] = base_value_of_derivative
            gradF[-2, i] = (F[i]/time_half_transit) + (-amplitude / dt)*(interpolate.splev(t_plus, spline)*t_plus - interpolate.splev(t_minus, spline) * t_minus)
            gradF[-1, i] = osnovna_vrednost

    return gradF

def gradient_logp_and_logp_TTV(parameters, Time, flux, Variance, spline, r, distribution_parametrs):
    F = np.zeros(len(Time))
    point_planet_grad = gradient_TTV(Time, F, parameters[:-2], parameters[-2], parameters[-1], spline)
    logp, point_pdf_grad = noise.funciton_derivative_logp(flux- F, Variance, *distribution_parametrs)

    return logp, np.array([-np.dot(point_pdf_grad, point_planet_grad[i, :]) for i in range(len(parameters))])

#moon
def gradient_moon(T, F, phases, time_half_transit, amplitude_planet, amplitude_moon, spline_planet, spline_moon, relative_moon_phases):
    """like gradient_TTV but also accounting for a planet`s moon"""
    if time_half_transit < 0.0:
        raise Exception('Invalid time_half_transit in gradient_and_profile_with_spline')

    gradF = np.zeros(shape=(len(phases) + 3, len(T)))

    for phase_num in range(len(phases)):
        #planet contribution
        for i in range(int((phases[phase_num]-time_half_transit)/dt), int((phases[phase_num]+time_half_transit)/dt)+2): #+1 for the for loop and +1 to include a point after the transit
            t_minus = (T[i]-phases[phase_num]-dt*0.5)/time_half_transit
            t_plus = (T[i]-phases[phase_num]+dt*0.5)/time_half_transit
            osnovna_vrednost = interpolate.splint(t_minus, t_plus, spline_planet) * time_half_transit / dt
            base_value_of_derivative = (-amplitude_planet/dt) * (interpolate.splev(t_plus, spline_planet) - interpolate.splev(t_minus, spline_planet))
            F[i] += amplitude_planet * osnovna_vrednost
            gradF[phase_num, i] += base_value_of_derivative
            gradF[-3, i] += (amplitude_planet * osnovna_vrednost/time_half_transit) + (-amplitude_planet / dt)*(interpolate.splev(t_plus, spline_planet)*t_plus - interpolate.splev(t_minus, spline_planet) * t_minus)
            gradF[-2, i] += osnovna_vrednost
        #moon_contribution
        for i in range(int((phases[phase_num]+relative_moon_phases[phase_num]-time_half_transit)/dt), int((phases[phase_num]+relative_moon_phases[phase_num]+time_half_transit)/dt)+2): #+1 for the for loop and +1 to include a point after the transit
            t_minus = (T[i]-phases[phase_num]+relative_moon_phases[phase_num]-dt*0.5)/time_half_transit
            t_plus = (T[i]-phases[phase_num]+relative_moon_phases[phase_num]+dt*0.5)/time_half_transit
            osnovna_vrednost = interpolate.splint(t_minus, t_plus, spline_moon) * time_half_transit / dt
            base_value_of_derivative = (-amplitude_moon/dt) * (interpolate.splev(t_plus, spline_moon) - interpolate.splev(t_minus, spline_moon))
            F[i] += amplitude_moon * osnovna_vrednost
            gradF[phase_num, i] += base_value_of_derivative
            gradF[-3, i] += (amplitude_moon * osnovna_vrednost/time_half_transit) + (-amplitude_moon / dt)*(interpolate.splev(t_plus, spline_moon)*t_plus - interpolate.splev(t_minus, spline_moon) * t_minus)
            gradF[-1, i] += osnovna_vrednost

    return gradF

def gradient_logp_and_logp_moon(parameters, Time, flux, Variance, spline_planet, spline_moon, relative_moon_phases, distribution_parametrs):
    F = np.zeros(len(Time))
    point_planet_grad = gradient_moon(Time, F, parameters[:-3], parameters[-3], parameters[-2], parameters[-1], spline_planet, spline_moon, relative_moon_phases)
    logp, point_pdf_grad = noise.funciton_derivative_logp(flux- F, Variance, *distribution_parametrs)

    return logp, np.array([-np.dot(point_pdf_grad, point_planet_grad[i, :]) for i in range(len(parameters))])


def planet_fit(x0_planet, Time, Flux, Variance, spline, covariance = False):
    #parameters: period, phase, time_half_transit, amplitude
    def f(time, *parameters):
        return profile_with_spline(time, *parameters, spline)

    def chi2(parameters):
        return np.sum(np.square((Flux - profile_with_spline(Time, *parameters, spline))/Variance))

    # def f_just_amplitude(time, *A):
    #     return profile_with_spline(time, *(x0_planet[:-2]), *A, spline, r)
    #def f_no_tau(time, P, phi, A):
    #    return profile_with_spline(time, P, phi, x0_planet[2], A, spline)
    #params1, cov = optimize.curve_fit(f_no_tau, Time, Flux, [x0_planet[0], x0_planet[1], x0_planet[3]], sigma=np.sqrt(Variance))#, bounds = ((dt, -np.inf), (4*dt,0.0), (-np.inf, 0.0)))
    #params = [params1[0], params1[1], x0_planet[2], params1[2]]
    #cov_ret = [cov[0, 0], cov[0, 1], cov[0, 2], cov[1, 1], cov[1, 2], cov[2, 2]]
    delta = np.array([dt, 3*dt, 2*dt, 5])
    #bounds = (np.array(x0_planet) - delta, np.array(x0_planet) + delta)
    # bounds[0][-1] = -np.inf
    # bounds[1][-1] = 0.0
    #params, cov = optimize.curve_fit(f, Time, Flux, x0_planet, sigma = np.sqrt(Variance), bounds= bounds)

    bound = [(x0_planet[i] - delta[i], x0_planet[i]+delta[i]) for i in range(4)]
    bound[3] = (-np.inf, 0.0)
    opt = optimize.fmin_l_bfgs_b(chi2, x0=x0_planet, bounds=bound, approx_grad=True,
                                 epsilon=1e-7)
    params = opt[0]
    if covariance:
        return params, None#cov
    else:
        return params

def planet_fit_brez_perioda(x0_planet, Time, Flux, Variance, spline, r):
    """a single transit fit"""
    def model_transition_with_spline(time, *parameters):
        return profile_with_spline(time, x0_planet[0], *parameters, spline)
    params, cov = optimize.curve_fit(model_transition_with_spline, Time, Flux, x0_planet[1:], sigma = np.sqrt(Variance))
    return params#, np.sqrt(np.diag(cov))

#following few functions are for fitting exomoon with known transit times, for cooperation with Yan Liang.

def flux_planet_moon(Time, parameters, spline_planet, spline_moon, r_planet, r_moon, relative_moon_phases):
    flux = np.zeros(len(Time))
    for i in range(len(relative_moon_phases)):
        flux += profile_with_spline(Time, Time[-1], parameters[i], parameters[-3], parameters[-2], spline_planet)
    for i in range(len(relative_moon_phases)):
        flux += profile_with_spline(Time, Time[-1], parameters[i] + relative_moon_phases[i], parameters[-3], parameters[-1], spline_moon)
    return flux

def planet_moon(Time, Flux, Variance, relative_moon_phases, amplitude):
    """kepler 90g and its moon
    usage: 1. you must be given Kepler 90g (planet) phases and moon phases. Planet phases do not need to be correct.
    relative_moon_phases = planet_phases - moon_phases (in the exact case of Kepler 90g erase 4th transit (around 700 days), because she gives you a transit where there are no measurements).
           2. change x0_planet, initial guess on fitted parameters, of the shape: [phases, planet_half_time_transit, amplitude_planet, amplitude_moon]
    """
    params0 = np.ndarray.tolist(np.load('out_of_phases_semi_kill.npy')[0]) + [amplitude, ]
    if not long_cadence:
        x0_planet = params0[3:]
    r_moon = 0.0009
    spline_moon = prepare_shape(r_moon)
    r_planet = 0.04339823993170634
    spline_planet = prepare_shape(r_planet)
    
    #parameters: [phase_planet_1, phase_planet_2, ... phase_planet_n, time_half_transit_planet, amplitude_planet, amplitude_moon]

    def f(time, *parameters):
        return flux_planet_moon(time, parameters, spline_planet, spline_moon, r_planet, r_moon, relative_moon_phases)
    # if logp_planets:
    #     borders = [(params0[phase_num] - 0.1, params0[phase_num] + 0.1) for phase_num in range(len(params0) - 3)] + \
    #               [(params0[-3] - 0.1, params0[-3] + 0.1), (-np.inf, np.inf), (-np.inf, np.inf)]
    #     params = optimize.fmin_l_bfgs_b(gradient_logp_and_logp_moon, x0=np.array(params0), bounds=borders,
    #                                      args=(Time, Flux, Variance, spline_planet, spline_moon, relative_moon_phases, distribution_parameters))[0]
    #     params = np.ndarray.tolist(params)
    # else:
    #     params, cov = optimize.curve_fit(f, Time, Flux, params0, sigma = np.sqrt(Variance))
    #
    #print("Moon amplitude = {0} +- {1}".format(params[-1], np.sqrt(cov[-1, -1])))
    return f(Time, *params0), params0

def plot_moon(Time, flux_no_nodes, Variance, params_transitions, parameters, splines, r, relative_moon_phases):
    """parameters: [phase_planet_1, phase_planet_2, ... phase_planet_n, time_half_transit_planet, amplitude_planet, amplitude_moon]"""
    r_moon = 0.0009
    spline_moon = prepare_shape(r_moon)
    r_planet = 0.04339823993170634
    spline_planet = prepare_shape(r_planet)

    flux_planet_with_moon = flux_planet_moon(Time, [*parameters[:-1], 0.0], spline_planet, spline_moon, r_planet, r_moon, relative_moon_phases)
    flux_moon = flux_planet_moon(Time, [*parameters[:-2], 0.0, parameters[-1]], spline_planet,spline_moon, r_planet, r_moon, relative_moon_phases)
    #return flux_moon
    flux_in_phase_planets = flux_transitions(params_transitions, Time, splines)
    flux_no_nodes_no_planets = flux_no_nodes - flux_planet_with_moon - flux_in_phase_planets #flux where planets and stars are eliminated, but not out of phase planets for simplicity.
    mask_moon_transit = flux_moon < -1e-14
    time, flux, err_flux = ts.fold_phases(Time, flux_no_nodes_no_planets, Variance, parameters[:-3] + relative_moon_phases, 3*dt, time_window = 2)
    time, moon_flux, err_model = ts.fold_phases(Time, flux_moon, Variance, parameters[:-3] + relative_moon_phases, 3*dt, time_window = 2)
    # plt.figure(figsize = (15, 10))
    # plt.errorbar(time, flux*0.000198663809786, yerr = err_flux*0.000198663809786, fmt = 'o', capsize = 2, label = 'flux - star - planets')
    # plt.plot(time, moon_flux*0.000198663809786, label = 'moon')
    # plt.legend()
    # plt.xlabel('time[days]')
    # plt.ylabel('flux[star`s flux]')
    # #plt.savefig('moon_long')
    # plt.close()

    plt.figure(figsize=(15, 10))
    relative_amplitude_array = np.linspace(-1, 1, 1000)
    log0 = np.sum(np.log(noise.pdf_noise((flux_no_nodes_no_planets)[mask_moon_transit], *distribution_parameters)))
    #posterior = np.array([np.exp(np.sum(np.log(noise.pdf_noise((flux_no_nodes_no_planets - amp*flux_moon)[mask_moon_transit], *distribution_parameters))) - log0) for amp in relative_amplitude_array])
    posterior = np.array([negative_2loglikelihood((flux_no_nodes_no_planets - amp * flux_moon)[mask_moon_transit], Variance[mask_moon_transit], distribution_parameters) for amp in relative_amplitude_array])
    #norm = np.sum(posterior) * (relative_amplitude_array[1] - relative_amplitude_array[0])
    #posterior /= norm
    #radii = amplitude_to_radius(relative_amplitude_array*parameters[-1])

    cumulative = np.cumsum(posterior*(relative_amplitude_array[1] - relative_amplitude_array[0]))
    #min = radii[find_index(0.025, cumulative)]
    #max = radii[find_index(1-0.025, cumulative)]
    #print('r_optimal = {0} R_earth'.format(radii[np.argmax(posterior)]))
    #print('r_min: {0} R_earth'.format(min))
    #print('r_max: {0} R_earth'.format(max))
    #posterior = (posterior * 2 * radii) / ((1 - (u1 / 3.0) - (u2 / 6.0)) * sigma_in_star_flux_units * (-parameters[-1]) * R_star ** 2)
    plt.plot(relative_amplitude_array, posterior)
    #plt.plot(radii, posterior)
    #plt.xlim(0, 2)
    #plt.ylim(0, 1.2*np.max(posterior))
    #plt.ylabel('posterior probability')
    #plt.xlabel(r'radius of the moon [$R_{Earth}$]')
    plt.xlabel('amplitude')
    plt.ylabel('-2 log p')
    plt.show()
    #plt.savefig('moon_posterior')
    plt.close()
    return flux_moon

def posterior_pdf(logp_function, parameter1, parameter2, sigma1, sigma2):
    """plot -2log p in the surrounding of minimum"""
    print(sigma1)
    print(sigma2)
    x = parameter1 + np.linspace(-1, 1, 30)*sigma1
    y = parameter2 + np.linspace(-1, 1, 30)*sigma2
    logp0 = logp_function(parameter1, parameter2)
    X, Y = np.meshgrid(x, y) #shape = (len(y), len(x))
    Z = np.array([[logp_function(X[i, j], Y[i, j]) for j in range(len(x))] for i in range(len(y))]) - logp0

    plt.figure(figsize = (15, 10))
    plt.title('-2log p')
    contours = plt.contour(X-parameter1, Y-parameter2, Z, colors='black', levels = [1, 4, 9])
    plt.clabel(contours, inline=True, fontsize=8)
    plt.imshow(Z, extent=[-sigma1, sigma1, -sigma2, sigma2], origin='center', cmap='viridis', alpha=0.5, vmin = 0, vmax = 10)
    plt.colorbar()
    plt.savefig('logp_profile')
    plt.show()


def logp_error_of_parameter(logp_function, parameter0, step, deviation):
    """step is advised to be on the order of deviation (for example np.sqrt(cov(parameter0)) if deviation = 1)"""
    logp0 = logp_function(parameter0)
    logp = logp0
    parameter = parameter0
    step_number = 0
    while logp - logp0 < deviation:
        parameter += step
        logp = logp_function(parameter)
        if step_number > 20:
            print("Step number in logp_error_of_parameter exceeded maximum step number = 10")
            raise RuntimeError
        step_number += 1

    upper = optimize.brentq(lambda p: logp_function(p) - logp0 - deviation, parameter - step, parameter)
    logp = logp0
    parameter = parameter0
    step_number = 0
    while logp - logp0 < deviation:
        parameter -= step
        logp = logp_function(parameter)
        if step_number > 10:
            print("Step number in logp_error_of_parameter exceeded maximum step number = 10")
            raise RuntimeError
        step_number += 1
    lower =  optimize.brentq(lambda p: logp_function(p) - logp0 - deviation, parameter, parameter + step)
    return [parameter0-lower, upper - parameter0]


def fit_out_of_phase_with_real_time(real_Time, flux, Variance):
    deviation = 2**2
    r_array = [0.014252558501556694, 0.01542692661778983, 0.04339823993170634, 0.06042657761708681] #Kepler 90 e, f, g, h respectively
    #r_array = [0.04339823993170634, 0.06042657761708681]
    splines = [prepare_shape(r) for r in r_array]

    # if long_cadence:
    #     Flux = (flux - flux_nodes(np.load("params_nodes_lc.npy"), len(Variance)))[Variance < 2]
    #     phases0 = [[1.55885605e+01, 2.26041792e+02, 4.36769846e+02, 8.57960419e+02, 1.06854995e+03, 1.28022221e+03],
    #                [8.97, 340.63, 1335.38, ]]
    #     time_half_transit = [2.39409012e-01, 0.27662010037398799]
    #     amplitudes = [-2.13050749e+01, -40.289663555792735]
    if long_cadence:
        Flux = (flux - flux_nodes(np.load("params_nodes_lc.npy"), len(Variance)))[Variance < 2]
        phases0 = [[2.7865268329682, 94.72634087318734, 186.66615491340647, 370.54578299384474, 462.48559703406386, 554.425411074283, 738.3050391547213, 830.2448531949404, 922.1846672351595, 1106.0642953155977, 1198.0041093558168, 1289.9439233960359],
                   [123.22434621200968, 747.7722780588771, 872.6818644282505, 1122.5010371669975],
                   [1.55885605e+01, 2.26041792e+02, 4.36769846e+02, 8.57960419e+02, 1.06854995e+03, 1.28022221e+03],
                   [8.97, 340.63, 1335.38, ]]
        time_half_transit = [0.18509365269252892, 0.21406437446323237, 2.39409012e-01, 0.27662010037398799]
        amplitudes = [-2.4269329710006096, -2.8907384595715637, -2.13050749e+01, -40.289663555792735]
        planet_name = ['e', 'f', 'g', 'h']
    else:
        Flux = (flux - flux_nodes(np.load("params_nodes_sc.npy"), len(Variance)))[Variance < 2]
        phases0 = [[8.57960419e+02, 1.06854995e+03, 1.28022221e+03], [ 1335.38, ]]
        time_half_transit = [2.39409012e-01, 0.27662010037398799]
        amplitudes = [-2.13050749e+01 / np.sqrt(30), -40.289663555792735 / np.sqrt(30)]

    def single_profile_with_spline_real_time_no_params(t, phas, tht, amp):
        return single_profile_with_spline_real_time(t, phas, tht, amp, spline)

    # fit both planets
    for planet in range(len(amplitudes)):
        arr_phase = np.zeros(shape = (len(phases0[planet]), 3))
        arr_duration = np.zeros(shape =(len(phases0[planet]), 3))
        spline = splines[planet]
        for tranist_num in range(len(phases0[planet])):
            params, cov = optimize.curve_fit(single_profile_with_spline_real_time_no_params, real_Time, Flux, [phases0[planet][tranist_num], time_half_transit[planet], amplitudes[planet]] )
            # err_phase = logp_error_of_parameter(lambda p: logp_model_sc(Flux - single_profile_with_spline_real_time_no_params(real_Time, p, params[1], params[2]), Variance), params[0], np.sqrt(cov[0, 0]), deviation)
            # err_duration = logp_error_of_parameter(lambda p: logp_model_sc(Flux - single_profile_with_spline_real_time_no_params(real_Time, params[0], p, params[2]), Variance), params[1], np.sqrt(cov[1, 1]), deviation)
            err_phase = logp_error_of_parameter(lambda p: negative_2loglikelihood(Flux - single_profile_with_spline_real_time_no_params(real_Time, p, params[1], params[2]), np.ones(len(real_Time)), distribution_parameters), params[0], np.sqrt(cov[0, 0]), deviation)
            err_duration = logp_error_of_parameter(lambda p: negative_2loglikelihood(Flux - single_profile_with_spline_real_time_no_params(real_Time, params[0], p, params[2]), np.ones(len(real_Time)), distribution_parameters), params[1], np.sqrt(cov[1, 1]), deviation)

            arr_phase[tranist_num, 0] = params[0] + 131.512439459
            arr_phase[tranist_num, 1] = err_phase[0]
            arr_phase[tranist_num, 2] = err_phase[1]
            arr_duration[tranist_num, 0] = params[1] *2
            arr_duration[tranist_num, 1] = err_duration[0]*2
            arr_duration[tranist_num, 2] = err_duration[1]*2

        np.savetxt('TTV_' + planet_name[planet] + '.txt', arr_phase)
        np.savetxt('TDV_' + planet_name[planet] + '.txt', arr_duration)


def planet_out_of_phase(Time, Flux, Variance, params0, splines, r_planets, type_of_fit, direct_return=False):
    """fit for planets with out of phase transits
    if direct_return == True, it does not perform fit, but just returns flux coresponding to given parameters.
    params0: [[phase1_planet1, phase2_planet1, ... phaseN_planet1, time_half_transit_planet1, amplitude_planet1],
              [phase1_planet2, phase2_planet2, ... phaseN_planet2, time_half_transit_planet2, amplitude_planet2],
                 .... ]
    splines = [spline_planet1, splines_planet2, ..., splines_planetN]
    r_planets = [r_planet1, r_planet2, ..., r_planetN]
    """

    def planet_phases(Time, *parameters):
        """allows for TTV -> transit time variations (changing phase of transit)"""
        flux = np.zeros(len(Time))
        for i in range(len(parameters) - 2):
            flux += profile_with_spline(Time, Time[-1], parameters[i], parameters[-2], parameters[-1], spline_planet)
        return flux

    def planet_phases_transits(Time, *parameters):
        """allows for TTV and TDV -> transit duration variation"""
        num_transits = (len(parameters) - 2) // 2
        flux = np.zeros(len(Time))
        for i in range(num_transits):
            flux += profile_with_spline(Time, Time[-1], parameters[i], parameters[i + num_transits], parameters[-1], spline_planet)
        return flux

    def planet_amplitude_tau(Time, A):
        """allows for amplitude and time of transit with fixed TTV"""
        flux = np.zeros(len(Time))
        for i in range(len(phases)):
            flux += profile_with_spline(Time, Time[-1], phases[i], tau, A, spline_planet)
        return flux

    flux_planets = np.zeros(len(Variance))
    params_return = []

    if direct_return == True:
        for p in range(len(r_planets)):
            spline_planet = splines[p]
            flux_planets += planet_phases(Time, *params0[p])
        return flux_planets

    # fit planets
    else:
        for p in range(len(r_planets)):
            spline_planet = splines[p]
            r_planet = r_planets[p]

            # logp fit
            if logp_planets:
                if p == 2:
                    params1 = params0[p]
                else:
                    borders = [(params0[p][phase_num] - 0.1, params0[p][phase_num] + 0.1) for phase_num in
                               range(len(params0[p]) - 2)] + [(params0[p][-2] - 0.1, params0[p][-2] + 0.1),
                                                              (-np.inf, np.inf)]
                    params1 = optimize.fmin_l_bfgs_b(gradient_logp_and_logp_TTV, x0=np.array(params0[p]), bounds=borders,
                                                     args=(Time, Flux-flux_planets, Variance, spline_planet, r_planet,
                                                           distribution_parameters))[0]

            # gaussian fit
            else:
                if type_of_fit[p] == 'TTV':
                    params1, cov = optimize.curve_fit(planet_phases, Time, Flux-flux_planets, params0[p], sigma=np.sqrt(Variance))
                elif type_of_fit[p] == 'A':
                    phases= params0[p][:-2]
                    tau = params0[p][-2]
                    params1, cov = optimize.curve_fit(planet_amplitude_tau, Time, Flux-flux_planets, params0[p][-1], sigma=np.sqrt(Variance))
                    params1 = np.concatenate((phases, [tau, params1[0]]))
                else:
                    print('Invalid type_of_fit in function planet_out_of_phase')
            flux_planets += planet_phases(Time, *params1)
            params_return.append(params1)
        return flux_planets, params_return


def phase_fit(Time, Flux, Variance, x0_transitions, r, number_of_planet, phase0):
    """separately fit phases of transitions"""
    period = x0_transitions[0+number_of_planet*4]
    phase = x0_transitions[1+number_of_planet*4]
    time_half_transit = x0_transitions[2+number_of_planet*4]
    amplitude = x0_transitions[3+number_of_planet*4]

    params0 = [Time[-1], phase0, time_half_transit, amplitude]
    spline = prepare_shape(r[number_of_planet])
    params, cov = planet_fit_brez_perioda(params0, Time, Flux, Variance, spline)
    return params[0], params[2], np.sqrt(cov[0, 0])

def ultimate_phase_fit(Time, flux_with_planets, Variance, x0_transitions, r, num, name):
    """computes phases for num th planet. name is name of the planet. x0_transitions is initial guess of parameters"""
    splines = prepare_splines(r)
    number_of_planet = num
    x0_transitions_brez = x0_transitions[:4*number_of_planet] + x0_transitions[4*(number_of_planet+1):]
    splines_brez = splines[:number_of_planet] + splines[number_of_planet+1:]
    r_brez = r[:number_of_planet] + r[number_of_planet+1:]
    Flux = flux_with_planets - flux_transitions(x0_transitions_brez, Time, splines_brez)
        
    period = x0_transitions[0 + number_of_planet * 4]
    phase = x0_transitions[1 + number_of_planet * 4]
    time_half_transit = x0_transitions[2 + number_of_planet * 4]
    amplitude = x0_transitions[3 + number_of_planet * 4]
    r_zdej = r[number_of_planet]
    number_of_transits = int(Time[-1] / period)
    if phase + time_half_transit < math.fmod(Time[-1], period):  # in case remainder contatins a full transit
        number_of_transits += 1
    print(number_of_transits)

    all_phases = [k*period+phase for k in range(number_of_transits)]
    k = 0
    ff = open('kepler90' + name + '.txt', 'w')
    amplitude_failure =0
    error_failure = 0
    success = 0
    for p in all_phases:
        #if there is no planet transit where it should be, optimization can return optimization error
        #that crashes program. Easiest way to avoid that:
        try:
            f, A, napaka = phase_fit(Time, Flux, Variance, x0_transitions, r, number_of_planet, p)
        except RuntimeError:
            pass
        print("{0}\t{1}\t{2}\t{3}".format(k, f - k * period - phase, napaka, A))
        if abs(A-amplitude)/abs(amplitude) > 1 :
            amplitude_failure +=1
        if napaka > 100000:
            error_failure += 1
        if abs(A-amplitude)/abs(amplitude) > 1 :
            amplitude_failure +=1
        if napaka > 100000:
            success += 1
            print("{0}\t{1}\t{2}".format(k, f, napaka), file = ff)

        k+=1
    print(amplitude_failure)
    print(error_failure)
    print(success)
    ff.close()


def map_u1u2_to_q1q2(u1, u2):
    return (u1+u2)**2, 0.5*u1/(u1+u2)

def map_q1q2_to_u1u2(q1, q2):
    return 2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)


def limb_darkening_impact():
    """estimate how small should intervals in u1 and u2 be, so that there is practically no loss in SNR (for transit search)"""
    time_half_transit = 0.2
    integer_time_half_transit = int(time_half_transit / dt) + 1
    r = 0.01
    steps = 20
    array1 = np.linspace(0, 1, steps)  # np.linspace(-1, 1, steps)*0.4 + q10
    array2 = array1  # np.linspace(-1, 1, steps)*0.4 + q20
    X, Y = np.meshgrid(array1, array2)
    Z = np.ones((steps, steps))
    #templates = np.array([[0.35, 0.25], [0.5, 0.4], [0.62, 0.57]])
    templates = np.array([[0.49, 0.35], ])
    for num_template in range(len(templates)):
        #u10, u20 = #u1_Kepler90, u2_Kepler90
        q10, q20 = templates[num_template]#map_u1u2_to_q1q2(u10, u20)
        u10, u20 = map_q1q2_to_u1u2(q10, q20)

        spline = prepare_shape(r, u10, u20)
        template0 = [- interpolate.splint((t - dt * 0.5) / time_half_transit, (t + dt * 0.5) / time_half_transit, spline) * time_half_transit / dt for t in (np.arange(-integer_time_half_transit, integer_time_half_transit) + 0.5) * dt]
        norm = np.sum(np.square(template0))
        template0 /= np.sqrt(norm)

        for i in range(steps):
            for j in range(steps):
                spline = prepare_shape(r, *map_q1q2_to_u1u2(X[i, j], Y[i, j]))
                template = [- interpolate.splint((t - dt * 0.5) / time_half_transit, (t + dt * 0.5) / time_half_transit, spline) * time_half_transit / dt for t in (np.arange(-integer_time_half_transit, integer_time_half_transit) + 0.5) * dt]
                template /= np.sqrt(norm)
                #planet_fit(x0_planet, Time, Flux, Variance, spline,
                delta = np.sqrt(np.sum(np.square(template - template0)))  # amplitude and time_half transit should really be to obtain lower numbers, this is an upper bound
                if delta < Z[i, j]:
                    Z[i, j] = delta

    xmax, ymax = 3, 5
    plt.figure(figsize=(xmax * 3, ymax * 3))
    plt.contourf(X, Y, Z, cmap = 'viridis')
    plt.plot(templates[:, 0], templates[:, 1], 'o', color = 'red')
    legend = plt.colorbar(orientation='horizontal')
    legend.set_label(r'$\sqrt{\Delta \chi^2} / SNR_0$')
    plt.xlabel(r'$q_1$')
    plt.ylabel(r'$q_2$')
    #plt.savefig('bound_on_SNR_loss_limbdarkening')
    plt.show()


def radius_impact():
    """estimate how small should intervals in u1 and u2 be, so that there is practically no loss in SNR (for transit search)"""
    time_half_transit = 0.2
    integer_time_half_transit = int(time_half_transit / dt) + 1
    r0 = 0.01
    radius = np.linspace(0.005, 0.04, 50)
    Z = np.empty(len(radius))
    q10, q20 = [0.49, 0.35]
    u10, u20 = map_q1q2_to_u1u2(q10, q20)
    spline = prepare_shape(r0, u10, u20)
    template0 = [- interpolate.splint((t - dt * 0.5) / time_half_transit, (t + dt * 0.5) / time_half_transit, spline) * time_half_transit / dt for t in (np.arange(-integer_time_half_transit, integer_time_half_transit) + 0.5) * dt]
    norm = np.sum(np.square(template0))
    template0 /= np.sqrt(norm)
    for num_template in range(len(radius)):

        spline = prepare_shape(radius[num_template], u10, u20)
        template = [- interpolate.splint((t - dt * 0.5) / time_half_transit, (t + dt * 0.5) / time_half_transit, spline) * time_half_transit / dt for t in (np.arange(-integer_time_half_transit, integer_time_half_transit) + 0.5) * dt]
        template /= np.sqrt(norm)
        #planet_fit(x0_planet, Time, Flux, Variance, spline,
        delta = np.sqrt(np.sum(np.square(template - template0)))  # amplitude and time_half transit should really be to obtain lower numbers, this is an upper bound
        Z[num_template] = delta

    plt.plot(radius, Z)
    plt.title('Impact on detected SNR if template has r = 0.01, but real radius is different')
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\sqrt{\Delta \chi^2} / SNR_0$')
    #plt.savefig('bound_on_SNR_loss_limbdarkening')
    plt.show()




def errors_planet_parameters(Time, Variance):
    """simulates flux and a planet and compares delta logp joint fit gives with what spline gives"""
    np.random.seed(0)
    r = 0.006
    spline = prepare_shape(r)
    SNR = 8.0
    #flux_base = np.load('search_flux_semi_kill.npy')
    Pk = np.load('Pk_posterior_sim.npy')
    Pk_paddled = np.load(home+'matched_filter/Pk_for_simulations.npy')
    x0_nodes = np.load('params_nodes_semi_kill.npy')
    period = 100.0#np.linspace(3, 300, 100)
    ponovitve = 1000
    deviations = np.empty(shape = (ponovitve, 4))

    for i in range(ponovitve):
        print(i)
        TryAgain = True
        while TryAgain:
            try:
                #simulation
                flux_planet, params = planet_simulation(Time, Variance, SNR, spline, period, phase_given=None, time_half_transit_given= 0.75 * q * np.cbrt(period), Pk = Pk_paddled)
                flux = flux_planet + star_noise_simulation(Variance, Pk_paddled)

                #joint fit
                deviations[i, :] = joint_fit(Time, flux, Variance, np.array(params), x0_nodes, [spline, ], [r, ], Pk, what_return='planets', type_of_fit='normal') - params

                TryAgain = False
            except RuntimeError:
                print('RuntimeError')
                TryAgain = True
            except ValueError:
                print('ValueError')
                TryAgain = True
    np.savetxt(home + 'look_elsewhere/FGP_frequentistic_bounds.txt', deviations)

def prepare_flux_for_emcee(Time, Variance):
    """simulates flux and a planet and compares delta logp joint fit gives with what spline gives"""
    np.random.seed(0)
    r = 0.006
    spline = prepare_shape(r)
    SNR = 8.0
    #flux_base = np.load('search_flux_semi_kill.npy')
    Pk = np.load('Pk_posterior_sim.npy')
    Pk_paddled = np.load(home+'matched_filter/Pk_for_simulations.npy')
    x0_nodes = np.load('params_nodes_semi_kill.npy')
    period = 100.0
    flux_planet, params = planet_simulation(Time, Variance, SNR, spline, period, phase_given=None, time_half_transit_given= 0.75 * q * np.cbrt(period), Pk = Pk_paddled)
    star = star_noise_simulation(Variance, Pk_paddled)
    flux = flux_planet + star
    flux_no_star = joint_fit(Time, flux, Variance, np.array(params), x0_nodes, [spline, ], [r, ], Pk, what_return='flux fit', type_of_fit='normal')
    np.save('flux_no_star.npy', flux_no_star)
    print(params)


def samples_from_posterior(Time, Variance):
    r = 0.006
    spline = prepare_shape(r)
    params0 = np.array([100.0, 54.86378834257569, 0.16353569051607389, -0.57168288900844699])
    lower_bound = [-2 * dt, -3 * dt, -5.9 * dt, -1 - params0[-1]]
    upper_bound = [2* dt, 3 * dt, 5.9 * dt, -0.1 - params0[-1]]
    flux = np.load('flux_no_star.npy')
    # t, f = ts.fold_flux(Time, flux, Variance, params0[0], dt, False, centers_of_bins = True, phase_given = None)
    # plt.plot(t, f)
    #plt.xlim(params0[1] - 2*params0[2], params0[1] + 2*params0[2])
    # plt.show()
    nwalkers = 1000

    #log likelihood function
    logp0 = -0.5 * np.sum(np.square(flux - profile_with_spline(Time, *(params0), spline))/(Variance))
    def log_prob(delta_params):
        #delta_params = ts.vector_inverse_of_map_to_real_line(delta_params, lower_bound, upper_bound)
        if np.sum(delta_params<lower_bound) + np.sum(delta_params> upper_bound) != 0: #delta_params is then out of bounds
            return -np.inf
        return -0.5 * np.sum(np.square(flux - profile_with_spline(Time, *(params0 + delta_params), spline))/Variance)  - logp0

    #initial position
    sigmas = np.array([10.0/(24*60.0), 10.0/(24*60.0), 10.0/(24*60.0), 0.57/8])*0.1
    p0 = np.random.randn(nwalkers, 4)*sigmas
    #p0 = ts.matrix_map_to_real_line(p0, lower_bound, upper_bound)
    sampler = emcee.EnsembleSampler(nwalkers, 4, log_prob)

    #burn in
    state = sampler.run_mcmc(p0, 1000, progress = True)
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    sampler.reset()

    #sampling
    sampler.run_mcmc(state, 5000, progress = True)
    samples = sampler.get_chain(flat=True)[-1000:, :]
    #samples = ts.matrix_inverse_of_map_to_real_line(samples, lower_bound, upper_bound)

    np.savetxt(home + 'look_elsewhere/FGP_emcee.txt', samples)
    print("Mean acceptance fraction: {0} \nAutocorrelation time: {1}".format(np.mean(sampler.acceptance_fraction), sampler.get_autocorr_time()))


def marginalize_with_integral(Time, Variance):
    r = 0.006
    spline = prepare_shape(r)
    params0 = np.array([100.0, 54.86378834257569, 0.16353569051607389, -0.57168288900844699])
    lower_bound = [-0.5 * dt, -2 * dt, -4*dt, -0.5]
    upper_bound = [0.5* dt, 2 * dt, 4*dt, 0.57]
    flux = np.load('flux_no_star.npy')

    logp0 = -0.5 * np.sum(np.square(flux - profile_with_spline(Time, *(params0), spline))/(Variance))

    def prob0(x1, x2, x3):
        return np.exp(-0.5 * np.sum(np.square(flux - profile_with_spline(Time, params0[0]+y0, params0[1]+x1, params0[2]+x2, params0[3]+x3, spline))/Variance)  - logp0)
    def prob1(x1, x2, x3):
        return np.exp(-0.5 * np.sum(np.square(flux - profile_with_spline(Time, params0[0]+x1, params0[1]+y0, params0[2]+x2, params0[3]+x3, spline))/Variance)  - logp0)
    def prob2(x1, x2, x3):
        return np.exp(-0.5 * np.sum(np.square(flux - profile_with_spline(Time, params0[0]+x1, params0[1]+x2, params0[2]+y0, params0[3]+x3, spline))/Variance)  - logp0)
    func = [prob0, prob1, prob2]


    index_y  = 2
    y = np.linspace(lower_bound[index_y], upper_bound[index_y], 200)

    p = np.empty(shape = (len(y), 3))
    for i in range(len(y)):
        print(i)
        y0 = y[i]
        #integrate.tplquad(func[0], lower_bound[3], upper_bound[3], lambda x: lower_bound[2], lambda x: upper_bound[2], lambda x, y: lower_bound[1], lambda x, y: upper_bound[1])
        #integrate.tplquad(func[1], lower_bound[3], upper_bound[3], lambda x: lower_bound[2], lambda x: upper_bound[2], lambda x, y: lower_bound[0], lambda x, y: upper_bound[0])
        value, err = integrate.tplquad(func[2], lower_bound[3], upper_bound[3], lambda x: lower_bound[1], lambda x: upper_bound[1], lambda x, y: lower_bound[0], lambda x, y: upper_bound[0])
        p[i, 0] = y[i]
        p[i, 1] = value
        p[i, 2] = err

    overall_normalization = integrate.quad(p[:, 1], p[:, 0])
    p /= overall_normalization
    word = ['period', 'phase', 'tau', 'amplitude']
    np.save('posterior'+ word[index_y] + '.npy', p)

    plt.errorbar(p[:, 0], p[:, 1], yerr = p[:, 2])
    plt.show()

def delta_logp_planets_individual_hyperpiror(Time, Flux, Variance, r, amplitude):
    """computes -2(logp(without given planet) - logp(all planets)) for all planets with parameters given in x0_transitions
    number of planets = len(r)""" 
    splines = prepare_splines(r)
    r_g, r_h = 0.04339823993170634, 0.06042657761708681
    spline_g = prepare_shape(r_g)
    spline_h = prepare_shape(r_h)
    params_g = [1.55885605e+01, 2.26041792e+02, 4.36769846e+02, 8.57960419e+02, 1.06854995e+03, 1.28022221e+03, 2.39409012e-01, -2.13050749e+01]
    params_h = [8.97, 340.63, 1335.38, 0.27662010037398799, -40.289663555792735]

    number_of_fourier_nodes = int(2.0 * dt * len(Time)) + 1
    Pk = np.load('Pk_semi_kill.npy')
    #s = np.fft.rfft(Flux)[1:number_of_fourier_nodes+1]
    x0_nodes = np.load('params_nodes_semi_kill.npy')
    x0_transitions= np.ndarray.tolist(np.load('params_planets_semi_kill.npy'))

    logp0 = joint_fit(Time, Flux, Variance, np.array(x0_transitions), x0_nodes, splines, r, Pk, None, [params_g, params_h], [spline_g, spline_h], [r_g, r_h], 'logp', 'out of phase')
    print(logp0)
    #planets in phase (first 6)
    for planet in range(len(r)):
        x0_transitions_brez = x0_transitions[:4*planet] + x0_transitions[4*(planet+1):]
        splines_brez = splines[:planet] + splines[planet+1:]
        r_brez = r[:planet] + r[planet+1:]
        #for iteration_number in range(5):
        logp = joint_fit(Time, Flux, Variance, np.array(x0_transitions_brez), x0_nodes, splines_brez, r_brez, Pk, None, [params_g, params_h], [spline_g, spline_h], [r_g, r_h], 'logp', 'out of phase')
        print(np.sqrt(logp - logp0))

    #kepler 90g
    logp = joint_fit(Time, Flux, Variance, np.array(x0_transitions), x0_nodes, splines, r, Pk, None, [params_h,], [spline_h,], [r_h, ], 'logp', 'out of phase')
    print(np.sqrt(logp - logp0))

    #kepler 90h
    logp = joint_fit(Time, Flux, Variance, np.array(x0_transitions), x0_nodes, splines, r, Pk, None, [params_g,], [spline_g,], [r_g, ], 'logp', 'out of phase')
    print(np.sqrt(logp - logp0))

    #extra planet candidate
    # r_extra = 0.007
    # spline_extra = prepare_shape(r_extra)
    # params_extra = [180.54297417678504, 340.39868562245323, 500.3647167326008, 660.3294275601119, 820.3090589407808, 980.1900849907786, 1140.0792414059274, 1300.0428512144013, q* np.cbrt(159.889), amplitude]
    # logp = joint_fit(Time, Flux, Variance, np.array(x0_transitions), x0_nodes, splines, r, Pk, None, [params_g, params_h, params_extra],[spline_g, spline_h, spline_extra], [r_g, r_h, r_extra ], 'planets', 'out of phase')
    # #print(logp0 - logp)


def candidate_delta_logp(Time, Flux, Variance, Pk, x0_transitions, x0_nodes, splines, spline_g, spline_h, r, logp0):
    logp = joint_fit(Time, Flux, Variance, np.array(x0_transitions[4:]), x0_nodes, splines[1:], r[1:], Pk, None, spline_g = spline_g, spline_h = spline_h)[-1]
    return np.sqrt(logp - logp0)


def moon_optimal_relative_phases(Time, Flux, Variance, x0_transitions, r, relative_moon_phases, amplitude_moon):
    splines = prepare_splines(r)
    x0_off = [[1.55885605e+01, 2.26041792e+02, 4.36769846e+02, 8.57960419e+02, 1.06854995e+03, 1.28022221e+03, 2.39409012e-01, -2.13050749e+01],
               [8.97, 340.63, 1335.38, 0.27662010037398799, -40.289663555792735], ]
    r_off = [0.04339823993170634, 0.06042657761708681]
    splines_off = prepare_splines(r_off)
    Pk = np.load('Pk_semi_kill.npy')
    x0_nodes = np.load('params_nodes_semi_kill.npy')
    planet_param = joint_fit(Time, Flux, Variance, np.array(x0_transitions), x0_nodes, splines, r, Pk, None, x0_off, splines_off, r_off, 'planets', 'moon', relative_moon_phases, amplitude_moon)


def single_joint_fit(Time, Flux, Variance, x0_transitions, r, Pk):
    splines = [prepare_shape(r), ]
    number_of_fourier_nodes =  int(2.0 * dt * len(Time)) + 1
    s = np.fft.rfft(Flux)[1:number_of_fourier_nodes+1]
    x0_nodes = np.concatenate((np.real(s), np.imag(s)))
    np.save('noise0_candidate9_fit_flux', joint_fit(Time, Flux, Variance, np.array(x0_transitions), x0_nodes, splines, [r,], Pk, what_return = 'flux fit', type_of_fit = 'normal'))


def iterate_hyperprior(Time, Flux, Variance, x0_transitions, r, type_of_fit):
    """performs joint fit and iterates to obtain a true hiperprior"""
    splines = prepare_splines(r)
    number_of_fourier_nodes =  int(2.0 * dt * len(Time)) + 1
    if long_cadence:
        ####for comparisson with the matched filter
        # Pk = np.load(home + 'matched_filter/Pk_for_simulations.npy')
        # Pk = (Pk * 1.96743434718) - len(Variance) #to make it white noise with sigma = 1 at high frequencies and then substract white noise, Fourier GP is invariant under P(k) -> alfa * P(k)
        # Pk = Pk[1:number_of_fourier_nodes + 1] #only up to frequencies 2/day
        # mask = Pk < 1000 #some P end up negative by substracting, lets fix this partialy, we will then iterae on Pk anyway
        # Pk[mask] = np.ones(np.sum(mask)) * 1000

        Pk = np.load('Pk_semi_kill.npy') #primary use, for Kepler 90
    else:
        Pk = np.load('Pk_sc.npy')
    # [49.03053471778504, 208.88624616345322, 368.8522772736008, 528.8169881011119, 688.7966194817808, 848.6776455317786, 1008.5668019469273, 1168.5304117554012, 1328.2883936188566, 1488.0327841049948]
    if long_cadence:
        x0_off = [[ 1.55885605e+01, 2.26041792e+02, 4.36769846e+02, 8.57960419e+02, 1.06854995e+03, 1.28022221e+03, 2.39409012e-01, -2.13050749e+01],
                  [8.97, 340.63, 1335.38, 0.27662010037398799, -40.289663555792735],  ]
    else:
        x0_off = [[8.57960419e+02, 1.06854995e+03, 1.28022221e+03, 2.39409012e-01, -4.40594706],
                  [ 1335.38, 0.27662010037398799, -8.93753168e+00] ]

    #r_off = [ 0.06042657761708681, ]
    r_off = [0.04339823993170634, 0.06042657761708681]
    splines_off = prepare_splines(r_off)

    x0_nodes = np.load('params_nodes_semi_kill.npy')#np.concatenate((np.real(s), np.imag(s)))
    #Pk = exponential_kill_high_frequencies(kji, Pk)
    return joint_fit(Time, Flux, Variance, np.array(x0_transitions), x0_nodes, splines, r, Pk, None, x0_off, splines_off, r_off, 'flux fit', type_of_fit)
    #for iteration_number in range(10):
        #Pk returned by joint fit will be used in the next iteration
        #Pk, nodes, planet_param, phases, power_sim, power_nodes = joint_fit(Time, Flux, Variance, np.array(x0_transitions), x0_nodes, splines, r, Pk, iteration_number, x0_off, splines_off, r_off, 'new hyperprior', type_of_fit)
    #    Pk = star_fit(Flux, Variance,x0_nodes, Pk, iteration_number)


        #print(logp)
        #Pk, nodes, planet_param, logp = joint_fit(Time, Flux, Variance, np.array(x0_transitions), x0_nodes, splines, r, Pk, iteration_number, spline_g, spline_h)

        #print(iteration_number)
        # np.save('out_of_phases_semi_kill.npy', phases)
        #np.save('Pk_posterior_sim.npy', Pk)
        # np.save('params_nodes_semi_kill.npy', nodes)
        # np.save('params_planets_semi_kill.npy', planet_param)
        # np.save('sim_semi_kill.npy', power_sim)
        # np.save('power_nodes_semi_kill.npy', power_nodes)

        #print(candidate_delta_logp(Time, Flux, Variance, Pk, planet_param, nodes, splines, spline_g, spline_h, r, logp))
    #candidate_plot(Time, Flux, Variance, np.ndarray.tolist(np.load('params_planets_candidate2.npy')), np.load('out_of_phases.npy'), np.load('params_nodes_candidate2.npy'), splines, spline_g, spline_h, r)


def star_fit(flux_no_planets, Variance, x0_nodes, Pk, current_number_of_iterations = None):
    """like joint fit but without planets so no iteration [planets <-> star] is needed"""
    Pk_double = np.concatenate((Pk, Pk)) #we need double Pk to directly divide (sx_k, sy_k) by P_k
    params_nodes = x0_nodes
    params_nodes = optimize.fmin_l_bfgs_b(chi_nodes, x0 = params_nodes, args= (flux_no_planets, Variance, Pk_double))[0]
    new_hp, power_sim, power_nodes = new_hyperprior(Variance, params_nodes, Pk, current_number_of_iterations)
    return new_hp


def joint_fit(Time, Flux, Variance, x0_transitions, x0_nodes, splines, r, Pk, current_number_of_iterations = None, x0_off =None, splines_off = None, r_off = None, what_return = 'new_hyperprior', type_of_fit = 'out of phase', relative_moon_phases = None, amplitue_moon = None):
    """measured Time and Flux of signal and Variance of Flux (mostly 1, expcept where there is mising data where it is np.inf,
    planets : number of planets to fit
    x0: initial guess for parameters
    r: table of radii of planets
    Pk: hyperprior"""

    if long_cadence: #in short cadence we do not fit for small planets (there was no need for now)
        planets = (len(x0_transitions)) // 4
        params_transitions = np.copy(x0_transitions)
        borders = [[(x0_transitions[p*4]-0.1, x0_transitions[p*4]+0.1), (x0_transitions[p*4+1]-0.1, x0_transitions[p*4+1]+0.1), (x0_transitions[p*4+2]-0.05, x0_transitions[p*4+2]+0.05), (-np.inf, np.inf)] for p in range(planets)]

    Pk_double = np.concatenate((Pk, Pk))  # we need double Pk to directly divide (sx_k, sy_k) by P_k
    params_nodes = x0_nodes
    flux_no_nodes = Flux - flux_nodes(params_nodes, len(Time))

    for i in range(20):

        #fit planets (/moon)
        if type_of_fit == 'moon':
            flux_planets, params_off = planet_out_of_phase(Time, flux_no_nodes, Variance, [x0_off[1], ], [splines_off[1], ], [r_off[1], ], ['TTV'])
            flux_moon, params_planet_moon = planet_moon(Time, flux_no_nodes, Variance, relative_moon_phases, amplitue_moon)
            flux_planets += flux_moon
        elif type_of_fit == 'out of phase':
            flux_planets, params_off = planet_out_of_phase(Time, flux_no_nodes, Variance, x0_off, splines_off, r_off, ['TTV', 'TTV', 'A'])
        elif type_of_fit == 'normal':
            flux_planets = np.zeros(len(Flux))
        else:
            raise Exception("Invalid value of parameter type_of_fit, it should be 'moon', 'out of phase' or 'normal', but is " + type_of_fit)
        if long_cadence:
            for p in range(planets-1, -1, -1):
                if logp_planets:
                    opt = optimize.fmin_l_bfgs_b(gradient_logp_and_logp, x0 = np.array(params_transitions[p*4: (p+1)*4]), bounds = borders[p], args = (Time, flux_no_nodes-flux_planets, Variance, splines[p], distribution_parameters))
                    params_transitions[p * 4: (p + 1) * 4] = opt[0]
                else:
                    params_transitions[p * 4: (p + 1) * 4], cov = planet_fit(x0_transitions[p * 4: (p + 1) * 4], Time, flux_no_nodes - flux_planets, Variance, splines[p], covariance = True)

                flux_planets += profile_with_spline(Time, *params_transitions[p*4: (p+1)*4], splines[p])

        #fit star
        flux_no_planets = Flux - flux_planets
        params_nodes = optimize.fmin_l_bfgs_b(chi_nodes, x0 = params_nodes, args= (flux_no_planets, Variance, Pk_double))[0]
        flux_no_nodes = Flux - flux_nodes(params_nodes, len(Time))

    #return options
    if what_return == 'new hyperprior':
        #print(logp_model(flux_no_nodes - flux_planets, params_nodes, Pk, Variance,distribution_parameters))
        new_hp, power_sim, power_nodes = new_hyperprior(Variance, params_nodes, Pk, current_number_of_iterations)
        #print(chi2_model(flux_no_nodes - flux_planets, params_nodes, np.concatenate((Pk, Pk)), Variance))
        return (new_hp, params_nodes, params_transitions, params_off, power_sim, power_nodes)#, logp)
    elif what_return == 'logp':
        return logp_model(flux_no_nodes - flux_planets, params_nodes, Pk, Variance, distribution_parameters)


    elif what_return == 'planets':
        #errors of inner planets amplitdes
        # for p in range(planets):
        #     err_amplitude = logp_error_of_parameter(lambda a: negative_2loglikelihood(
        #         flux_no_nodes - flux_planets - profile_with_spline(Time, *params_transitions[p*4: (p+1)*4-1], -a, splines[p], r[p]),
        #         Variance, distribution_parameters), 0.0, 0.1, 4.0)
        #     print((err_amplitude[1]+err_amplitude[0])/4)

        # posterior_pdf(lambda period, phase: negative_2loglikelihood(
        #     flux_no_nodes - profile_with_spline(Time, period, phase, *params_transitions[2: 4], splines[p], r[p]),
        #     Variance, distribution_parameters), params_transitions[0], params_transitions[1], 2*cov[0], 2*cov[1])
        return params_transitions#, cov_planet_i

        # err_period = logp_error_of_parameter(lambda P: negative_2loglikelihood(
        #     flux_no_nodes - profile_with_spline(Time, P, *params_transitions[1: 4], splines[p], r[p]),
        #     Variance, distribution_parameters), params_transitions[0], cov[0], 1.0)
        # err_phase = logp_error_of_parameter(lambda phase: negative_2loglikelihood(
        #     flux_no_nodes - profile_with_spline(Time, params_transitions[0], phase, *params_transitions[2:], splines[p], r[p]),
        #     Variance, distribution_parameters), params_transitions[1], cov[1], 1.0)
        # return np.concatenate((err_period, err_phase))

        #out of phase planets
        # for p in range(len(params_off)):
        #     params_for_f = [[params_off[i][j] for j in range(len(params_off[i]))] for i in range(len(params_off))]
        #     def f(a):
        #         for p2 in range(len(params_off)):
        #             params_for_f[p2][-1] = 0.0
        #         params_for_f[p][-1] = -a
        #         return negative_2loglikelihood(flux_no_nodes - flux_planets
        #                     - planet_out_of_phase(Time, flux_no_nodes, Variance, params_for_f, splines_off, r_off, ['TTV', 'TTV', 'A'], direct_return=True),
        #                                 Variance, distribution_parameters)
        #     err_amplitude = logp_error_of_parameter(f, 0.0, 0.1, 4.0)
        #     print(amplitude_to_radius(np.array([params_off[-1][-1] + err_amplitude[0], params_off[-1][-1], params_off[-1][-1] - err_amplitude[0]])))
        #     #print((err_amplitude[1] + err_amplitude[0]) / 4)

    elif what_return == 'flux fit':
        return flux_no_planets, params_off

    else:
        raise Exception("Invalid value of parameter what_return, it should be 'new hyperprior', 'logp', 'planets', 'flux fit', but is " + what_return)


def covariance_planets(Time, Flux, Variance, params_nodes, params_transitions, splines, r):
    flux_no_nodes = Flux - flux_nodes(params_nodes, len(Time))
    flux_planets = np.zeros(len(Time))
    planets = len(params_transitions)//4
    for p in range(planets-1, -1, -1):
        params, cov = planet_fit(params_transitions[p * 4: (p + 1) * 4], Time, flux_no_nodes - flux_planets, Variance, splines[p], covariance = True)
        flux_planets += profile_with_spline(Time, *params, splines[p], r[p])
        np.save('parameters' + str(p), params)
        np.save('covariance' + str(p), cov)

def flux_transitions(params_transitions, Time, splines):
    """flux impact of planets, excluding out of phase planets"""
    F = np.zeros(len(Time))
    for planet_number in range(len(params_transitions) // 4):
        gradient_and_profile_with_spline(Time, F, *(params_transitions[planet_number * 4:(planet_number + 1) * 4]), splines[planet_number])
    return F

def flux_grad_transitions(params_transitions, Time, splines):
    """flux impact of planets, excluding out of phase planets"""
    F = np.zeros(len(Time))
    gradient = []
    for planet_number in range(len(params_transitions) // 4):
        gradient.append(gradient_and_profile_with_spline(Time, F, *( params_transitions[planet_number * 4:(planet_number + 1) * 4]), splines[planet_number]))
    return F, np.concatenate(gradient)

def flux_nodes(params_nodes, N):
    """flux impact of star. N = len(Time), params_nodes = (x_1, x_2, ..., y_1, y_2, ...) ; s_k = x_k + i y_k"""
    number_of_fourier_nodes = len(params_nodes)//2
    return np.fft.irfft(np.insert(params_nodes[: number_of_fourier_nodes] + 1j * params_nodes[number_of_fourier_nodes :], 0, 0.0), n=N)

def chi_transitions(params_transitions, Time, flux, Variance, splines, r):
    """fits planets at fixed fourier nodes, already accounted for in flux"""
    F = np.zeros(len(Time))
    gradF = np.zeros(shape=(len(params_transitions), len(flux)))
    for planet_number in range(len(params_transitions)//4):
        gradF[planet_number * 4:(planet_number + 1) * 4, :] = gradient_and_profile_with_spline(Time, F, *(params_transitions[planet_number * 4:(planet_number + 1) * 4]), splines[planet_number])
    difference = (F - flux) / Variance
    return np.sum(np.square(difference)), np.array([2 * np.dot(gradF[i, :], difference) for i in range(len(params_transitions))])

def chi_nodes(params_nodes, flux, Variance, Pk_double):
    """fits fourier nodes at fixed planet parameters, already accounted for in flux"""
    number_of_fourier_nodes = len(params_nodes)//2
    difference = (flux_nodes(params_nodes, len(flux)) - flux) / Variance #it should be np.sqrt(Variance) but Variance[i] can be just 1 or np.inf, so it doesnt matter
    inv_fourier_razlike = np.fft.ifft(difference)[1:number_of_fourier_nodes+1] #we eliminate the first fourier component
    grad_nodes = 4*np.concatenate((np.real(inv_fourier_razlike), -np.imag(inv_fourier_razlike))) + 4*(params_nodes / Pk_double)
    chi = np.sum(np.square(difference)) + 2*np.sum(np.square(params_nodes) / Pk_double)
    return chi, grad_nodes

def negative_2loglikelihood(residuals, Variance, distribution_parameters):
    return - 2*np.sum(np.log(noise.pdf_noise(residuals, *distribution_parameters))/Variance)

def original_chi2(residuals, Variance):
    return np.sum(np.square(residuals/Variance)) + len(Variance) * np.log(2*np.pi)

def chi2_model(residuals, params_nodes, Pk_double, Variance):
    return np.sum(np.square(residuals/Variance)) + 2*np.sum(np.square(params_nodes) / Pk_double)

def logp_model(residuals, params_nodes, Pk, Variance, distribution_parameters):
    """returns -2log p of residuals"""
    Pk_double = np.concatenate((Pk, Pk))
    return 2*np.sum(np.square(params_nodes) / Pk_double) + negative_2loglikelihood(residuals, Variance, distribution_parameters)


def logp_model_sc(residuals, Variance):
    """-2 logp for a data with varying distribution parmeters"""
    logp = 0.0
    for era in range(np.shape(distribution_array)[0]):
        mask = Identificator == era
        logp += negative_2loglikelihood(residuals[mask], np.ones(np.sum(mask)), distribution_array[era])
    return logp


def exponential_kill_high_frequencies(freq, Pk):
    mask_fit = (freq< 0.5) & (freq>0.35)
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(freq[mask_fit]), np.log(Pk[mask_fit]))
    f = 1 / (1 + np.exp(-(freq - 0.5) / 0.03))
    Pk_new = np.exp(slope*np.log(freq) + intercept)
    Pk_new = np.cbrt(Pk_new, np.square(Pk))
    Pk_new[freq <0.3] = 0.0
    return Pk_new*f + Pk*(1-f)


def band_half_width_FourierGP(kji, i):
    """used e.g. for iterating hyperprior in new_hypeprior where Pk only up to some limit in frequency (2/day) is needed as this are the only frequency components that we fit"""
    return int(10 + 40 * np.log(1.0 + kji[i] / 0.3))

def band_half_width_Pk_extraction(kji, i):
    """used for extracting Pk from data for matched filter, where a full Pk is required (of length len(Time)/2 )"""
    return int(10 + 40 * np.log(1.0 + kji[i] / 0.3)  + 2000/(np.exp((3-kji[i])/0.3)+1))


def individual_band_smooth(kji, hiper, band_half_width):
    """smoothing hyperprior by mooving average method
    we use unequal width bins to get better S/N at higher freqencies. band_half_width is a function that gives half of a width of a bin as a function of frequency (k)"""

    smooth = np.zeros(len(hiper))
    #we want to find indeks_min such that indeks_min = band_half_width(indeks_min), which is done by iteration
    indeks_min = band_half_width(kji, 0)
    while indeks_min != band_half_width(kji, indeks_min):
        indeks_min = band_half_width(kji, indeks_min)
    indeks_maks = len(hiper)-1-band_half_width(kji, len(hiper)-1) #chosen in such a way that when averiging we do not have out of range
    for i in range(indeks_min, indeks_maks+1):
        band = band_half_width(kji, i)
        smooth[i] = np.average(hiper[i-band: i+band+1])

    for i in range(indeks_min):
        smooth[i] = smooth[indeks_min]
    for i in range(indeks_maks+1, len(hiper)):
        smooth[i] = smooth[indeks_maks]
    return smooth

def constant_band_smooth(hiper, band_half_width):
    """smoothing hiperprior by mooving average method"""
    smooth = np.zeros(len(hiper))
    indeks_min = band_half_width
    while indeks_min != band_half_width:
        indeks_min = band_half_width
    indeks_maks = len(hiper)-1-band_half_width #chosen in such a way that when averiging we do not have out of range
    for i in range(indeks_min, indeks_maks+1):
        smooth[i] = np.average(hiper[i-band_half_width: i+band_half_width+1])

    for i in range(indeks_min):
        smooth[i] = smooth[indeks_min]
    for i in range(indeks_maks+1, len(hiper)):
        smooth[i] = smooth[indeks_maks]
    return smooth


def get_Pk_rescaled(fluxFT, Variance):
    power_data = np.square(np.abs(fluxFT))
    kji = np.fft.rfftfreq(len(Variance), d=dt)
    #b, a = signal.butter(5, 0.1, 'low')  # got from https://dsp.stackexchange.com/questions/49460/apply-low-pass-butterworth-filter-in-python
    #Pk = signal.filtfilt(b, a, individual_band_smooth(kji, power_data, band_half_width_Pk_extraction))
    Pk = individual_band_smooth(kji, power_data, band_half_width_Pk_extraction)
    alfa = np.sum(1 / Variance) / len(Variance) #number of non-gaps / number of all data points
    Pk /= alfa
    return Pk[1:]


def Pk_gap_simulation(Pk, Variance):
    num_simulations = 10
    alfa = np.sum(1 / Variance) / len(Variance)
    Pk1 = np.zeros(len(Pk))
    for i in range(num_simulations):
        Pk1 += np.square(np.abs(np.fft.rfft(star_noise_simulation(Variance, Pk))[1:]))
    return Pk1/(alfa*num_simulations)

def get_Pk_gap_correction(fluxFT, Variance):
    Pk0 = get_Pk_rescaled(fluxFT, Variance)
    kji = np.fft.rfftfreq(len(Variance), d=dt)
    Pk1 = Pk0
    for i in range(3):
        #plt.plot(Pk1, label = str(i))
        Pk1 = individual_band_smooth(kji, np.abs(Pk0 + Pk1 - Pk_gap_simulation(Pk1, Variance)), band_half_width_Pk_extraction)
    return Pk1


def new_hyperprior(Variance, params_nodes, Pk, current_number_of_iterations = None):
    """computes hiperprior that will be used in the next iteration"""
    power_data = np.square(np.abs(np.fft.rfft(flux_nodes(params_nodes, len(Variance)))))
    kji = np.fft.rfftfreq(len(Variance), d=dt)
    kji = kji[1:1+len(params_nodes) // 2]

    #simulation
    power_simulation = np.average([power_spectrum_simulation(Variance, Pk) for i in range(10)], axis = 0)
    # it is sometimes better to try np.ones(len(Variance)) instead of Variance
    #new hiperprior
    new_hp = Pk + power_data[1:len(Pk)+1] - power_simulation
    # smoother new hiperprior
    b, a = signal.butter(5, 0.1, 'low') # got from https://dsp.stackexchange.com/questions/49460/apply-low-pass-butterworth-filter-in-python
    smooth_hiperprior = signal.filtfilt(b, a, individual_band_smooth(kji, new_hp, band_half_width_FourierGP))

    #plots graphs
    if current_number_of_iterations != None:
        plt.figure(figsize=(15, 10))
        plt.plot(kji, signal.filtfilt(b, a, individual_band_smooth(kji, power_data[1:len(Pk)+1], band_half_width_FourierGP)), '.', label = r'$<P_{data}>$', color = 'skyblue')
        plt.plot(kji, signal.filtfilt(b, a, individual_band_smooth(kji, power_simulation, band_half_width_FourierGP)), label = r'$<P_{sim}>$', color = 'violet')
       
#        plt.plot(kji, signal.filtfilt(b, a, power_data[1:len(Pk)+1]), '.', label = r'$<P_{data}>$', color = 'skyblue')
#        plt.plot(kji, signal.filtfilt(b, a, power_simulation), label = r'$<P_{sim}>$', color = 'violet')
        plt.plot(kji, Pk, label = 'hiper prior', color = 'green')
        plt.plot(kji, smooth_hiperprior, color='black', label='new hyper prior')
        plt.plot([kji[0], kji[-1]], [len(Variance), len(Variance)], color = 'brown', label = 'noise')
        #plt.errorbar((np.array(index)+1)*(kji[1]-kji[0]), pl ot_hiperprior, plot_sigma, fmt = 'o', color = 'black', label = 'new hiper prior')
        plt.yscale('log')
        plt.xlabel(r'$\nu [day^{-1}]$')
        plt.ylabel(r'$P(\nu)$')
        plt.legend()
        plt.savefig('hiperprior_band_60_180' + str(current_number_of_iterations))
        plt.close()

    #return smooth_hiperprior
    return smooth_hiperprior, power_simulation, power_data[1:len(Pk)+1]

def power_spectrum_simulation(Variance, Pk):
    """simulates power spectrum of star and noise"""
    Pk_double = np.concatenate((Pk, Pk))
    phases = np.random.uniform(0, 2*np.pi, len(Pk))
    nodes0 = np.sqrt(Pk_double)*np.concatenate((np.cos(phases), np.sin(phases)))
    flux_noise = np.random.normal(size = len(Variance))
    flux_simulation = (flux_nodes(nodes0, len(Variance)) + flux_noise)/Variance
    nodes_simulation = optimize.fmin_l_bfgs_b(chi_nodes, x0=nodes0, args=(flux_simulation, Variance, Pk_double))[0]
    power_simulation = np.square(nodes_simulation[:len(Pk)]) + np.square(nodes_simulation[len(Pk):])
    return power_simulation


def star_noise_simulation(Variance, Pk, nongaussian = False):
    """simulates star + noise signal, Pk is hyperprior on star variability and flat at high frequencies which is stationary noise"""
    Pk_double = np.concatenate((Pk, Pk))
    phases = np.random.uniform(0, 2 * np.pi, len(Pk))
    nodes0 = np.sqrt(Pk_double) * np.concatenate((np.cos(phases), np.sin(phases)))
    if nongaussian:
        flux= flux_nodes(nodes0, len(Variance))
        #average, sigma = prepare_data.normalization(flux)
        #flux /= sigma
        mask = np.random.random(len(flux)) < distribution_parameters[0]
        outliers = stats.nct.rvs(*distribution_parameters[1:], size=np.sum(mask))
        flux[mask] = outliers
        return flux / Variance
    else:
        return (flux_nodes(nodes0, len(Variance))) / Variance


def planet_simulation(Time, Variance, SNR, spline, period_given, phase_given, time_half_transit_given, Pk):
    """simulates planet flux with given S/N
    returns simulated flux and parametrs of simulated planet"""

    #parameters of the planet
    period = np.random.uniform(7, 100) if (period_given == None) else period_given
    time_half_transit = (q_Kepler90 * np.cbrt(period)) if (time_half_transit_given == None) else time_half_transit_given
    phase = np.random.uniform(time_half_transit * 1.1, period - time_half_transit * 1.1) if (phase_given == None) else phase_given


    #simulate the planet
    params = [period, phase, time_half_transit, -1]
    flux_planet = flux_transitions(params, Time, [spline,  ])/Variance
    #norm flux of a planet to a desired SNR
    if Pk is not None:
        #frequency dependant noise
        template_fourier_domain = np.fft.rfft(flux_planet)[1:]
        norm = 2 * np.sum(np.square(np.abs(template_fourier_domain)) / Pk)
    else:
        norm = np.sum(np.square(flux_planet))
    flux_planet *= (SNR/np.sqrt(norm))
    params[-1] *= (SNR/np.sqrt(norm))
    return flux_planet, params
