import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
plt.rc('text', usetex=True)


from constants import *
import fitting_planets_functions as func
from noise_analysis import noise
import transit_search as ts
from matplotlib.patches import Ellipse

def make_sound(num_times = 3):
    import winsound
    import time
    t = time.localtime()[3]
    if t > 7 and t< 23:
        for i in range(num_times):
            winsound.PlaySound("SystemExit", winsound.SND_ALIAS)

def first_fit_planets(Time, Flux, Variance):
    """seperatly fit just planets to get initial estimate of parameters"""
    splines = func.prepare_splines(r)
    flux_planets = np.zeros(len(Time))
    params_transitions = Kepler90_planets_parameters[:]
    for p in range(6-1, -1, -1):
            #params_transitions[p * 4: (p + 1) * 4] = planet_fit_loglikelihood(params_transitions[p*4: (p+1)*4], Time, flux_brez_nodes-flux_planets, splines[p], Kepler90_planets_radius[p], Variance)
            params_transitions[p * 4: (p + 1) * 4] = func.planet_fit(params_transitions[p * 4: (p + 1) * 4], Time, Flux - flux_planets, Variance, splines[p], Kepler90_planets_radius[p])
            flux_planets += func.profile_with_spline(Time, *params_transitions[p*4: (p+1)*4], splines[p], Kepler90_planets_radius[p])
    plt.plot(Time, flux_planets)
    print(params_transitions)
    

def zero_planet(Time, Flux, Variance, period, start, end):
    """eliminates flux where there was a planet. That is:
        Flux[k*period + start: k*period + end] = 0
        Variance[k*period + start: k*period + end] = inf
        for k natural number"""
    ne_prehodna_tocka = np.array([not (math.fmod(t, period) > start and math.fmod(t, period)<end) for t in Time]) #true if a point is not during a time of transit
    flux = np.copy(Flux)
    flux[~ne_prehodna_tocka] = 0.0
    variance = np.copy(Variance)
    variance[~ne_prehodna_tocka] = np.inf
    return flux, variance




def discard_all(time, flux, variance, params):
    """discards flux for all planets (see discard_planet)"""
    planetov = len(params)//4
    flux0 = np.copy(flux)
    variance0 = np.copy(variance)
    for i in range(planetov):
        flux0, variance0 = discard_planet(time, flux0, variance0, params[4*i], params[4*i + 1] - 1.1*params[4*i+2], params[4*i + 1] + 1.1*params[4*i+2])
    return flux0, variance0


def discard_out_of_phase_planets(time, flux, variance):
    """discards big planets that are out of phase. This function is no longer needed because it is better to use func.out_of_phase to fit thoose transits instead"""
    #drugi planet
    #flux0, variance0 = discard_planet(time, flux, variance, time[-1], 1279.93, 1280.52) #zadnji prhod drugega planeta
    #flux0, variance0 = discard_planet(time, flux0, variance0, time[-1], 1278.88, 1279.41) #tam kjer bi moral biti zadnji prehod drugega planeta, da ne unicimo fita amplitude
    #flux0, variance0 = discard_planet(time, flux0, variance0, time[-1], 225.762, 226.45 ) #drugi prehod drugega plneta, ker je za cpp iz faze

    #prvi planet
    flux0, variance0 = discard_planet(time, flux, variance, time[-1], 1335.05, 1336.3) #zadnji prehod prvega planeta
    return flux0, variance0



def deltalogp_spline(Flux, Variance, params_transitions, r):
    """computes chi^2(without given planet) - chi^2 (all planets) for all planets with parameters given in x0_transitions
    number of planets = len(r)"""
    spline = func.prepare_shape(r)
    params_transitions = func.planet_fit(params_transitions, Time, Flux, Variance, spline, r)
    flux_planets = func.profile_with_spline(Time, *params_transitions, spline, r)
    logp = func.negative_2loglikelihood(Flux, Variance, func.distribution_parameters) - func.negative_2loglikelihood(Flux - flux_planets, Variance, func.distribution_parameters)
    print(np.sqrt(logp))


def deltalogp_spline_Kepler90(Time, Flux, Variance, params_transitions0, r):
    """computes chi^2(without given planet) - chi^2 (all planets) for all planets with parameters given in x0_transitions
    number of planets = len(r)"""
    planets = len(r) #+ out of phase planets
    splines = func.prepare_splines(r)
    r_g, r_h = 0.04339823993170634, 0.06042657761708681
    spline_g = func.prepare_shape(r_g)
    spline_h = func.prepare_shape(r_h)
    params_g = [1.55885605e+01, 2.26041792e+02, 4.36769846e+02, 8.57960419e+02, 1.06854995e+03, 1.28022221e+03, 2.39409012e-01, -2.13050749e+01]
    params_h = [8.97, 340.63, 1335.38, 0.27662010037398799, -40.289663555792735]
    flux_planets = np.empty((planets + 2, len(Time)))
    flux_planets[-1, :], phases =  func.planet_out_of_phase(Time, Flux, Variance, [params_h, ], [spline_h, ], [r_h, ])
    #print(phases[0][-1])
    flux_planets[-2, :], phases = func.planet_out_of_phase(Time, Flux - flux_planets[-1, :], Variance, [params_g, ], [spline_g, ], [r_g, ])
    #print(phases[0][-1])
    params_transitions = np.copy(params_transitions0)
    for p in range(planets-1, -1, -1):
        params_transitions[p * 4: (p + 1) * 4] = func.planet_fit(params_transitions0[p * 4: (p + 1) * 4], Time, Flux - np.sum(flux_planets[p+1:, :], axis = 0), Variance, splines[p], Kepler90_planets_radius[p])
        flux_planets[p, :] = func.profile_with_spline(Time, *params_transitions[p*4: (p+1)*4], splines[p], Kepler90_planets_radius[p])

    #print(params_transitions[3::4])
    flux_no_planets = Flux - np.sum(flux_planets, axis = 0)
    chi2 = -np.ones(len(r) + 2) * func.original_chi2(flux_no_planets, Variance)
    logp = -np.ones(len(r) + 2) * func.negative_2loglikelihood(flux_no_planets, Variance, func.distribution_parameters)
    for planet in range(len(r) + 2): #assumes only two out of phase planets!!!
        chi2[planet] += func.original_chi2(flux_no_planets + flux_planets[planet, :], Variance)
        logp[planet] += func.negative_2loglikelihood(flux_no_planets + flux_planets[planet, :], Variance, func.distribution_parameters)
    print (np.ndarray.tolist(np.sqrt(chi2)))
    print (np.ndarray.tolist(np.sqrt(logp)))


def mark_planets():
    for planet in range(4, 5):
        tji = [Kepler90_planets_parameters[4*planet] * k + Kepler90_planets_parameters[4*planet+1] for k in range(int(Time[-1]/Kepler90_planets_parameters[4*planet]))]
        print([tji[i] for i in [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]])
        #[0, 5, 6, 8]
        return
        plt.scatter(tji, np.zeros(len(tji))+5, marker = 'x', label = str(planet))
    plt.legend()



def save_search_flux(Time, Flux, Variance):
    r_g, r_h = 0.04339823993170634, 0.06042657761708681
    spline_g = func.prepare_shape(r_g)
    spline_h = func.prepare_shape(r_h)
    splines = func.prepare_splines(r)
    search_flux = Flux - func.flux_nodes(np.load('params_nodes_semi_kill.npy'), len(Time))
    search_flux -= func.planet_out_of_phase(Time, Flux, Variance, np.load('out_of_phases_semi_kill.npy'), [spline_g, spline_h], [r_g, r_h], direct_return=True)
    search_flux -= func.flux_transitions(np.ndarray.tolist(np.load('params_planets_semi_kill.npy')), Time, splines)
    np.save('search_flux_semi_kill.npy', search_flux)

def moon_planet_posterior():
    """a collection of function calls for determening different posteriors"""

    # hypothetical planet -2logp(amplitude) dependance, when you fix amplitude and fit for a star. You probably need to change delta_logp_planets_individual_hyperpiror a little

    amplitudes = [-1.2, ]#np.linspace(-1, -0.3, 20)
    for i in range(len(amplitudes)):
        func.delta_logp_planets_individual_hyperpiror(Time, Flux, Variance, r, amplitudes[i])
    #func.logp_test(Time, Variance)
    func.likelihhoods_for_planets(Time, Flux, Variance, np.ndarray.tolist(np.load('params_planets.npy')), r)
    relative_array = [[147.1800207041652, 357.34423960223273, 568.4936635313701, 989.3012335918152, 1200.2765473760799, 1411.5287168865625], [147.17600783035675, 357.3472967430199, 568.4975052008849, 989.3050086670313, 1200.2778759392886, 1411.5298551492626],                      [147.17133033291873, 357.3502712805363, 568.5004222246264, 989.3069157765168, 1200.279884774266, 1411.5337142856183],                      [147.16744955504677, 357.35236170129815, 568.5021580011095, 989.3082658173748, 1200.2814490915855, 1411.5369179057432],                      [147.16451114604868, 357.3541541395761, 568.5035502569049, 989.3102558270682, 1200.2825499361102, 1411.5386362732308],                      [147.16105116663107, 357.35644878273706, 568.5056938556745, 989.314618603788, 1200.2835321570344, 1411.538385439292],                      [147.15764824817754, 357.3586612278954, 568.5071746009407, 989.3190625942891, 1200.2842209862276, 1411.5379605637982],                      [147.15408160736678, 357.35946279068776, 568.5060076361909, 989.3139218817747, 1200.2864699613958, 1411.5478583411198],                      [147.14931518609194, 357.36171044274244, 568.5072497379659, 989.3178249670405, 1200.2877463107795, 1411.548635403431],                      [147.14669780929023, 357.36327837686514, 568.5075247187342, 989.3189497038502, 1200.2887388495265, 1411.5512564541973],                      [147.14356512899693, 357.3643421544284, 568.5074626562845, 989.3196037172793, 1200.289727328866, 1411.5541214742275],                      [147.14061439499577, 357.36589644740036, 568.5076943265358, 989.3211208300289, 1200.2906412829507, 1411.5561632735369],                      [147.1365177934376, 357.36698419201986, 568.5072821627487, 989.3208172454083, 1200.2919207234222, 1411.561311409014],                      [147.1329576735349, 357.36926700439005, 568.5077502676436, 989.3252149390557, 1200.2927284767081, 1411.5602582056197],                      [147.12871246542807, 357.3693817519891, 568.5064939058368, 989.3229064545582, 1200.2938971040141, 1411.567661574595],                      [147.12521658383514, 357.37287797059616, 568.5075588588153, 989.3295725264093, 1200.2946878702753, 1411.5639359785548],                      [147.12207947816287, 357.37447977237935, 568.5073580233458, 989.3310794077864, 1200.2954614655807, 1411.5658102915356],                      [147.11711146298973, 357.3762104294686, 568.5067410491309, 989.3335860176942, 1200.2966300319583, 1411.5679190897672],                      [147.112231003331, 357.3768795617203, 568.5052638746372, 989.3308546680561, 1200.2978695036343, 1411.5771640186633],                      [147.10749637841275, 357.3791965822516, 568.5048057400166, 989.3338073265141, 1200.2988854936268, 1411.5785647486794],                      [147.1028687158151, 357.3802995721371, 568.5038242073665, 989.3337722819235, 1200.2997238100597, 1411.5833526448394],                      [147.10077540128182, 357.3823001777279, 568.5037689720373, 989.3365692180286, 1200.3003030058746, 1411.5826662662441],                      [147.09718083142056, 357.38355482475976, 568.503272334575, 989.3374835495454, 1200.3011648424363, 1411.5855575599714],                      [147.09365770866913, 357.3853560027712, 568.5025735070512, 989.3393800516326, 1200.3017178856012, 1411.5866963334286],                      [147.0909811195412, 357.3880996836511, 568.5023973993982, 989.3441552904627, 1200.302064714114, 1411.583043443445],                      [147.08708107555532, 357.3885264342949, 568.5014039163258, 989.3436139742358, 1200.302846186743, 1411.5875067491584],                      [147.08355466172262, 357.390107028125, 568.5007900155132, 989.3447573224778, 1200.3036161705213, 1411.5900434445516],                      [147.0794155592776, 357.3921568046861, 568.5001263870753, 989.347522693598, 1200.304329249546, 1411.590267643221],                      [147.07783425501063, 357.3917507726052, 568.4996001408388, 989.3441948678285, 1200.304605757668, 1411.597020498376],                      [147.0727854279144, 357.39387585902597, 568.4987132494463, 989.3457220306349, 1200.3054451825933, 1411.6001106255392],                      [147.0701023104548, 357.3955688868297, 568.4982217024757, 989.3478871762014, 1200.305903063036, 1411.5997645662367],                      [147.06631325669758, 357.3961035167148, 568.4974626501909, 989.3468083119026, 1200.306446173164, 1411.6050851955679],                      [147.0627459015303, 357.40026271271233, 568.4969773141984, 989.3546502692393, 1200.3072136400876, 1411.597586695466],                      [147.06121999834627, 357.39955077504555, 568.4966689219216, 989.3503625279074, 1200.3072676286133, 1411.6052672993808],                      [147.05821710579121, 357.40109359700796, 568.496084263644, 989.352329682606, 1200.3077559539418, 1411.6052442320424],                      [147.0546247157894, 357.40147650414497, 568.4954510424561, 989.3503375106848, 1200.308134428652, 1411.611669383912],                      [147.0520191809335, 357.40305933625837, 568.4951386225764, 989.3515390620587, 1200.3084981547374, 1411.612502052691],                      [147.04945074578782, 357.40485710122385, 568.494715617253, 989.3531286365134, 1200.308858249139, 1411.6127466638686],                      [147.046673024753, 357.40576721942983, 568.4943144127961, 989.3531999546467, 1200.3091674363945, 1411.6151306295153],                      [147.04397200364917, 357.40784055804465, 568.4939661361332, 989.3551511771942, 1200.3095177820367, 1411.614807252998],                      [147.0411797713967, 357.40851161921034, 568.493495293151, 989.3547863571476, 1200.3098009866826, 1411.6177752576941],                      [147.03851822223524, 357.4103139242676, 568.4931823253171, 989.3560100682776, 1200.3100845763693, 1411.618397738605],                      [147.03565399621564, 357.41141830320834, 568.4927730109918, 989.3562424458382, 1200.3103493216538, 1411.6204469090699],                      [147.03289232687715, 357.41318576957883, 568.4924534253142, 989.3573456614881, 1200.3106141266776, 1411.6211876576458],                      [147.03005136130517, 357.4144946424207, 568.4921688582062, 989.3576017278949, 1200.3108101436528, 1411.6230731606256],                      [147.02814137568393, 357.41671811885675, 568.4916497684884, 989.3611710496394, 1200.3111711566114, 1411.6193801096977],                      [147.02419850528952, 357.4173740513673, 568.4915208894039, 989.3585373454923, 1200.3112139325858, 1411.6261607339718],                      [147.0210008313856, 357.4194099335882, 568.4910353795748, 989.3602390077932, 1200.3114988581804, 1411.6259315850814],                      [147.01831124749128, 357.42101943449984, 568.4909551502328, 989.3602714633661, 1200.311560429326, 1411.6278650525728],                      [147.0150244117323, 357.4225421483596, 568.4905736692956, 989.360918347135, 1200.3117584748593, 1411.6290135932882]]
    #hypothetical moon -2logp(relative phases) dependance
    for i in range(len(relative_array)):
        relative_moon_phases = np.array(relative_array[i]) - np.array([147.10056470725027, 357.55634473813916, 568.2870891849191, 989.4804905763432, 1200.0697883584196, 1411.740763070372])
        func.moon_optimal_relative_phases(Time, Flux , Variance, planets, r, relative_moon_phases)

    # hypothetical moon -2logp(amplitude) dependance, when you fix amplitude and fit for a star.
    relative_moon_phases = np.array(relative_array[24]) - np.array([147.10056470725027, 357.55634473813916, 568.2870891849191, 989.4804905763432, 1200.0697883584196, 1411.740763070372])

    amplitudes = [0.0, ] #np.linspace(0.2, 0.9, 25)
    for i in range(len(amplitudes)):
        func.moon_optimal_relative_phases(Time, Flux, Variance, planets, r, relative_moon_phases, amplitudes[i])

def fit_with_real_time():
    real_Time = np.array(pd.read_csv('data/' + 'kepler' + str(eight_planet_id) + '.txt')["Time"])
    func.fit_out_of_phase_with_real_time(real_Time, Flux, Variance)

def planet_residuals(flux_given):
    #Flux = np.load('search_flux_semi_kill.npy')
    if not long_cadence:
        def planet_phases(Time, *parameters):
            """allows for TTV -> transit time variations (changing phase of transit)"""
            flux = np.zeros(len(Time))
            for i in range(len(parameters) - 2):
                flux += func.profile_with_spline(Time, Time[-1], parameters[i], parameters[-2], parameters[-1],
                                            splines_off[0])
            return flux

        params_off = [[8.57960419e+02, 1.06854995e+03, 1.28022221e+03, 2.39409012e-01, -2.13050749e+01/np.sqrt(15)],
         [1335.38, 0.27662010037398799, -40.289663555792735/np.sqrt(15)]]
        r_off = [0.04339823993170634, 0.06042657761708681]
        splines_off = func.prepare_splines(r_off)
        flux_planets, params = func.planet_out_of_phase(Time, flux_given, Variance, params_off, splines_off, r_off, type_of_fit = ['TTV', 'TTV'], direct_return=False)
        params_nodes = np.load('params_nodes_sc.npy')
        flux_star = func.flux_nodes(params_nodes, len(Time))
        Flux = flux_given-flux_planets-flux_star

    else:
        Flux = np.copy(flux_given)
        params_off = np.load('out_of_phases_semi_kill.npy')



    tau = params_off[0][-2]
    num_points = 3
    deltat = num_points*dt
    x, y, var = ts.fold_ttv(Time, Flux, Variance, params_off[0][:-2], deltat, 6 * tau)
    fig, ax = plt.subplots()
    plt.plot(x, y, '.')
    plt.title('Kepler 90g residuals (short cadence)')
    plt.ylabel('folded(flux - planet model - star, $\Delta t = ${0} min)'.format(np.round(deltat*24*60, 2)))
    plt.xlabel('t - t0 [days]')
    plt.plot([tau, tau], [-2, 2], ':', color='red')
    plt.plot([-tau, -tau], [-2, 2], ':', color='red')
    plt.ylim(-2, 2)
    circleL = Ellipse([-tau, 1], width = 0.02, height = 0.02*4/(6*tau), color='black')
    circleR = Ellipse([tau, 1], width=0.02, height=0.02 * 4 / (6 * tau), color='black')
    ax.add_artist(circleL)
    ax.add_artist(circleR)
    plt.savefig('Kepler90g_residulas_nostar_sc_' + str(num_points))
    plt.show()


if __name__ == '__main__':
    if long_cadence:
        file = 'zeros_kepler' + str(eight_planet_id)
    else:
        file = 'zeros_kepler_short' + str(eight_planet_id)
    print(file)

    df = pd.read_csv('data/'+file+'.txt')
    Flux = np.array(df["Flux"])
    Time = np.array(df["Time"])
    Variance = np.array(df["Variance"])

    #planet_residuals(Flux)

    spline = func.prepare_shape(0.04)

    print(np.load('out_of_phases_semi_kill.npy'))

    # flux, params = func.iterate_hyperprior(Time, Flux, Variance, np.ndarray.tolist(np.load('params_planets_semi_kill.npy')), Kepler90_planets_radius, 'out of phase')
    # np.save('extracted_planets', flux)

    #Flux = np.load('flux_star.npy')
    #func.prepare_flux_for_emcee(Time, Variance)
    #func.samples_from_posterior(Time, Variance)
    #func.marginalize_with_integral(Time, Variance)

    #func.radius_impact()
    #func.errors_planet_parameters(Time, Variance)
    #which_planet = 1
    #func.amplitude_test(Time, Variance, planets[4*which_planet], Kepler90_planets_radius[which_planet], names_planets[which_planet])#np.ones(len(Time)))
    #func.logp_test(Time, Variance, 210.59820446235335, 0.04339823993170634, 'g')


    #func.single_joint_fit(time, flux, variance, [time[-1], 141.225, 2*dt, -4.0], 0.01, Pk)
    #func.iterate_hyperprior(Time, Flux, Variance, None, None, None)#np.ndarray.tolist(np.load('params_planets_semi_kill.npy')), Kepler90_planets_radius, 'out of phase')

