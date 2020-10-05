import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import pandas as pd
plt.rc('text', usetex=True)
from scipy.integrate import quad
from scipy.integrate import simps
from scipy.stats import norm
from math import erf
from scipy.stats.distributions import t
from scipy.stats import nct as noncentralT
from scipy import stats
from scipy import interpolate
import seaborn as sns
sns.set(style="ticks", palette="muted")
import pickle
#from pynverse import inversefunc

from constants import *
if not remote:
    import prepare_data

#spline for gradient of noise PDF
loaded_derivative = np.load(home+'noise_analysis/derivative_noncentralT_including_star.npy')
spline_derivative = loaded_derivative.item()['spline']
asymptote_derivative = loaded_derivative.item()['asymptote']

#spline for inverse of G
with open(home+ 'noise_analysis/spline_inverse.pkl', 'rb') as f:
    spline_inverse = pickle.load(f)


def make_bins(X, Y, binsize, errors = False):
    """returns bins with data averaged inside bin, binsize is size of bins in indexes (integer)"""
    if errors:
        return (np.array([np.average(X[i*binsize:(i+1)*binsize]) for i in range(len(X)//binsize)]), np.array([np.average(Y[i*binsize:(i+1)*binsize]) for i in range(len(Y)//binsize)]), np.array([np.std(Y[i*binsize:(i+1)*binsize])/np.sqrt(binsize) for i in range(len(Y)//binsize)]))
    else:
        return (np.array([np.average(X[i * binsize:(i + 1) * binsize]) for i in range(len(X) // binsize)]), np.array([np.average(Y[i * binsize:(i + 1) * binsize]) for i in range(len(Y) // binsize)]))


def autocorr(x):
    """autocorelation of x"""
    result = np.fft.irfft(np.square(abs(np.fft.rfft(x))), len(x))#sc.signal.correlate(x, x, mode='full', method = 'fft')
    return result#[result.size//2:]

def autocorr_with_gaps(x, invvar):
    lenn = len(x)
    return [np.sum(x[:lenn-tind]*invvar[:lenn-tind]*x[tind:]*invvar[tind:]) / np.sum(invvar[:lenn-tind]*invvar[tind:]) for tind in range(len(x))]


def Power_spectrum(Flux, binsize = 1, errors = False):
    """computes power spectrum |fft(Flux)|^2 and optionaly joins in bins"""
    flux = Flux-np.average(Flux)
    spectrum = np.fft.rfft(flux) #returns only positive k, as negative ones are symetric
    freq = np.fft.rfftfreq(len(Flux), d = dt)
    if binsize == 1:
        freq, np.square(abs(spectrum))
    return make_bins(freq, np.square(abs(spectrum)), binsize, errors)


def lin_plot(x, y, xlabel = '', ylabel = '', fig_name = None):
    """plots linear fit"""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.figure(figsize=(15, 15))
    plt.plot([x[0], x[-1]], [x[0] * slope + intercept, x[-1] * slope + intercept], ':')
    #plt.title(r'slope: {0}, intercept: {1}, slope error: {2}'.format(slope, intercept, std_err), fontsize = 22)
    plt.title(r'$\sigma_P = A P^k, \; k = {0}, \; A = {1}$'.format(slope, np.exp(intercept)), fontsize=22)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.plot(x, y, '.')
    if fig_name != None:
      plt.savefig(fig_name)
    plt.show()

def mybins(flux):
    ordered = np.sort(flux)
    N = len(flux)
    num_out = 30
    num_med = N//500
    avg_out = 10
    avg_med = 30
    avg_main = 1000
    return np.concatenate((ordered[:20:5], ordered[20:num_med: avg_med], ordered[num_med:-num_med:avg_main], ordered[-num_med: -num_out: avg_med], ordered[-num_out::avg_out]))


def fit_pdf(concatenated_fluxes):
    """fits pdf noise model and returns parameters of distribution"""
    N = norm.pdf(concatenated_fluxes, scale=1.0)

    def minus_logp_nc(parameters):
        """noise is modeled as a combination of gaussian majority and noncentral student t distribution modeling outliers
        fit is performed with no gradient as gradient of noncentral T is complicated. We use Powell method which does not accept bounds,
        therefore parameters are bijected on a suitable interval"""
        A = 0.5 * np.exp(-np.abs(parameters[0]))  # (-inf, inf) -> (0, 0.5]
        probabilities = (1 - A) * N + A * noncentralT.pdf(concatenated_fluxes, *np.abs(parameters[1:]))
        return - np.sum(np.log(probabilities))

    initial_guess = [4.09055969, 1.60403835, 0.85777229, 1.93993457, 0.74511898]
    optimize = sc.optimize.minimize(minus_logp_nc, initial_guess, method='Powell')
    params = optimize['x']
    params = np.abs(params)
    params[0] = 0.5 * np.exp(-np.abs(params[0]))
    return params

def fit_pdf_per_era():
    eight_planet_id = 11442793
    file = 'kepler_short' + str(eight_planet_id)
    df = pd.read_csv('../joint_fit_and_preprocessing//data/'+file+'.txt')
    Identificator = np.array(df["Identificator"])
    flux = np.load('../joint_fit_and_preprocessing//flux_ttv_sc.npy')
    num_eras = Identificator[-1] + 1
    distribution_array = np.zeros(shape = (num_eras, 5))
    t = np.linspace(-15, 15, 100)
    for era in range(num_eras):
        distribution_array[era, :] = fit_pdf(flux[Identificator == era])
        plt.plot(t, pdf_noise(t, *distribution_array[era, :]), label = str(era))

    np.savetxt('distribution_array.txt', distribution_array)
    plt.legend()
    plt.yscale('log')
    plt.show()

def histogram(y, bins):
    p_cel, x = np.histogram(y, bins=bins)
    p = p_cel / ((x[1:] - x[:-1]) * len(y))
    #p_err = np.sqrt(p_cel) / ((x[1:] - x[:-1]))
    #x_edges = np.copy(x)
    x = 0.5 * (x[1:] + x[:-1])
    return x, p

def plot_pdf(idji):
    """fits our model of noise and plots fancy graphs"""
    lower = -10
    upper = 20
    concatenated_fluxes = np.concatenate(np.array([pd.read_csv('C:/Users/USER/Documents/Physics/kepler/non_planets_candidates/including_star/' + str(id) + '.txt')["Flux"] for id in idji]))
    #concatenated_fluxes = np.load('../joint_fit_and_preprocessing//noise2.npy') #for Kepler 90 only
    #invVariance = np.loadtxt('../matched_filter//search_periodogram/inverse_Variance.txt')
    #concatenated_fluxes = np.load('../joint_fit_and_preprocessing//search_flux.npy') [Variance > 0.5]#star + noise for Kepler 90
    #concatenated_fluxes = func.flux_nodes(np.load('../joint_fit_and_preprocessing//params_nodes.npy'), len(invVariance))[invVariance > 0.5]

    avg, sigma = prepare_data.normalization(concatenated_fluxes)
    concatenated_fluxes = (concatenated_fluxes-avg)/sigma
    maska = (concatenated_fluxes > lower) & (concatenated_fluxes < upper)
    my_bins = mybins(concatenated_fluxes[maska])
    x, p = histogram(concatenated_fluxes[maska])

    N = norm.pdf(concatenated_fluxes, scale=1.0)

    initial_guess = [ 4.09055969,  1.60403835,  0.85777229,  1.93993457,  0.74511898]
    params = fit_pdf(concatenated_fluxes)
    print(params)
    # ploting histogram
    plt.rcParams['xtick.labelsize']=15
    plt.rcParams['ytick.labelsize']=15
    plt.figure(figsize=(15, 10))

    t = np.linspace(lower, upper, 1000)
    mask = norm.pdf(t) > 1e-6
    #plt.errorbar(x[:-1], p[:-1] , p_err[:-1], fmt = 'o', capsize = 2, color='black', label='noise distribution')
    zero = my_bins[0]
    my_bins[0] = -np.inf
    my_bins[-1] = np.inf
    #integrals = len(concatenated_fluxes) * np.array([quad(pdf_noise, my_bins[i], my_bins[i+1], args = (params[0], params[1], params[2], params[3], params[4])) for i in range(len(x))])[:, 0]/ ((x_edges[1:] - x_edges[:-1]) )
    ymin = 0.3
    #plt.bar(x[:-1], integrals[:-1], width = x_edges[1:-1] - x_edges[:-2], bottom = ymin, edgecolor = 'royalblue', color = 'royalblue', label='model')
    #plt.step(x_edges[:-1], integrals, where = 'pre')
    plt.plot(x, p, '.', color='black', label='noise distribution')
    plt.plot(t, pdf_noise(t, *params) , color = 'blue', label = 'model')
    plt.plot(t[mask], norm.pdf(t[mask]), ':', label = 'Gaussian contribution')

    plt.yscale('log')
    plt.xlabel(r"$n / \sigma$", fontsize = 19)
    plt.ylabel(r"$p(n)$", fontsize = 19)
    plt.legend( fontsize = 17, loc = 2)
    plt.grid(False)
    #plt.xlim(-7, 8.9)
    #plt.xlim(-15, 25)
    #plt.title('Kepler 90', fontsize = 30, pad = 20)
    #plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    #plt.yticks([])
    #plt.ylim(ymin, 33851)
    #plt.ylim(1e-6, 1)
    #plt.savefig('noise_pdf_kepler90.svg')
    plt.savefig("noisePDF_including_star")
    plt.show()


def pdf_noise(x, A, df, nc, loc, scale):
    """noise is modeled as a combination of gaussian majority and non central student t distribution model fot outliers"""
    return norm.pdf(x) + A * noncentralT.pdf(x, df, nc, loc= loc, scale = scale)


def derivative_logp(x, Variance, A, df, nc, loc, scale):
    """pointwise derivative with respect to x returns array of shape (len(x)). It is correct only at the fixed values of df, nc and scale, precomputed"""
    N = norm.pdf(x)
    pdf = (1.0 - A) * N + A * noncentralT.pdf(x, df, nc, loc=loc, scale=scale)
    #return np.sum(np.square(x)/Variance), -2*( (-x) )/(Variance)
    return -2*np.sum(np.log(pdf)/Variance), -2*((1 - A) * N * (-x) + A * derivative_noncentralT(x, loc))/(pdf*Variance)


def prepare_derivative_noncentralT_pdf(df, nc, loc, scale):
    """prepares spline representation of derivative for use in function derivative_noncentralT"""

    #asymptotic behaviour
    x = np.linspace(50, 100, 1000)
    slope_plus, intercept_plus, r_value, p_value, std_err = stats.linregress(np.log(x), np.log(noncentralT.pdf(x, df, nc, loc= 0.0, scale = scale)))
    slope_minus, intercept_minus, r_value, p_value, std_err = stats.linregress(np.log(x), np.log(noncentralT.pdf(-x, df, nc, loc=0.0, scale=scale)))
    min, max, num = -20.0, 20.0, 1000

    #spline fit of the main part
    x = np.linspace(min, max, num)
    dx = (max-min)/num
    values = noncentralT.pdf(x, df, nc, loc= loc, scale = scale)
    derivatives = (values[1:] - values[:-1])/dx
    real_x = x[:-1] + dx*0.5
    spline = interpolate.splrep(real_x, derivatives)

    #save results
    np.save('derivative_noncentralT_including_star.npy', {'spline': spline, 'asymptote': [[slope_plus-1.0, np.exp(intercept_plus)*slope_plus], [slope_minus-1.0, -np.exp(intercept_minus)*slope_minus]]})


def derivative_noncentralT(x, loc):
    """calculate derivatrive of noncentral t distribution with a pre prepared spline representation and asymptotic coefficients"""
    derivative = np.empty(len(x))
    mask_minus = x< -20
    mask_plus = x > 20
    mask_central = (x> -20) & (x< 20)

    derivative[mask_plus] = asymptote_derivative[0][1] * np.power(x[mask_plus]-loc, asymptote_derivative[0][0])
    derivative[mask_minus] = asymptote_derivative[1][1] * np.power(-x[mask_minus]+loc, asymptote_derivative[1][0])
    derivative[mask_central] = interpolate.splev(x[mask_central], spline_derivative)

    return derivative


def G(flux):
    """transform CDF of a noncentral Student T distribution to a gaussian one"""
    return norm.ppf((1-distribution_parameters[0]) * norm.cdf(flux) + distribution_parameters[0] * noncentralT.cdf(flux, distribution_parameters[1], distribution_parameters[2], loc= distribution_parameters[3], scale = distribution_parameters[4]) )

def derivative_G(flux, A, df, nc, loc, scale):
    return pdf_noise(flux, A, df, nc, loc, scale)/norm.pdf(norm.ppf((1-A) * norm.cdf(flux) + A * noncentralT.cdf(flux, df, nc, loc= loc, scale = scale) ))

def inverse_G(x):
    derivative = np.empty(len(x))
    min = -4.29391861627
    max = 3.73338275904
    mask_minus = x< min
    mask_plus = x > max
    mask_central = (x> min) & (x< max)

    derivative[mask_plus] = inverse_G_slow(x[mask_plus])
    derivative[mask_minus] = inverse_G_slow(x[mask_minus])
    derivative[mask_central] = interpolate.splev(x[mask_central], spline_inverse)

    return derivative

inverse_G_slow = inversefunc(G) #transform gaussian CDF to a CDF of a noncentralT distribution

def prepare_inverse_G():
    #this spline will work for the interval [-10, 15] in t. In G(t) it will therefore work on the interval [G(-10), G(15)].
    # Copy the printed values and set them as borders for the mask in inverse_G
    print(G(-10), G(15), sep = ', ')
    t = np.linspace(-10.1, 15.1, 1000)
    Gt = G(t)
    spline = interpolate.splrep(Gt, t)

    with open('spline_inverse.pkl', 'wb') as f:
        pickle.dump(spline, f)

def probability_x_is_outlier(x, A, df, nc, loc, scale):
    PN, PO = norm.pdf(x), noncentralT.pdf(x, df, nc, loc=loc, scale=scale)
    return A*PO/(A*PO + (1-A)*PN)

def g_2point(x, y, sigma_2point):
    return (G(x)-x) * np.exp(-0.5*np.square(y/sigma_2point)) + x

def G_2point(x):
    Gs = G(x)
    sigma_2point = 3
    gplus = np.empty(len(x))
    gminus = np.empty(len(x))
    gplus[:-1] = (Gs[:-1]-x[:-1]) * np.exp(-0.5*np.square(x[1:]/sigma_2point)) + x[:-1]
    gminus[1:] = (Gs[1:]-x[1:]) * np.exp(-0.5*np.square(x[:-1]/sigma_2point)) + x[1:]
    gplus[-1] = Gs[-1]
    gminus[0] = Gs[0]

    return np.sign(Gs)*np.sqrt(np.abs(gplus*gminus))

def G_3point(x):
    Gs = G(x)
    sigma_2point = 5
    G3 = np.empty(len(x))
    G3[1:-1] = (Gs[1:-1]-x[1:-1]) * np.exp(-0.5*(np.square(x[2:])+ np.square(x[:-2]))/np.square(sigma_2point)) + x[1:-1]
    G3[0] = (Gs[0]-x[0]) * np.exp(-0.5*(np.square(x[1]))/np.square(sigma_2point)) + x[0]
    G3[-1] = (Gs[-1] - x[-1]) * np.exp(-0.5 * (np.square(x[-2])) / np.square(sigma_2point)) + x[-1]
    return G3

def G_filter_correlated_structures(x):
    Gs = G(x)
    P_N = 1-probability_x_is_outlier(x, *distribution_parameters)
    G3 = np.empty(len(x))
    G3[1:-1] = P_N[:-2]*P_N[2:]*Gs[1:-1] +  (1-P_N[:-2]*P_N[2:])*x[1:-1]
    G3[0] = P_N[1]*Gs[0] + (1-P_N[1])*x[0]
    G3[-1] = P_N[-2]*Gs[-1] + (1-P_N[-2])*x[-1]
    return G3

def fermi(x):
    return 1/(1+np.exp(0.5*(x-4)))


def G_3point_fermi(x):
    delta = np.empty(len(x)+1)
    delta[1:-1] = x[1:]-x[:-1]
    delta[0], delta[-1] = 0, 0
    f = fermi(-delta[:-1]*delta[1:])
    return (1-f)*G(x) + f*x


def fit_CDF_transformation(params0):
    plt.plot(t, np.square(G(t, *params0)), label = r'$G^2$')
    plt.plot(t, -2*np.log( pdf_noise(t, *params0)) -np.log(2*np.pi), label=r'$-2\log(p) - \log(2 \pi)$')
    plt.legend()
    plt.show()

# def transform_to_real(flux, A, df, nc, loc, scale):
#     return norm.ppf((1-A) * norm.cdf(flux) + A * noncentralT.cdf(flux, df, nc, loc= loc, scale = scale) )


def simulate_noise(N, params):
    """Generates noise signal according to our noise distribution = Gauss + noncentral Student`s t
    N = number of realizations
    params =[A, df, nc, loc, scale] see: https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.stats.nct.html
    """
    N_outliers = int(params[0]*N) #number of outliers
    Gaussian = np.random.normal(loc = 0.0, scale = 1.0, size= N - N_outliers)
    outliers = noncentralT.rvs(*params[1:], size=N_outliers)
    fluxi = np.concatenate([Gaussian, outliers])
    np.random.shuffle(fluxi)
    return fluxi


def chi_signal_to_noise():
    """for converting results of chi_spline and other similar function in fitting_planets.py in latex friendly shape"""
    chi_spline_gauss = [16.202680582309682, 14.011573548213033, 4.937760946607105, 35.12756866904892, 28.550660519665183, 24.583186752459223, 224.10534489974526, 336.62763892071547]
    chi_spline = [16.80341870360947, 15.464438227214043, 5.35832167563199, 35.11130001160402, 28.580385245520812, 24.60843476071911, 61.57172467926813, 49.943404842]
    chi_my = [15.3288016017, 15.6415956903, 6.61819300763, 29.1440319388, 23.1048723094, 16.3915595901, 143.829731113, 114.006262605]
        #np.sqrt([238.678656802, 255.029853771, 45.2424919452, 903.884209509, 570.199216419, 287.588560268, 19990.0144827, 12489.5771445])
    chi_official = [16.7, 16.6, 0, 18.9, 25.1, 30.2, 121.8, 233.3]
    R_my = [1.28998515,  1.35579144,  1.02016117,  2.8340451,   2.72054549,  2.69455491, 7.72448005,  10.82475816]
    R_spline = [1.21846978,  1.22169667,  0.80754951,  2.63064171,  2.48780238,  2.70069302, 7.72301198,  10.81073073]
    d = {'planet': ['b', 'c', 'i', 'd', 'e', 'f', 'g', 'h'], 
         'splineGauss': np.round([chi for chi in chi_spline_gauss], 1),
         'splineP': np.round([chi for chi in chi_spline], 1),
         'myR': np.round([chi for chi in R_my], 2),
         'splineR': np.round([chi for chi in R_spline], 2),
         'myP': np.round([chi for chi in chi_my], 1),
         'official': np.round([chi for chi in chi_official], 1)}
    df = pd.DataFrame(data=d)
    df.to_csv('deltalog.csv', index=False)

def prior(tau, tauK):
    return tau / (np.square(tauK) * np.sqrt(1-np.square(tau/tauK)))


def prior_with_error_on_tauk(tau, mu, sigma):
    deg = 100
    a, c = (mu-tau)*0.5, (mu+tau)*0.5
    def f_prohibited_tau(u):
        tk = mu + np.sqrt(2*(sigma**2)*u + np.square(tau -mu))
        return np.power(u, 0.5)/((tk - mu) * tk * np.sqrt(np.square(tk) - np.square(tau)))
    def f_upper(u):
        tk = mu + sigma*np.sqrt(2 * u )
        return np.power(u, 0.5) / ((tk - mu) * tk * np.sqrt(np.square(tk) - np.square(tau)))
    def f_lower(x):
        tk = a*x + c
        return np.sqrt(1+ (tk-c)/a)*prior(tau, tk)*np.exp(-0.5*np.square(tk-mu)/sigma**2)

    def f_prohibited_tau_closemu(u):
        tk = mu + np.sqrt(2*(sigma**2)*u + np.square(tau -mu))
        return np.power(u, 0.75)/((tk -mu) * tk * np.sqrt(np.square(tk) - np.square(tau)))
    # def f_upper_closemu(u):
    #     tk = mu + sigma*np.sqrt(2 * u )
    #     return np.power(u, 0.75) / ((tk - mu) * tk * np.sqrt(np.square(tk) - np.square(tau)))
    # def f_lower_closemu(x):
    #     tk = a*x + c
    #     return np.power(1+ (tk-c)/a, 0.75) *prior(tau, tk)*np.exp(-0.5*np.square(tk-mu)/sigma**2)


    xL, wL = sc.special.roots_genlaguerre(deg, alpha = -0.5) #gauss-laguerre quadrature
    xJ, wJ = sc.special.roots_jacobi(deg, alpha = 0, beta = -0.5) #gauss-jacobi quadrature
    xLc, wLc = sc.special.roots_genlaguerre(deg, alpha= -0.75)  # gauss-laguerre quadrature
    # xJc, wJc = sc.special.roots_jacobi(deg, alpha=0, beta= -0.75)  # gauss-jacobi quadrature

    if tau > mu:
        return np.sum(wL * f_prohibited_tau(xL)) * sigma * tau* np.exp(-np.square(tau-mu)/(2*sigma**2)) / np.sqrt(2*np.pi)

    else:
        return  np.sum(wL * f_upper(xL)) * sigma *tau / np.sqrt(2*np.pi) + np.sum(wJ * f_lower(xJ))*a/(np.sqrt(2*np.pi) * sigma)


def prior_with_error_on_tauk_at_mu(mu, sigma):
    deg = 100

    def f_prohibited_tau_closemu(u):
        tk = mu + np.sqrt(2*(sigma**2)*u)
        return np.power(u, 0.75)/((tk -mu) * tk * np.sqrt(np.square(tk) - np.square(mu)))


    xLc, wLc = sc.special.roots_genlaguerre(deg, alpha= -0.75)  # gauss-laguerre quadrature

    return np.sum(wLc * f_prohibited_tau_closemu(xLc)) * sigma * mu / np.sqrt(2 * np.pi)


def corrected_prior(tau, tauK, sigma_tauK, f1, f2, f3, f4, weight_sigma):
    p0 = np.array([prior_with_error_on_tauk(t, tauK, sigma_tauK) for t in tau])
    p00 = prior_with_error_on_tauk_at_mu(tauK, sigma_tauK)
    weights = np.exp(-0.5*np.square(tau-tauK)/weight_sigma**2)
    return p0*(1.0-weights) + (p00 + f1 * (tau-tauK) + f2 * np.square(tau-tauK) + f3 *np.power(tau-tauK, 3) + f4* np.power(tau-tauK, 4))*weights


def fit_corrected_prior(tauK, sigma_tauK, weight_sigma):
    tau = np.concatenate((np.linspace(tauK-3*weight_sigma, tauK-1.5*weight_sigma, 200), np.linspace(tauK+1.5*weight_sigma, tauK+3*weight_sigma, 200)))
    p = np.array([prior_with_error_on_tauk(t, tauK, sigma_tauK) for t in tau])
    return sc.optimize.curve_fit(lambda t, x1, x2, x3, x4: corrected_prior(t, tauK, sigma_tauK, x1, x2, x3, x4, weight_sigma), tau, p, p0 = [-2, -7, 0, 0])[0]

def prepare_spline_prior(sigma_tauK):
    tau = np.linspace(0, 4, 100)
    weight_sigma = 0.5 * sigma_tauK
    taylor_coefficients = fit_corrected_prior(1.0, sigma_tauK, weight_sigma)
    f1, f2, f3, f4, = taylor_coefficients[0], taylor_coefficients[1], taylor_coefficients[2], taylor_coefficients[3]
    p = corrected_prior(tau, 1.0, sigma_tauK, f1, f2, f3, f4, weight_sigma)
    return interpolate.splrep(tau, p)

def plot_prior():
    tauK, sigma_tauK= 1.0, 0.15
    tau = np.linspace(0, 1.6, 1000)
    spline_prior = prepare_spline_prior(sigma_tauK)
    p = interpolate.splev(tau, spline_prior)

    print(np.sum(p)*(tau[1]-tau[0]))

    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    f = 22
    plt.figure(figsize=(15, 10))
    plt.plot(tau, p, color = 'yellowgreen', label = r'$\sigma_{\tau_K} / \tau_K = 0.15$')
    noerr = np.zeros(len(tau))
    mask = tau < tauK
    noerr[mask]= tau[mask]/(np.square(tauK)*np.sqrt(1-np.square(tau[mask]/tauK)))
    print(np.sum(noerr) * (tau[1] - tau[0]))
    plt.plot(tau, noerr, color = 'royalblue', label = r'$\sigma_{\tau_K} = 0$')
    plt.title('a)', fontsize = f)
    plt.xlabel(r'$\tau/\tau_K$', fontsize = f)
    plt.ylabel(r'$p(\tau)$', fontsize = f)
    plt.ylim(-0.1, 2.6)
    plt.xlim(0, 1.5)
    plt.legend(fontsize = f)
    plt.savefig('TauPrior.pdf')
    plt.show()



if __name__ == '__main__':
    #candidates = np.loadtxt("C:/Users/USER/Documents/Physics/kepler/non_planets_candidates/non_planets_ids.txt", dtype = int)
    #plot_pdf(candidates)
    #fit_pdf_per_era()
    #fit_CDF_transformation(func.distribution_parameters)
    #chi_signal_to_noise()
    #derivative_noncentralT(np.linspace(-30, 30, 10000), 1.93993457)
    #prepare_derivative_noncentralT_pdf(*distribution_parameters[1:])
    plot_prior()

    #print(quad(lambda x: pdf_noise(np.tan(x), *distribution_parameters) / np.square(np.cos(x)), np.arctan(3), np.pi/2))
    # print(quad(lambda x: pdf_noise(np.tan(x), *distribution_parameters) / np.square(np.cos(x)), -np.pi / 2, -np.arctan(3)))
