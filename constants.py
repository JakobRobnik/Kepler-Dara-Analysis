###collected parameters of analysis and properties of Kepler 90 star###

####our analysis parameters###
remote = False
home = "C:\\Users\\USER\\Documents\\Physics\\machine\\" #basic directory, change this if you are working on another computer
if remote:
    home = 'C:\\Users\\jakob\\Documents\\machine\\'
long_cadence = True #if you change this to false check functions in fitting_planets_functions.py: profile_with_spline, gradient_and_profile_with_spline (and perhaps other gradients) to check if everything with T[0] is right, especially inside the inner loops
logp_planets = False #fit planets with nongaussian noise distribution in joint_fit

                            #A                  df            nc          loc          scale
#distribution_parameters = [ 0.00827352,  1.55134374,  0.86912289,  1.98435008,  0.71418219] #non-gaussian distribution parameters
distribution_parameters = [ 0.00775706,  1.54855715,  0.85056464,  2.00960796,  0.7098278 ] #this is including star
#distribution_array = np.loadtxt('C:\\Users\\USER\\Documents\\Physics\\machine\\noise\\distribution_array.txt')

dt = 0.0204334994196 #in days, sampling rate of the measurement = <Time[i+1] - Time[i]>
if not long_cadence:
    dt = 0.00068112

period_min, period_max = 3, 350 # limits of search
period_step = 0.1*dt
required_tau_precise = 0.1 * dt  # how dense templates with different time of transit will be used for periodogram search (real template step will be smaller or equal to template_step)
required_tau_rough = 0.1
radius_planet = 0.007

####star properties, change if you are working with another star####

#general properties
eight_planet_id = 11442793 #kepler id of Kepler90
u1_Kepler90 = 0.42 #quadratic limb darkening parameters
u2_Kepler90 = 0.278
GM_Kepler90 = 1.12 * 1.989e30 * 6.673e-11 #mass of the Kepler 90
Radius_star_Kepler90 = 1.29 * 6.957e8 #radius of Kepler 90 star in meters
relative_error_Radius_star_Kepler90 = 0.14791666666666667
q_Kepler90 = 0.04697692862748636 #Used for computing initial guess on time of transit given the period time_half_transit = q period^(1/3), everything in units of days. Assumes circular orbit with no inclination
errq_Kepler90 = 0.15016099154340684

#measurment properties
time0_short = 677.014598924 #for short cadence Kepler90 flux
lenTime_Kepler90 = 71427+300 #for long cadence Kepler90
#sigma_in_star_flux_units = 0.000934925153147 #numeric value of sigma in units of star flux (for Kepler 90)
# if long_cadence:
#     sigma_in_star_flux_units = 0.000198663809786
#

# planets parameters, this are only approximate values, initial guesses for optimization, they are listed as: [period, phase, half time of transit, amplitude]
names_planets = ['b', 'c', 'i', 'd', 'e', 'f', 'g', 'h']
                # 0    1    2    3    4    5    6    7
Kepler90_planets_parameters = \
          [7.0081588051562127, 6.1627398073846322, 0.084387927037585658, -0.65639318727415086,
           8.7197818679008954, 8.0050706748166007, 0.08034343724825907, -0.55537413584163142,
           14.448635844620199, 0.77034291242783226, 0.098731811805326794, -0.42267326324558169,
           59.736697969292628, 27.447580871203765, 0.16295542993502785, -2.6086103072160687,
           91.939814040219133, 2.7865268329682, 0.18509365269252892, -2.4269329710006096,
           124.90958636937347, 123.22434621200968, 0.21406437446323237, -2.8907384595715637,
 210.59820446235335, 15.5619169054129, 0.24051744936482589, -16.06488847504648]
# 331.59942659327555, 8.9842825444383987, 0.27662010037398799, -40.289663555792735]

# relative radius of planets (R_planet/R_star)
Kepler90_planets_radius = [0.006992828328479127, 0.0062988835325231834, 0.007046208697398815, 0.015373546248870142, 0.014252558501556694, 0.01542692661778983]  # , 0.04339823993170634, 0.06042657761708681]
# times_separate = [38.2526205124,128.732165713,220.884987619,311.99883848,407.957698392,498.682497994,677.02447534,775.353724395,869.71580825,1051.24544065,1142.64755134,1241.99553758,1427.73459222]


