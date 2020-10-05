# Kepler-Dara-Analysis
Unified analysis of outliers, stellar variability and planet transits using Fouirer Gaussian process

A collection of python scripts accompaining an article on Fourier GP https://iopscience.iop.org/article/10.3847/1538-3881/ab8460 (https://arxiv.org/abs/1910.01167) is presented.

Main file of this repository is fitting_planets_functions.py which is meant to be called from the fitting_planet.py .

prepare_data.py is for importing Kepler light curves and prepare them for the analysis. 

noise.py containts non-Gaussian outlier distribution and determination of it's parameters.

Then there is a collection of files that can be used for downloading Kepler light curves. As an example I uploaded light curve for Kepler 90 (zeros_....txt), extracted planet parameters, power spectrum and stellar parameters (...semi_kill.npy) and a residual flux search_flux.npy .

If you have troubles with dependancies or functions not working properly please contact me at jrobnik@student.ethz.ch .

A new repository containing Gaussianized matched filtering for planet detections, look elsewhere effect in the planet search and more will follow.
