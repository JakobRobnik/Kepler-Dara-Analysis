# Kepler-Dara-Analysis
Unified analysis of outliers, stellar variability and planet transits using Fouirer Gaussian process

A collection of python scripts accompaining an article on Fourier GP https://iopscience.iop.org/article/10.3847/1538-3881/ab8460 (https://arxiv.org/abs/1910.01167) is presented.
Main file of this repository is fitting_planets_functions.py which is meant to be called from the fitting_planet.py 
prepare_data.py is for importing Kepler light curves and prepare them for the analysis. 
noise_analysis.py containts non-Gaussian outlier distribution and determination of it's parameters.

A new repository containing Gaussianized matched filtering for planet detections, look elsewhere effect in the planet search and more will follow.
