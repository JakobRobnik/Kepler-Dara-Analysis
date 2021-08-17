# Kepler-Data-Analysis
Unified analysis of outliers, stellar variability and planet transits using a Fouirer Gaussian process.

A collection of python scripts accompaining articles:
1. Kepler data analysis: non-Gaussian noise and Fourier Gaussian process analysis of stellar variability (https://iopscience.iop.org/article/10.3847/1538-3881/ab8460  or arxhiv version: https://arxiv.org/abs/1910.01167) 
2. Kepler-90: Giant transit-timing variations reveal a super-puff (https://iopscience.iop.org/article/10.3847/1538-3881/abe6a7 or arxhiv version: https://arxiv.org/abs/2011.08515)

Main file of this repository is fitting_planets_functions.py which is meant to be called from the file fitting_planet.py .

prepare_data.py is for importing Kepler light curves and prepare them for the analysis. 

noise.py containts non-Gaussian outlier distribution and determination of it's parameters.

If you have any issues please contact me at jakob_robnik@berkeley.edu .

A new repository containing Gaussianized matched filtering for planet detections, look elsewhere effect in the planet search and more will follow.
