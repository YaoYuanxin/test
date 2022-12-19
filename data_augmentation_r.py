"""
Data Augmentation in R using nlme package
"""
# Import rpy2 dependencies
import rpy2.robjects.packages as rpackages
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr

utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)


# Install packages
utils.install_packages('nlme')
utils.install_packages('caret')
utils.install_packages('readr')

#Load packages
nlme = importr('nlme')
nlme = importr('caret')
nlme = importr('readr')
