import pandas as pd
import os
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

main_dir = "C:/Users/angie/Git Root/StereoTraining/GameObservers/"
sub_dir = "/DartBoard/"
results_dir = "C:/Users/angie/Git Root/StereoTraining/figs/rawLogScale/"

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/data')
transferData = pd.read_csv('transferData.csv')

subjects = transferData.subject.unique()

