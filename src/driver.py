import numpy as np
from linalg_interp import gauss_iter_solve

def main():
    water_temp = np.loadtxt('water_density_vs_temp_usgs.txt')
    #convert 2nd column value from g/m^3 to kg/m^3
    water_temp[:, 1] = water_temp[:, 1] / 1000
    #replace 1st column with kg/m^3
    water_temp_kg = water_temp
    



if __name__ == "__main__":
    main()