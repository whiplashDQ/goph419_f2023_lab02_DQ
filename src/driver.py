import numpy as np
import matplotlib.pyplot as plt
from linalg_interp import spline_function

def main():
    # load water_temp vs density data
    water_temp = np.loadtxt('water_density_vs_temp_usgs.txt')
    #convert 2nd column value from g/m^3 to kg/m^3
    water_temp[:, 1] = water_temp[:, 1] / 1000
    water_t = water_temp[:, 0]
    water_d = water_temp[:, 1]
    # load air_temp vs density data    
    air_temp = np.loadtxt('air_density_vs_temp_eng_toolbox.txt')
    air_t = air_temp[:, 0]
    air_d = air_temp[:, 1]
    
    plt.figure(figsize = (10, 10))
    # set a dataset index and name
    plot_name = ['Water', 'Air']
    title_index = 0
    subplot_index = 1   # set a subplot index
    for data in [(water_t, water_d), (air_t, air_d)]:
        for order in range(1, 4):
            spline = spline_function(data[0], data[1],order)
            plt.subplot(2, 3, subplot_index)
            plt.scatter(data[0], data[1], label = 'Data')
            plt.plot(data[0], spline(data[0]), label= f' Order = {order}')
            plt.xlabel('Temperature (C)')
            plt.ylabel('Density (kg/m^3)')
            plt.title(f'{plot_name[title_index]} Temp vs Density')
            plt.legend()
            subplot_index += 1 # increase subplot index by 1
        title_index += 1
    plt.tight_layout()   
    plt.savefig('./figures/water_and_air_temp_vs_density.png')
    plt.show() 


if __name__ == "__main__":
    main()