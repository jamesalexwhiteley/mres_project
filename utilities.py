# type: ignore
import matplotlib.pyplot as plt
import compas_fea.utilities
from sklearn.linear_model import LinearRegression
from scipy.spatial import Delaunay, Voronoi
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import numpy as np
import pickle 
import itertools
import sys, os
import csv 
import subprocess
from scipy.interpolate import griddata

# Author: James Whiteley (github.com/jamesalexwhiteley)

#=========================================
# Classes 
#=========================================

# for passing parameter info between functions in optimization  
class Parameters():

    def __init__(self, beam_depth, tck_beam, tck_top, fck, len_x, len_y, nrib=0):
        # free parameters 
        self.beam_depth = beam_depth
        self.tck_beam = tck_beam
        self.tck_top = tck_top
        self.fck = fck
        self.nrib = nrib
        # fixed parameters 
        self.len_x = len_x
        self.len_y = len_y
        # property 
        self.surface_area = None

        # carbon coefficients ICE v3.0
        self.cement =  {"20": 0.112, 
                        "25": 0.119, 
                        "28": 0.126,
                        "30": 0.132,
                        "32": 0.138,
                        "35": 0.149,
                        "40": 0.159,
                        "45": 0.169,
                        "50": 0.179}
        
    def __str__(self):
        attributes = vars(self)
        attributes_string = ", ".join([f"{key}={value}" for key, value in attributes.items()])
        return f"Parameters({attributes_string})"
    
    def get_beam_depth(self):
        return self.beam_depth
    
    def get_tck_beam(self):
        return self.tck_beam
    
    def get_tck_top(self):
        return self.tck_top
    
    def get_fck(self):
        return self.fck
    
    def get_len_x(self):
        return self.len_x
    
    def get_len_y(self):
        return self.len_y
    
    def get_cement_type(self):
        return self.cement
    
    def get_nrib(self):
        return self.nrib
    
    def get_surface_area(self):
        return self.surface_area
    
    def set_surface_area(self, surface_area):
        self.surface_area = surface_area
  

#=========================================
# Utility funcitons 
#=========================================

# copy from compas_fea
def load_from_obj(filename, output=True):
    
        with open(filename, 'rb') as f:
            structure = pickle.load(f)

        if output:
            print('***** Structure loaded from: {0} *****'.format(filename))

        return structure


# def get_elm_field(mdl, field, minmax, all_elms): 
#     """
#         Parameters
#         ----------
#         field : str
#             e.g. 'sf4'
#         minmax : str 
#             'min' or 'max' of elm ip values 
#         all_elms : list
#             element index list
#     """

#     elm_dict = mdl.get_element_results('step_load', field, elements='all')
#     _, elm_data = process_data(elm_dict, 'element', minmax, None, all_elms, mdl.node_count())
#     return elm_data 


# generally for surface data
def plot_fig(X, Y, Z, X1=np.zeros((2,2)), Y1=np.zeros((2,2)), Z1=np.zeros((2,2))):
     
    if X1.all() == 0 and Y1.all() == 0 and Z1.all() == 0: 
        nodes = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    else: 
        nodes0 = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        nodes1 = np.column_stack((X1.ravel(), Y1.ravel(), Z1.ravel()))
        nodes = np.vstack((nodes0, nodes1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.scatter(nodes[:,0], nodes[:,1], nodes[:,2], c='b')

    ax.set_box_aspect([5, 5, 0.2])
    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for i, node in enumerate(nodes):
        ax.text(node[0], node[1], node[2], f'({i})', color='red', fontsize=7.5)

    plt.show()  


# disable / enable calls to print 
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__


# matlab tic toc functions
def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + "{:.2f}".format(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")
    return float(time.time() - startTime_for_tictoc)


# generate all combinations of items from three sets
def generate_combinations(set1, set2, set3):
    
    combinations = itertools.product(set1, set2, set3)

    found_values = []
    for combination in combinations:

        if len(set(combination)) == 3:
            found_values.append(list(combination))

    return found_values


# remove penalty function related entires from data 
def clean_fitness_data(data):
    iterations = np.linspace(0, len(data)-1, len(data)) 
    mask = [np.all(data[i,:] < 1e90) for i in range(len(iterations))]
    return iterations[mask], data[mask]


# plotting fitness vs iteration 
def plot_fitdata(iterations, fitdata):
    archive_minfit1 = fitdata[:,0]
    archive_avgfit1 = fitdata[:,1]
    archive_maxfit1 = fitdata[:,2]
    archive_minfit2 = fitdata[:,3]
    archive_avgfit2 = fitdata[:,4]
    archive_maxfit2 = fitdata[:,5]
    
    RoyalBlue = (65/255, 105/255, 225/255)
    Crimson = (220/255, 20/255, 60/255)

    DarkCyan = (0/255, 139/255, 139/255)
    Coral = (255/255, 127/255, 80/255)

    MediumPurple = (147/255, 112/255, 219/255)
    DarkOliveGreen = (85/255, 107/255, 47/255)

    DarkSlateGray = (47/255, 79/255, 79/255)
    Tomato = (255/255, 99/255, 71/255)

    Orange = (255/255, 165/255, 0/255)
    DeepSkyBlue = (0/255, 191/255, 255/255)

    DarkGrey = (169/255, 169/255, 169/255)
    Teal = (0/255, 128/255, 128/255)

    LimeGreen = (50/255, 205/255, 50/255)
    Green = (0/255, 128/255, 0/255)

    max_iter = len(archive_minfit1)

    plt.plot(iterations[0:max_iter], archive_minfit1[0:max_iter], color='Red', linestyle='--', linewidth=0.2, label='Min/max f1: cost')
    plt.plot(iterations[0:max_iter], archive_avgfit1[0:max_iter], color='Red', linestyle='-', linewidth=1, label='Average f1: cost')
    plt.plot(iterations[0:max_iter], archive_maxfit1[0:max_iter], color='Red', linestyle='--', linewidth=0.2)
    plt.plot(iterations[0:max_iter], archive_minfit2[0:max_iter], color=Green, linestyle='--', linewidth=0.2, label='Min/max f2: carbon')
    plt.plot(iterations[0:max_iter], archive_avgfit2[0:max_iter], color=Green, linestyle='-', linewidth=1, label='Average f2: carbon')
    plt.plot(iterations[0:max_iter], archive_maxfit2[0:max_iter], color=Green, linestyle='--', linewidth=0.2)

    plt.fill_between(iterations[0:max_iter], archive_minfit1[0:max_iter], archive_maxfit1[0:max_iter], color='Red', alpha=0.05)
    plt.fill_between(iterations[0:max_iter], archive_minfit2[0:max_iter], archive_maxfit2[0:max_iter], color=Green, alpha=0.05)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (£ or kgCO2e)')
    plt.legend()
    plt.show()


# save to csv format 
def save_to_csv(path, data, name):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    # print(''f'***** data saved to {name} successfully *****')


# load from csv format 
def load_from_cvs(path):
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = np.array([[float(item) for item in row] for row in reader])
        return data 


def plot_ccdata(data, cement, one=False, ribbed=False, shell=False, best_fit_line=False):

    fig, ax = plt.subplots()

    x1 = np.array([row[0] for row in data]).reshape(-1, 1) # cost
    x2 = np.array([row[1] for row in data]).reshape(-1, 1) # carbon 

    # print('\nBest cost solution: ' + str(data[np.where(x1.ravel() == np.min(x1))[0][0]]))
    # print('\nBest carbon solution: ' + str(data[np.where(x2.ravel() == np.min(x2))[0][0]]))

    correlation_coefficient, p_value = pearsonr(x1.ravel(), x2.ravel())
    if p_value < 0.01: p_value = 0.01
    # print("Pearson's correlation coefficient, r:", correlation_coefficient) # r2 = R2 since single variable model 
    # print("p-value:", p_value) 

    # linear regression model 
    model = LinearRegression()
    model.fit(x1, x2)
    y_pred = model.predict(x1)   
    r2 = r2_score(x2, y_pred) 
    x1, y_pred = x1.ravel(), y_pred.ravel()
    # x1, y_pred = np.sort(x1.ravel()), np.sort(y_pred.ravel())
    dx, dy = x1[1] - x1[0], y_pred[1] - y_pred[0]
    
    # plot figure 
    ax.scatter(x1, x2, marker="d", color='k', s=11.5)
    # for i, pt in enumerate(data): plt.text(pt[0], pt[1], f'({i})', color='red', fontsize=7.5)

    if best_fit_line: 
        angle = np.rad2deg(np.arctan2(dy, dx))
        # plt.text(x1[-1], y_pred[-1], f'R$^2$= {r2:.4f}      p-value: {p_value:.2f}', ha='right', va='bottom',
        #             transform_rotates_text=True, rotation=angle, rotation_mode='anchor')    
        line = plt.plot(x1, y_pred, 'k-', linewidth=1) 

    # plt.rcParams["figure.dpi"] = 300
    plt.xlabel("$f_1:$ $cost$ $(£)$")
    plt.ylabel("$f_2:$ $carbon$ $(kgCO2e)$")
    plt.title("NSGA-II results")   

    def add_pt_box(pt_ind, ha, va):

        if one: 
            fck = float(cement[int(data[pt_ind][5])])
            pt_text = f'x1 : {data[pt_ind][2]:.0f} \n x2 : {data[pt_ind][3]:.0f} \n x3 : {data[pt_ind][4]:.0f} \n x4 : {fck:.0f}'

        if ribbed:
            fck = float(cement[int(data[pt_ind][5])])
            pt_text = f'x1 : {data[pt_ind][2]:.0f} \n x2 : {data[pt_ind][3]:.0f} \n x3 : {data[pt_ind][4]:.0f} \n x4 : {fck:.0f} \n x5 : {data[pt_ind][6]:.0f} \n x6 : {data[pt_ind][7]:.0f} \n x7 : {data[pt_ind][8]:.0f} \n x8 : {data[pt_ind][9]:.0f}'

        if shell: 
            fck = float(cement[int(data[pt_ind][4])])
            pt_text = f'x1 : {data[pt_ind][2]:.0f} \n x2 : {data[pt_ind][3]:.0f} \n x4 : {fck:.0f}'

        ax.annotate(pt_text, xy=(data[pt_ind][0], data[pt_ind][1]), xytext=(-20,20), 
            textcoords='offset points', ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
                            color='red'))
        
    # mesh_one 
    # add_pt_box(pt_ind=8, ha='right', va='top')
    # add_pt_box(pt_ind=3, ha='right', va='top')
    # add_pt_box(pt_ind=2, ha='left', va='bottom')

    # mesh_shell
    # add_pt_box(pt_ind=8, ha='right', va='top')
    # add_pt_box(pt_ind=3, ha='left', va='bottom')
    # add_pt_box(pt_ind=2, ha='left', va='bottom')
    # add_pt_box(pt_ind=1, ha='left', va='bottom')

    # mesh_ribbed 
    add_pt_box(pt_ind=4, ha='right', va='bottom')
    add_pt_box(pt_ind=2, ha='right', va='top')
    add_pt_box(pt_ind=5, ha='right', va='bottom')

    highlight_ind = 8
    circle = plt.Circle((data[highlight_ind][0], data[highlight_ind][1]), radius=0.1, color='red', fill=False)
    ax.add_artist(circle)   
    highlight_ind = 7
    circle = plt.Circle((data[highlight_ind][0], data[highlight_ind][1]), radius=0.1, color='red', fill=False)
    ax.add_artist(circle) 

    if best_fit_line:
        plt.legend(line, [f'R$^2$= {r2:.4f} | p-value: {p_value:.2f}'])
    
    plt.show()


def plot_landata(points, zlabel, name):

    myid = 'james'
    path = 'C:/Users/'+myid+'/mres/compas_fea/figs/'

    # filter infeasible points
    fontsize = 13
    filtered_points = points[points[:, 2] <= 1e10]
    x = filtered_points[:, 0]
    y = filtered_points[:, 1]
    z = filtered_points[:, 2]

    # print(np.min(x))
    # print(np.min(y))
    # print(np.min(z))

    num = 100
    xi = np.linspace(min(x), max(x), num=num)
    yi = np.linspace(min(y), max(y), num=num)
    X, Y = np.meshgrid(xi, yi)

    # interpolate z values
    Z = griddata((x, y), z, (X, Y), method='cubic')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))
    ax.grid(False)

    ax.set_xlabel('x1: Beam depth (m)', fontsize=fontsize)
    ax.set_ylabel('x3: Slab thickenss (m)', fontsize=fontsize)
    ax.set_zticks([])
    ax.set_zlabel('')
    ax.w_zaxis.line.set_lw(0)

    ax.view_init(elev=30, azim=230) 

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=True)
    # surf = ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='k', linewidth=2)

    colorbar = fig.colorbar(surf, ax=ax, shrink=0.4, aspect=10)
    colorbar.set_label(zlabel, fontsize=fontsize)

    ax.scatter(0.644028041, 0.1686423901, 6315.860513, color='red', s=10, label=f'Pareto front solution: [{644.028041:.0f}, {168.6423901:.0f}, {6315.860513:.0f}]')
    ax.legend()

    plt.savefig(path + name + '_landscape_' + zlabel[0:2] + '.png', dpi=300, bbox_inches='tight')

    plt.show()


def main():
    # Run other scripts using subprocess 
    # subprocess.run(["python", "mesh_one.py", "-run"])
    subprocess.run(["python", "mesh_ribbed.py", "-run"])
    # subprocess.run(["python", "mesh_shell.py", "-run"])

if __name__ == "__main__":
    # main()
    plot_landata()