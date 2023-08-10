from compas_fea.cad import rhino
from compas_fea.structure import ShellSection
from compas_fea.structure import ElasticIsotropic
from compas_fea.structure import ElementProperties as Properties
from compas_fea.structure import GeneralDisplacement
from compas_fea.structure import GeneralStep
from compas_fea.structure import PinnedDisplacement, FixedDisplacement 
from compas_fea.structure import PointLoad, GravityLoad
from compas_fea.structure import Structure

from platypus import NSGAII, Problem, Integer, Real 
from platypus import SBX, HUX, PM, BitFlip, CompoundOperator

import numpy as np
import sys 
from utilities import Parameters
from utilities import load_from_obj
from utilities import blockPrint, enablePrint, tic, toc 
from utilities import save_to_csv, load_from_cvs
from utilities import plot_fitdata, clean_fitness_data
from utilities import plot_ccdata
from utilities import generate_combinations
from design import Design 

# Author: James Whiteley (github.com/jamesalexwhiteley)

#=========================================
# mesh with python 
#=========================================

def mesh_python(parameters, vis=False):

    width = parameters.get_len_x()
    height = parameters.get_beam_depth()

    # Meshgrid
    mesh_div = width * 3
    width = width / 2 
    x = np.linspace(-width, width, mesh_div)
    y = np.linspace(-width, width, mesh_div)
    X, Y = np.meshgrid(x, y)

    # Sphere equation (cartesian)
    def sphere_eq(x, y, diameter):
        return np.sqrt(diameter**2 - x**2 - y**2)

    # Find diameter
    diameter = width * 2  # min val

    while True:
        Z = sphere_eq(X, Y, diameter)

        distance = np.max(Z) - np.min(Z)
        if np.abs(distance - height) < 0.001:
            break
        diameter += 0.01

    Z_min = np.min(Z)
    Z = Z - Z_min

    X = X + width
    Y = Y + width

    # surface_area
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    surface_area = sum(
    0.5 * np.sqrt((Z[j, i] + Z[j+1, i])**2 + (Z[j, i] + Z[j, i+1])**2 + (Z[j, i] - Z[j+1, i+1])**2) * dx * dy
    for i in range(len(x) - 1)
    for j in range(len(y) - 1)
    )
    
    parameters.set_surface_area(surface_area)

    faces = [
    [i*len(y) + j, i*len(y) + j + 1, i*len(y) + j + len(y) + 1, i*len(y) + j + len(y)]
    for i in range(len(x)-1)
    for j in range(len(y)-1)
    ]           

    return np.column_stack((X.ravel(), Y.ravel(), Z.ravel())), faces

#=========================================
# model with compas_fea  
#=========================================

def model_compas_fea(nodes, faces, parameters):

    if not output: blockPrint()

    shell_tck = parameters.get_tck_beam()
    width = parameters.get_len_x()
    X, Y, Z  = nodes[:,0], nodes[:,1], nodes[:,2]

    # Structure
    mdl = Structure(name='mesh_shell', path='C:/Temp/')

    # Nodes
    mdl.add_nodes(nodes=nodes)

    if output: 
        print('***** mesh: {0} nodes *****' .format(len(nodes)))
        print()

    mdl.add_elements(elements=faces, type='ShellElement')

    support_ind = [i for i, node in enumerate(nodes)
                if (node[2] == 0 and ((node[0] == 0 or node[0] == width) and (node[1] == 0 or node[1] == width)))]
    internal_ind = [i for i, node in enumerate(nodes) if node[2] == 0]
    int_half_ind = [i for i, node in enumerate(nodes) if 0 <= node[0] <= width/2]

    mdl.add_set(name='nset_corner', type='node', selection=support_ind)
    mdl.add_set(name='nset_internal', type='node', selection=internal_ind)
    mdl.add_set(name='elset_shell', type='element', selection=[f for f in range(len(faces))])
    mdl.add_set(name='nset_int_half', type='node', selection=int_half_ind)

    # Materials
    mdl.add(ElasticIsotropic(name='mat_elastic', E=30*10**9, v=0.2, p=4/3 * 2400))

    # Sections
    mdl.add(ShellSection(name='sec_shell', t=shell_tck))

    # Properties 
    mdl.add(Properties(name='ep_shell', material='mat_elastic', section='sec_shell', elset='elset_shell'))

    # Loads 
    live_load = (2.5 * (width * width) * 1000) / (mdl.node_count()-4) # 2.5 kPa 
    mdl.add(PointLoad(name='load_udl', nodes='nset_internal', z=-live_load))
    mdl.add(PointLoad(name='load_half', nodes='nset_int_half', z=-live_load)) 
    mdl.add(GravityLoad(name='load_gravity', elements='elset_shell'))

    # Supports 
    mdl.add(PinnedDisplacement(name='disp_pinned', nodes='nset_corner'))

    # Steps
    mdl.add([
        GeneralStep(name='step_bc', displacements='disp_pinned'),
        GeneralStep(name='step_udl', loads=['load_gravity', 'load_udl'], factor={'load_gravity': 1.35, 'load_udl': 1.50}),
        GeneralStep(name='step_half', loads=['load_gravity', 'load_half'], factor={'load_gravity': 1.35, 'load_half': 1.50}),
    ])
    mdl.set_steps_order(['step_bc', 'step_udl', 'step_half'])

    # Summary
    # mdl.summary()

    # Run
    if '-run' in sys.argv:
        mdl.analyse_and_extract(software='abaqus', fields=['u', 'sf', 'sm', 's', 'rf', 'cf'], cpus=1)
        mdl.save_to_obj() 

    mdl = load_from_obj('C:/Temp/' + name + '.obj')

    if not output: enablePrint() 

    return mdl, [f for f in range(len(faces))]


#=========================================
# design with python Design() object  
#=========================================

def design_python(mdl, select, parameters):

    """ 
    design to eurocode 2   
    
    Returns
    -------
    volume_reinf: float  
        volume reinf in slab (m3)
    volume_concrete: float  
        volume steel in slab (m3)
    penalty: float
        large number (penalty function for opt)

    """

    if not output: blockPrint()
    
    tck = parameters.get_tck_beam()
    fck = parameters.get_fck()
    surface_area = parameters.get_surface_area()

    volume_reinf = [0, 0]
    volume_concrete = surface_area * tck

    penalty = [0, 0]

    for i, step in enumerate(['step_udl', 'step_half']):

        design = Design(mdl, step) 

        # slab design, note abaqus sf, sm are per unit width 
        design.set_design_properties(b=1e3, d=tck*1e3, fck=fck, fyk=775, cover=5) 
        as0, p0 = design.slab_bending_and_shear('slab (sag)', 'local_1', select, 'min', 'abs', output, shell=True)
        as1, p1 = design.slab_bending_and_shear('slab (hog)', 'local_1', select, 'max', 'abs', output, shell=True)
        as2, p2 = design.slab_bending_and_shear('slab (sag)', 'local_2', select, 'min', 'abs', output, shell=True)
        as3, p3 = design.slab_bending_and_shear('slab (hog)', 'local_2', select, 'max', 'abs', output, shell=True)

        volume_reinf[i] = (np.sum((as0, as1, as2, as3)) * 1e-6 * np.sqrt(surface_area) * np.sqrt(surface_area)) 
        penalty[i] = np.sum((p0, p1, p2, p3))

        if output: print() 

    if not output: enablePrint() 
    
    return np.max(volume_reinf), volume_concrete, np.max(penalty)


#=========================================
# optimize with Platypus  
#=========================================

def objective_shell(parameters):

    """ 
    carries out the meshing, analysis and design and then computes the cost 

    """

    nodes, faces = mesh_python(parameters)

    # tic()
    mdl, select = model_compas_fea(nodes, faces, parameters)
    # toc()

    volume_steel, volume_concrete, p = design_python(mdl, select, parameters)
    
    density_reinf = 2700 # kg/m3 
    density_concrete = 4/3 * 2400 # kg/m3 

    mass_concrete = volume_concrete * density_concrete # kg 
    mass_reinf = volume_steel * density_reinf # kg 

    fck = parameters.get_fck()
    cement_type = parameters.get_cement_type()

    # A brief guide to calculating embodied carbon 2020 (IStructE)    
    carbon_f = 1.35 # kgCO2e/kg A1-A3 (material production) - ICE V3.0 Fibreglass (average)
    carbon_c = cement_type[str(fck)]

    # See Concept_v4_5 (TCC) 
    cost_f = 2.10 # pounds/kg 
    cost_c = 145 # pounds/m3 
    cost_form = 66 # pounds/m2
    
    cost = (cost_c * volume_concrete) + (cost_f * mass_reinf) + (cost_form * parameters.get_len_x() * parameters.get_len_y())
    carbon = (carbon_c * mass_concrete) + (carbon_f * mass_reinf)

    print([cost + p, carbon + p])

    return [cost + p, carbon + p]


class Optimize_shell(Problem):

    def __init__(self):
        super().__init__(3, 2) # 3 decision variables, 2 objectives, 0 constraints       
        self.types[:] = [var0, var1, var2]

    def evaluate(self, solution):
        beam_depth = solution.variables[0] * 1e-3
        tck_beam = solution.variables[1] * 1e-3
        fck = cement[solution.variables[2]]

        parameters = Parameters(beam_depth=beam_depth, tck_beam=tck_beam, tck_top=None, fck=fck, len_x=9, len_y=9)
        solution.objectives[:] = objective_shell(parameters)


if __name__ == "__main__":

    name = 'mesh_shell'
    myid = 'james'
    fitpath = 'C:/Users/'+myid+'/mres/compas_fea/data/'+name+'_fitdata.csv'
    ccpath = 'C:/Users/'+myid+'/mres/compas_fea/data/'+name+'_ccdata.csv'
    output = False    

    # test 
    parameters = Parameters(beam_depth=1.691, tck_beam=0.083, tck_top=None, fck=50, len_x=9, len_y=9)
    objective_shell(parameters)

    cement = [20, 25, 28, 30, 32, 35, 40, 45, 50] # cement types 
    var0 = Real(0, 1500) # -> x1: shell_depth  
    var1 = Real(80, 400)  # -> x2: thickness  
    var2 = Integer(0, 8)  # -> x3: fck 

    function_evals = 800
    population_size, crossover, mutation = 10, 0.6, 0.01

    # run optimization 
    if '-opt' in sys.argv: 

        tic()

        algorithm = NSGAII(Optimize_shell(), population_size=population_size, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip())) 
        algorithm.crossover_probability = crossover  # crossover rate
        algorithm.mutation_probability = mutation  # mutation rate  

        archives = np.zeros((int(function_evals/population_size), 2, 3))

        for iteration in range(int(function_evals/population_size)):
            print('Iteration: ' + str(iteration) + ' | ' + f'{(iteration+1)/int(function_evals/population_size) * 100:.0f}'  +' %')
            algorithm.step()     
            solutions = np.array([solution.objectives for solution in algorithm.result])

            archives[iteration, 0, 0] = np.min(solutions[:, 0])
            archives[iteration, 0, 1] = np.mean(solutions[:, 0])
            archives[iteration, 0, 2] = np.max(solutions[:, 0])

            archives[iteration, 1, 0] = np.min(solutions[:, 1])
            archives[iteration, 1, 1] = np.mean(solutions[:, 1])
            archives[iteration, 1, 2] = np.max(solutions[:, 1])

        runtime = toc()

        # cost/carbon data
        objectives = np.array([solution.objectives for solution in algorithm.result if solution.objectives[0] < 1e98])
        try: 
            min_objectives, max_objectives = np.min(objectives, axis=0), np.max(objectives, axis=0)
        except ValueError as ve:
            min_objectives, max_objectives = np.array([0, 0]), np.array([0, 0]) 

        variable0 = [solution.variables[0:2] for solution in algorithm.result if solution.objectives[0] < 1e98]
        variable2 = [var2.decode(solution.variables[2]) for solution in algorithm.result if solution.objectives[0] < 1e98]
        variables = [var0 + [var2] for var0, var2 in zip(variable0,  variable2)]
        ccdata = np.column_stack((objectives, variables, np.ones(len(variables)) * runtime))
        save_to_csv(ccpath, ccdata, name)

        # fitness_vs_iterations data
        fitdata = np.concatenate((archives[:,0,:], archives[:,1,:]), axis=1)
        save_to_csv(fitpath, fitdata, name)


    if '-plot' in sys.argv:

        # fitness_vs_iterations data 
        fitdata = load_from_cvs(fitpath)
        iters, fitdata = clean_fitness_data(fitdata)
        plot_fitdata(iters, fitdata)

        # cost/carbon data
        ccdata = load_from_cvs(ccpath)
        plot_ccdata(ccdata, cement, shell=True, best_fit_line=True) 