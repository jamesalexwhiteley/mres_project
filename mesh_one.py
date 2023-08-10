# type: ignore
from math import inf
from re import L
from tkinter import NSEW
from turtle import pen
from compas_fea.structure import ShellSection 
from compas_fea.structure import ElasticIsotropic, Concrete 
from compas_fea.structure import ElementProperties as Properties
from compas_fea.structure import GeneralDisplacement, RollerDisplacementXY
from compas_fea.structure import GeneralStep
from compas_fea.structure import PinnedDisplacement, FixedDisplacement
from compas_fea.structure import PointLoad, GravityLoad
from compas_fea.structure import Structure 
from compas_fea.utilities import process_data 

from platypus import NSGAII, Problem, Integer, Real 
from platypus import SBX, HUX, PM, BitFlip, CompoundOperator
from sklearn.linear_model import LinearRegression
from scipy.spatial import Delaunay, Voronoi
from scipy.stats import pearsonr

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pyvista as pv 
import sys, os 
import numpy as np 
import meshio 
import csv 
import gmsh 

from utilities import load_from_obj, blockPrint, enablePrint, tic, toc 
from utilities import save_to_csv, load_from_cvs
from utilities import plot_fitdata, clean_fitness_data
from utilities import plot_ccdata, plot_landata
from utilities import generate_combinations
from utilities import Parameters
from design import Design 

font = {'family': 'Times New Roman'}
plt.rcParams['font.family'] = font['family']

# Author: James Whiteley (github.com/jamesalexwhiteley)

#=========================================
# mesh with gmsh  
#=========================================

def mesh_gmsh(parameters):

    b, l = parameters.get_len_x(), parameters.get_len_x()
    beam_depth = parameters.get_beam_depth()
      
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0) 
    gmsh.model.add(name)

    lc = 0.5 # min ~0.35-0.4 
    d = beam_depth
    
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(b, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(b, l, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, l, 0, lc)
    p5 = gmsh.model.geo.addPoint(0, 0, -d, lc)
    p6 = gmsh.model.geo.addPoint(0, l, -d, lc)
    p7 = gmsh.model.geo.addPoint(b, 0, -d, lc)
    p8 = gmsh.model.geo.addPoint(b, l, -d, lc)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    l5 = gmsh.model.geo.addLine(p4, p6)
    l6 = gmsh.model.geo.addLine(p6, p5)
    l7 = gmsh.model.geo.addLine(p5, p1)
    l8 = gmsh.model.geo.addLine(p3, p8)
    l9 = gmsh.model.geo.addLine(p8, p7)
    l10 = gmsh.model.geo.addLine(p7, p2)

    c1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    c2 = gmsh.model.geo.addCurveLoop([-l4, l5, l6, l7])
    c3 = gmsh.model.geo.addCurveLoop([l2, l8, l9, l10])

    s1 = gmsh.model.geo.addPlaneSurface([c1])
    s2 = gmsh.model.geo.addPlaneSurface([c2])
    s3 = gmsh.model.geo.addPlaneSurface([c3])

    # p9 = gmsh.model.geo.addPoint(0, l/2, -beam_depth/2, 0.1*lc)
    gmsh.model.geo.synchronize()
    # gmsh.model.mesh.embed(0, [p9], 2, s2)
    gmsh.model.mesh.generate(2)

    if '-vis' in sys.argv:
        gmsh.fltk.run() 

    _, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elementType = gmsh.model.mesh.getElementType("triangle", 1)
    faceNodes = gmsh.model.mesh.getElementFaceNodes(elementType, 3)
    edgeNodes = gmsh.model.mesh.getElementEdgeNodes(elementType)

    nodes = np.reshape(nodeCoords, (int(len(nodeCoords)/3), 3))
    faces = np.reshape(faceNodes, (int(len(faceNodes)/3), 3))
    edges = np.reshape(edgeNodes, (int(len(edgeNodes)/2), 2))

    # print("Nodes:")
    # print(nodes)
    # print("Faces:")
    # print(faces)

    # get top faces
    elementType, faceType = gmsh.model.mesh.getElementType("triangle", 1), 3
    faceNodes = gmsh.model.mesh.getElementFaceNodes(elementType, faceType, s1)
    top_faces = np.reshape(faceNodes, (int(len(faceNodes)/3), 3))

    if '-nodpo' in sys.argv:
        if len(nodes) > 1000: 
            raise ValueError('***** >1000 ({0}) nodes in gmsh mesh*****' .format(len(nodes)))
    
    if output: 
        print('***** Gmsh mesh: {0} nodes *****' .format(len(nodes)))
        print()

    gmsh.finalize()

    return nodes, faces, top_faces 


#=========================================
# model with compas_fea 
#=========================================

def model_compas_fea(nodes, faces, top_faces, parameters):

    # if not output: blockPrint()
    blockPrint()

    tck_beam = parameters.get_tck_beam()
    tck_top = parameters.get_tck_top()
    beam_depth = parameters.get_beam_depth()
    b, l = parameters.get_len_x(), parameters.get_len_x()

    # Structure
    mdl = Structure(name=name, path='C:/Temp/')
    d = beam_depth

    mdl.add_nodes(nodes=nodes)
    faces = (faces-1).tolist() # gmsh indexing starts from 1
    top_faces = (top_faces-1).tolist()

    mdl.add_elements(elements=top_faces, type='ShellElement') 
    num_tf = mdl.element_count() 
    select_tf = [i for i in range(0, num_tf)] 

    mdl.add_elements(elements=faces, type='ShellElement') 
    num_bf = mdl.element_count() 
    select_bf = [i for i in range(num_tf, num_bf)] 

    # Sets 
    support_ind = [i for i, node in enumerate(nodes)
                if (node[2] == -d and ((node[0] == 0 or node[0] == l) and (node[1] == 0 or node[1] == l)))]
    internal_ind = [i for i, node in enumerate(nodes) if node[2] == 0]
    int_half_ind = [i for i, node in enumerate(nodes) if node[2] == 0 and 0 <= node[0] <= b/2]

    # get beam faces excluding ends 
    ign = min(b, l) # fraction ignored 
    select_bf2 = list(set([face_ind for face_ind in select_bf
                for node_ind in mdl.elements[face_ind].nodes
                if (1/ign)*l < mdl.nodes[node_ind].y < (1-1/ign)*l]))

    # get beam middle  
    ign = 2.5 # fraction ignored 
    select_bf3 = list(set([face_ind for face_ind in select_bf
                for node_ind in mdl.elements[face_ind].nodes
                if (1/ign)*l < mdl.nodes[node_ind].y < (1-1/ign)*l]))

    # get top faces excluding edge 
    ign = min(b, l) # fraction ignored 
    select_tf2 = list(set([face_ind for face_ind in select_tf
                for node_ind in mdl.elements[face_ind].nodes
                if (1/ign)*l < mdl.nodes[node_ind].y < (1-1/ign)*l and
                    (1/ign)*b < mdl.nodes[node_ind].x < (1-1/ign)*b]))

    mdl.add_set(name='nset_pinned', type='node', selection=support_ind[0])
    mdl.add_set(name='nset_roller', type='node', selection=support_ind[1:])
    mdl.add_set(name='nset_internal', type='node', selection=internal_ind)
    mdl.add_set(name='nset_int_half', type='node', selection=int_half_ind)
    mdl.add_set(name='elset_top', type='element', selection=select_tf)
    mdl.add_set(name='elset_beam', type='element', selection=select_bf)

    # Materials
    mdl.add(ElasticIsotropic(name='mat_elastic', E=30*10**9, v=0.2, p=2400))

    # Sections
    mdl.add(ShellSection(name='sec_top', t=tck_top))
    mdl.add(ShellSection(name='sec_beam', t=tck_beam))

    # Properties
    mdl.add([
        Properties(name='ep_top', material='mat_elastic', section='sec_top', elset='elset_top'),
        Properties(name='ep_beam', material='mat_elastic', section='sec_beam', elset='elset_beam'),
    ])

    # Loads 
    live_load = 2.5 * l * b * 1000 / len(internal_ind) 
    mdl.add(PointLoad(name='load_udl', nodes='nset_internal', z=-live_load)) 
    mdl.add(PointLoad(name='load_half', nodes='nset_int_half', z=-live_load)) 
    mdl.add(GravityLoad(name='load_gravity', elements=['elset_top', 'elset_beam']))

    # Supports 
    mdl.add([
        PinnedDisplacement(name='disp_pinned', nodes='nset_pinned'),
        RollerDisplacementXY(name='disp_roller', nodes='nset_roller'),
    ])

    # Steps
    mdl.add([
        GeneralStep(name='step_bc', displacements=['disp_pinned', 'disp_roller']),
        GeneralStep(name='step_udl', loads=['load_gravity', 'load_udl'], factor={'load_gravity': 1.35, 'load_udl': 1.50}),
        # GeneralStep(name='step_half', loads=['load_gravity', 'load_half'], factor={'load_gravity': 1.35, 'load_half': 1.50}),
    ])
    mdl.set_steps_order(['step_bc', 'step_udl'])

    # Summary
    # mdl.summary()
    # res = mdl.results
    # print(res['step_udl']['element'].keys())

    # Run
    if '-run' in sys.argv:
        mdl.analyse_and_extract(software='abaqus', fields=['u', 'sf', 'sm', 's', 'rf', 'cf'], cpus=1)
        mdl.save_to_obj() 

    mdl = load_from_obj('C:/Temp/' + name + '.obj')

    # if not output: enablePrint()
    enablePrint()

    return mdl, select_tf2, select_bf2, select_bf3


#=========================================
# design with python Design() object  
#=========================================

def design_python(mdl, select_tf2, select_bf2, select_bf3, parameters):

    """
    design to eurocode 2   
    
    Returns
    -------
    volume_steel: float  
        volume steel in slab (m3)
    volume_concrete: float  
        volume steel in slab (m3)
    penalty: float
        large number (penalty function for opt)

    """

    if not output: blockPrint()
    
    len_x, len_y = parameters.get_len_x(), parameters.get_len_x()
    beam_depth = parameters.get_beam_depth()
    tck_beam = parameters.get_tck_beam()
    tck_top = parameters.get_tck_top()
    fck = parameters.get_fck()

    volume_steel = [0, 0]
    volume_concrete = len_y * len_x * tck_top + 2 * (len_y * np.max((beam_depth - tck_top) * tck_beam, 0))
    lv = 2 * ((beam_depth - 2*0.03) + (tck_beam - 2*0.03)) # length shear link mm 

    penalty = [0, 0]

    for i, step in enumerate(['step_udl']):

        design = Design(mdl, step) 

        # slab design, note abaqus sf, sm are per unit width 
        design.set_design_properties(b=1e3, d=tck_top*1e3, fck=fck) 
        as0, p0 = design.slab_bending_and_shear('slab (sag)', 'local_1', select_tf2, 'min', 'abs', output)
        as1, p1 = design.slab_bending_and_shear('slab (hog)', 'local_1', select_tf2, 'max', 'abs', output)
        as2, p2 = design.slab_bending_and_shear('slab (sag)', 'local_2', select_tf2, 'min', 'abs', output)
        as3, p3 = design.slab_bending_and_shear('slab (hog)', 'local_2', select_tf2, 'max', 'abs', output)

        # beam design, local 2 = global y, hence sm1 is torsion, s22 is bending stress. sf3 is shear 
        design.set_design_properties(b=tck_beam*1e3, d=beam_depth*1e3, hf=tck_top*1e3 ,fck=fck) 
        ab, av, p4 = design.beam_bending_and_shear('beam (sag)', 'local_2', select_bf3, select_bf2, 'abs', output) 

        volume_steel[i] = (np.sum((as0, as1, as2, as3)) * 1e-6 * len_x * len_y) + 2 * ((ab * 1e-6 * len_y) + (av * 1e-6 * lv * len_y/0.2)) # 200mm link spacing assumed 
        penalty[i] = np.sum((p0, p1, p2, p3, p4))

        if output: print() 

    if not output: enablePrint() 
    
    return np.max(volume_steel), volume_concrete, np.max(penalty)


#=========================================
# optimize with Platypus  
#=========================================

def objective_one(parameters):
    
    """ 
    carries out the meshing, analysis and design and then computes the cost 

    """

    nodes, faces, top_faces = mesh_gmsh(parameters)

    # tic()
    mdl, select_tf2, select_bf2, select_bf3 = model_compas_fea(nodes, faces, top_faces, parameters)
    # toc()

    volume_steel, volume_concrete, p = design_python(mdl, select_tf2, select_bf2, select_bf3, parameters)

    density_steel = 7850 # kg/m3 
    density_concrete = 2400 # kg/m3 

    mass_concrete = volume_concrete * density_concrete # kg 
    mass_steel = volume_steel * density_steel # kg 

    fck = parameters.get_fck()
    cement_type = parameters.get_cement_type()

    # A brief guide to calculating embodied carbon 2020 (IStructE)    
    carbon_s = 0.684 # kgCO2e/kg A1-A3 (material production) - UK: BRC EPD
    # carbon_c = 0.103 # kgCO2e/kg A1-A3 (material production) - UK: In situ C30/37 (35% cement replacement)
    carbon_c = cement_type[str(fck)]

    # See Concept_v4_5 (TCC) 
    cost_c = 145 # pounds/m3 
    cost_s = 0.98 # pounds/kg 
    cost_form = 36 # pounds/m2
    
    cost = (cost_c * volume_concrete) + (cost_s * mass_steel) + (cost_form * parameters.get_len_x() * parameters.get_len_y())
    carbon = (carbon_c * mass_concrete) + (carbon_s * mass_steel)

    return [cost + p, carbon + p]


class Optimize_one(Problem):

    def __init__(self):
        super().__init__(4, 2) # 3 decision variables, 2 objectives, 0 constraints       
        self.types[:] = [var0, var1, var2, var3]

    def evaluate(self, solution):
        beam_depth = solution.variables[0] * 1e-3
        tck_beam = solution.variables[1] * 1e-3
        tck_top = solution.variables[2] * 1e-3
        fck = cement[solution.variables[3]]

        parameters = Parameters(beam_depth=beam_depth, tck_beam=tck_beam, tck_top=tck_top, fck=fck, len_x=9, len_y=9)
        solution.objectives[:] = objective_one(parameters)


if __name__ == "__main__":

    name = 'mesh_one'
    myid = 'james'
    fitpath = 'C:/Users/'+myid+'/mres/compas_fea/data/'+name+'_fitdata.csv'
    ccpath = 'C:/Users/'+myid+'/mres/compas_fea/data/'+name+'_ccdata.csv'
    lanpath = 'C:/Users/'+myid+'/mres/compas_fea/data/'+name+'_landata.csv'
    output = False # algorithm output    
    
    function_evals = 800

    cement = [20, 25, 28, 30, 32, 35, 40, 45, 50] # cement types 
    var0 = Real(80, 900) # -> x1: beam depth 
    var1 = Real(80, 600) # -> x2: beam tck 
    var2 = Real(80, 600) # -> x3: slab tck   
    var3 = Integer(0, 8)  # -> x4: fck  

    # mesh sensitivity study 
    # parameters = Parameters(beam_depth=0.6, tck_beam=0.3, tck_top=0.2, fck=30, len_x=9, len_y=9)

    # test 
    parameters = Parameters(beam_depth=0.632, tck_beam=0.105, tck_top=0.148, fck=40, len_x=9, len_y=9)
    objective_one(parameters)

    # hyperparameter tuning 
    # np.random.seed(0) 
    # hypdata = np.zeros((8, 5)) 
    # combinations = generate_combinations([10, 20], [0.6, 0.9], [0.01, 0.1]) 
    # for i, combination in enumerate(combinations):
    #     population_size, crossover, mutation = combination[0], combination[1] , combination[2] 

    population_size, crossover, mutation = 10, 0.6, 0.01

    # run optimization 
    if '-opt' in sys.argv: 

        tic()

        algorithm = NSGAII(Optimize_one(), population_size=population_size, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip())) 
        algorithm.crossover_probability = crossover  # crossover rate
        algorithm.mutation_probability = mutation  # mutation rate  
        # algorithm.run(function_evals) # there are actually ceil(evals/popsize) iterations 

        # hyperparam study: fitness stored in 2D array 
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

        # print results 
        # for solution in algorithm.result:
        #     solution_fck = var3.decode(solution.variables[3])
            # print('\n'f"Decision Variables: {solution.variables[0]:.0f}, {solution.variables[1]:.0f}, {solution.variables[2]:.0f}, {solution_fck:.0f}")
            # print(f"Objectives: {solution.objectives}")
            
        # cost/carbon data
        objectives = np.array([solution.objectives for solution in algorithm.result if solution.objectives[0] < 1e98])
        try: 
            min_objectives, max_objectives = np.min(objectives, axis=0), np.max(objectives, axis=0)
        except ValueError as ve:
            min_objectives, max_objectives = np.array([0, 0]), np.array([0, 0]) 

        variable0 = [solution.variables[0:3] for solution in algorithm.result if solution.objectives[0] < 1e98]
        variable3 = [var3.decode(solution.variables[3]) for solution in algorithm.result if solution.objectives[0] < 1e98]
        variables = [var0 + [var3] for var0, var3 in zip(variable0,  variable3)]
        ccdata = np.column_stack((objectives, variables, np.ones(len(variables)) * runtime))
        save_to_csv(ccpath, ccdata, name)

        # hyperparameter tuning data 
        # hypdata[i,0:3] = [population_size, crossover, mutation]
        # hypdata[i,3:5] = [min_objectives[0], min_objectives[1]]

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
        plot_ccdata(ccdata, cement, one=True, best_fit_line=True)


    if '-lan' in sys.argv:

        # fitness landscape 
        num = 50
        landata = np.zeros((num**2, 4))

        for i, depth in enumerate(np.linspace(0.5, 1.5, num=num)):
            for j, tck in enumerate(np.linspace(0.15, 0.25, num=num)):

                print('Iteration: ' + f'{(i * num + j):.0f}' + ' | ' + f'{(i * num + j + 1)/(num**2) * 100:.0f}' + ' %')

                parameters = Parameters(beam_depth=depth, tck_beam=0.1323, tck_top=tck, fck=30, len_x=9, len_y=9)
                cc_pts = objective_one(parameters) # cost, carbon 
                landata[i * num + j] = [depth, tck, cc_pts[0], cc_pts[1]]

        save_to_csv(lanpath, landata, name)

    if '-lanplot' in sys.argv:
        landata = load_from_cvs(lanpath)
        plot_landata(landata[:, :3], 'f1: cost (£)', name)
        plot_landata(landata[:, [0, 1, 3]], 'f2: carbon (kgCO2e)', name)
    


