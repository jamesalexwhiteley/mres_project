# type: ignore
from compas_fea.cad import rhino
from compas_fea.structure import ShellSection
from compas_fea.structure import ElasticIsotropic
from compas_fea.structure import ElementProperties as Properties
from compas_fea.structure import GeneralDisplacement, RollerDisplacementXY
from compas_fea.structure import GeneralStep
from compas_fea.structure import PinnedDisplacement, FixedDisplacement 
from compas_fea.structure import PointLoad, GravityLoad 
from compas_fea.structure import Structure

from platypus import NSGAII, Problem, Integer, Real 
from platypus import SBX, HUX, PM, BitFlip, CompoundOperator

from utilities import Parameters
from utilities import load_from_obj
from utilities import blockPrint, enablePrint, tic, toc 
from utilities import save_to_csv, load_from_cvs
from utilities import plot_fitdata, clean_fitness_data
from utilities import plot_ccdata
from utilities import generate_combinations

from itertools import accumulate
from design import Design 
import matplotlib.pyplot as plt
import numpy as np
import gmsh 
import sys 
import os 

# Author: James Whiteley (github.com/jamesalexwhiteley)

#=========================================
# mesh with gmsh  
#=========================================

def mesh_gmsh(parameters, vis=False):

    x, y = parameters.get_len_x(), parameters.get_len_x()
    beam_depth = parameters.get_beam_depth()
    nrib = parameters.get_nrib()
      
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0) 
    gmsh.model.add(name)

    lc = 0.5 # ~0.5 
    d = beam_depth

    # rib spacing 
    yvals = np.linspace(0, y, nrib) 
    x0i, xli = [], []

    # points
    for yval in yvals:
        pi = gmsh.model.geo.addPoint(0, yval, 0, lc)
        x0i.append(pi)
    for yval in yvals:
        pi = gmsh.model.geo.addPoint(0, yval, -d, lc)
        x0i.append(pi)

    for yval in yvals:
        pi = gmsh.model.geo.addPoint(x, yval, 0, lc)
        xli.append(pi)
    for yval in yvals:
        pi = gmsh.model.geo.addPoint(x, yval, -d, lc)
        xli.append(pi)

    # ribs 
    for i in range(nrib):
        gmsh.model.geo.addLine(x0i[i], x0i[i+nrib]) # vertical left
        gmsh.model.geo.addLine(x0i[i], xli[i]) # ribs upper 
        gmsh.model.geo.addLine(xli[i], xli[i+nrib]) # vertical right
        gmsh.model.geo.addLine(x0i[i+nrib], xli[i+nrib]) # ribs lower       

    nrib_lines = list(range(1, nrib*4+1))
    rib_groups = [nrib_lines[i:i+4] for i in range(0, len(nrib_lines), 4)]

    for group in rib_groups:
        ci = gmsh.model.geo.addCurveLoop([group[0], -group[1], -group[2], group[3]]) 
        gmsh.model.geo.addPlaneSurface([ci])   

    gmsh.model.geo.synchronize()
    nrib_surfs = [e[1] for e in gmsh.model.getEntities(2)][-1]+1

    # edge beams 
    for i in range(nrib-1):
        gmsh.model.geo.addLine(x0i[i], x0i[i+1]) # ortho upper left 
        gmsh.model.geo.addLine(x0i[i+nrib], x0i[i+nrib+1]) # ortho lower left
          
    for i in range(nrib-1):
        n = nrib_lines[-1]
        ci = gmsh.model.geo.addCurveLoop([-(i*4+1), (i*2+1+n), (i*4+1+4), -(i*2+1+n+1)])  
        gmsh.model.geo.addPlaneSurface([ci]) 

    for i in range(nrib-1):
        gmsh.model.geo.addLine(xli[i], xli[i+1]) # ortho upper right 
        gmsh.model.geo.addLine(xli[i+nrib], xli[i+nrib+1]) # ortho lower right

    for i in range(nrib-1):
        n = nrib_lines[-1] + 2*(nrib-1)
        ci = gmsh.model.geo.addCurveLoop([-(i*4+3), (i*2+1+n), (i*4+3+4), -(i*2+1+n+1)])  
        gmsh.model.geo.addPlaneSurface([ci]) 

    gmsh.model.geo.synchronize()
    nebm_surfs = [e[1] for e in gmsh.model.getEntities(2)][-1]+1

    # top
    for i in range(nrib-1):
        n1 = nrib_lines[-1]
        n2 = nrib_lines[-1] + 2*(nrib-1)
        ci = gmsh.model.geo.addCurveLoop([(i*4+2), (i*2+1+n2), -(i*4+2+4), -(i*2+1+n1)])  
        gmsh.model.geo.addPlaneSurface([ci]) 

    gmsh.model.geo.synchronize()
    top_surfs = [e[1] for e in gmsh.model.getEntities(2)][-1]+1

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    if '-vis' in sys.argv or vis == True:
        gmsh.fltk.run() 

    # ranges for indexing surfaces 
    rrib = range(1, nrib_surfs)
    rbeam = range(nrib_surfs, nebm_surfs)
    rtop = range(nebm_surfs, top_surfs)

    elementType = gmsh.model.mesh.getElementType("triangle", 1)

    # get sets of faces 
    rib_faces = []
    for i in rrib: 
        faceNodes = gmsh.model.mesh.getElementFaceNodes(elementType, 3, i)
        rib_faces.append(faceNodes)
        # rib_faces = np.append(rib_faces, faceNodes)
    # rib_faces = np.reshape(rib_faces, (int(len(rib_faces)/3), 3))

    beam_faces = np.array([])
    for i in rbeam: 
        faceNodes = gmsh.model.mesh.getElementFaceNodes(elementType, 3, i)
        beam_faces = np.append(beam_faces, faceNodes)
    beam_faces = np.reshape(beam_faces, (int(len(beam_faces)/3), 3))

    top_faces = np.array([])
    for i in rtop: 
        faceNodes = gmsh.model.mesh.getElementFaceNodes(elementType, 3, i)
        top_faces = np.append(top_faces, faceNodes)
    top_faces = np.reshape(top_faces, (int(len(top_faces)/3), 3))

    # all nodes and faces
    _, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elementType = gmsh.model.mesh.getElementType("triangle", 1)
    faceNodes = gmsh.model.mesh.getElementFaceNodes(elementType, 3)

    nodes = np.reshape(nodeCoords, (int(len(nodeCoords)/3), 3))
    all_faces = np.reshape(faceNodes, (int(len(faceNodes)/3), 3))

    if '-nodpo' in sys.argv:
        if len(nodes) > 1000: 
            raise ValueError('***** >1000 ({0}) nodes in gmsh mesh*****' .format(len(nodes)))
    
    if output: 
        print('***** Gmsh mesh: {0} nodes *****' .format(len(nodes)))
        print()

    gmsh.finalize()

    faces = [all_faces, rib_faces, beam_faces, top_faces]

    return nodes, faces 

#=========================================
# model with compas_fea 
#=========================================

def model_compas_fea(nodes, faces, parameters):

    if not output: blockPrint()

    tck_beam = parameters.get_tck_beam()
    tck_top = parameters.get_tck_top()
    beam_depth = parameters.get_beam_depth()
    b, l = parameters.get_len_x(), parameters.get_len_y()
    nrib = parameters.get_nrib()

    all_faces, beam_faces, top_faces = (faces[0]-1).tolist(), (faces[2]-1).tolist(), (faces[3]-1).tolist()
    # all_faces = np.array(all_faces).astype(int).tolist()
    beam_faces = np.array(beam_faces).astype(int).tolist()
    top_faces = np.array(top_faces).astype(int).tolist()

    rib_faces = np.concatenate(faces[1])
    rib_faces = np.reshape(rib_faces-1, (int(len(rib_faces)/3), 3)).tolist()

    cum_rib = list(accumulate([int(len(f)/3) for f in faces[1]]))
    cum_rib.insert(0,0)

    # Structure
    mdl = Structure(name=name, path='C:/Temp/')
    d = beam_depth

    mdl.add_nodes(nodes=nodes)
    # all_faces = (faces[0]-1).tolist() # gmsh indexing starts from 1
    # mdl.add_elements(elements=all_faces, type='ShellElement')
    
    mdl.add_elements(elements=rib_faces, type='ShellElement')
    num_rib = mdl.element_count() 
    select_rib = [i for i in range(0, num_rib)] 

    mdl.add_elements(elements=beam_faces, type='ShellElement')
    num_beam = mdl.element_count() 
    select_beam = [i for i in range(num_rib, num_beam)] 

    mdl.add_elements(elements=top_faces, type='ShellElement')
    num_top = mdl.element_count() 
    select_top = [i for i in range(num_beam, num_top)] 

    # num_faces = mdl.element_count() 
    # select_faces = [i for i in range(0, num_faces)] 

    # Sets 
    support_ind = [i for i, node in enumerate(nodes)
                if (node[2] == -d and ((node[0] == 0 or node[0] == l) and (node[1] == 0 or node[1] == l)))]
    internal_ind = [i for i, node in enumerate(nodes) if node[2] == 0]
    int_half_ind = [i for i, node in enumerate(nodes) if node[2] == 0 and 0 <= node[0] <= b/2]

    # get edge beam faces excluding ends 
    ign = min(b, l) 
    select_beam2 = list(set([face_ind for face_ind in select_beam
                for node_ind in mdl.elements[face_ind].nodes
                if (1/ign)*l < mdl.nodes[node_ind].y < (1-1/ign)*l]))

    # # get edge beam middle faces 
    ign = 2.5 
    select_beam3 = list(set([face_ind for face_ind in select_beam
                for node_ind in mdl.elements[face_ind].nodes
                if (1/ign)*l < mdl.nodes[node_ind].y < (1-1/ign)*l]))

    # get top faces excluding edge 
    ign = min(b, l) 
    select_top2 = list(set([face_ind for face_ind in select_top
                for node_ind in mdl.elements[face_ind].nodes 
                if (1/ign)*l < mdl.nodes[node_ind].y < (1-1/ign)*l and 
                    (1/ign)*b < mdl.nodes[node_ind].x < (1-1/ign)*b and 
                    np.abs(mdl.nodes[node_ind].z - 0) < 1e-6]))
    
    # get rib faces excluding ends 
    ign = min(b, l)  
    select_rib2 = [(list(set([face_ind for face_ind in select_rib
                for node_ind in mdl.elements[face_ind].nodes
                if (1/ign)*l < mdl.nodes[node_ind].x < (1-1/ign)*l and 
                np.abs(mdl.nodes[node_ind].y - i*l/(nrib-1)) < 1e-6]))) for i in range(nrib)]  

    # # get rib middle faces 
    ign = 2.5  
    select_rib3 = [(list(set([face_ind for face_ind in select_rib
                for node_ind in mdl.elements[face_ind].nodes
                if (1/ign)*l < mdl.nodes[node_ind].x < (1-1/ign)*l and 
                np.abs(mdl.nodes[node_ind].y - i*l/(nrib-1)) < 1e-6]))) for i in range(nrib)]  
    
    select = [select_top2, select_beam2,  select_beam3, select_rib2, select_rib3] # select faces to get forces from [for design]

    mdl.add_set(name='nset_pinned', type='node', selection=support_ind[0])
    mdl.add_set(name='nset_roller', type='node', selection=support_ind[1:])
    mdl.add_set(name='nset_internal', type='node', selection=internal_ind)
    mdl.add_set(name='nset_int_half', type='node', selection=int_half_ind)

    mdl.add_set(name='elset_beam', type='element', selection=select_beam)
    mdl.add_set(name='elset_top', type='element', selection=select_top)

    cum_rib_range = [range(cum_rib[i], cum_rib[i+1]) for i in range(len(cum_rib)-1)]
    select_rib = [[i for i in rib_range] for rib_range in cum_rib_range]
    [mdl.add_set(name='elset_rib'+str(i), type='element', selection=select) for i, select in enumerate(select_rib)]

    # Materials
    mdl.add(ElasticIsotropic(name='mat_elastic', E=30*10**9, v=0.2, p=2400))

    # Sections
    mdl.add(ShellSection(name='sec_top', t=tck_top))
    mdl.add(ShellSection(name='sec_beam', t=tck_beam[-1]))
    for i in range(len(select_rib)):
        mdl.add(ShellSection(name='sec_rib'+str(i), t=tck_beam[i])) 

    # Properties
    mdl.add([
        Properties(name='ep_beam', material='mat_elastic', section='sec_beam', elset='elset_beam'),
        Properties(name='ep_top', material='mat_elastic', section='sec_top', elset='elset_top'),
    ])
    [mdl.add(Properties(name='ep_rib'+str(i), material='mat_elastic', section='sec_rib'+str(i), elset='elset_rib'+str(i))) for i in range(len(select_rib))] 

    # Loads 
    live_load = 2.5 * l * b * 1000 / len(internal_ind) 
    mdl.add(PointLoad(name='load_udl', nodes='nset_internal', z=-live_load)) 
    mdl.add(PointLoad(name='load_half', nodes='nset_int_half', z=-live_load)) 
    mdl.add(GravityLoad(name='load_gravity', elements=['elset_top', 'elset_beam'] + [('elset_rib'+str(i)) for i in range(len(select_rib))])) 

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

    # Run
    if '-run' in sys.argv:
        mdl.analyse_and_extract(software='abaqus', fields=['u', 'sf', 'sm', 's', 'rf', 'cf'], cpus=1)
        mdl.save_to_obj() 

    # Summary
    # mdl.summary()

    mdl = load_from_obj('C:/Temp/' + name + '.obj')

    if not output: enablePrint() 

    return mdl, select 


#=========================================
# design with python Design() object  
#=========================================

def design_python(mdl, select, parameters):

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
    tck_beam = parameters.get_tck_beam()[-1]
    tck_rib = parameters.get_tck_beam()[:-1]
    tck_top = parameters.get_tck_top()
    fck = parameters.get_fck()
    nrib = parameters.get_nrib()
    
    select_top2, select_beam2, select_beam3 = select[0], select[1], select[2]
    select_rib2, select_rib3 = select[3], select[4]

    volume_steel = [0, 0]
    volume_concrete_ribs = np.sum([tck_rib[i] * np.max((beam_depth - tck_top), 0) * len_x for i in range(nrib)])
    volume_concrete = len_y * len_x * tck_top + 2 * (len_y * np.max((beam_depth - tck_top) * tck_beam, 0)) + volume_concrete_ribs
    lv0 = 2 * ((beam_depth - 2*0.03) + (tck_beam - 2*0.03)) # length shear link mm 
    lv1 = np.mean([2 * ((beam_depth - 2*0.03) + (tck_rib[i] - 2*0.03)) for i in range(nrib)])

    penalty = [0, 0]

    for i, step in enumerate(['step_udl']):

        design = Design(mdl, step) 

        # slab design, note abaqus sf, sm are per unit width 
        design.set_design_properties(b=1e3, d=tck_top*1e3, fck=fck) 
        as0, p0 = design.slab_bending_and_shear('slab (sag)', 'local_1', select_top2, 'min', 'abs', output)
        as1, p1 = design.slab_bending_and_shear('slab (hog)', 'local_1', select_top2, 'max', 'abs', output)
        as2, p2 = design.slab_bending_and_shear('slab (sag)', 'local_2', select_top2, 'min', 'abs', output)
        as3, p3 = design.slab_bending_and_shear('slab (hog)', 'local_2', select_top2, 'max', 'abs', output)

        # edge beam design, local 2 = global y, hence sm1 is torsion, s22 is bending stress. sf3 is shear 
        design.set_design_properties(b=tck_beam*1e3, d=beam_depth*1e3, hf=tck_top*1e3 ,fck=fck) 
        ab0, av0, p4 = design.beam_bending_and_shear('beam (sag)', 'local_2', select_beam3, select_beam2, 'abs', output) 

        # rib design, local 1 = global x, hence sm2 is torsion, s11 is bending stress. sf3 is shear 
        ab1, av1, p5 = 0, 0, 0 
        for j in range(nrib): 
            if select_rib3[j] == []: mesh_gmsh(parameters, True)
            design.set_design_properties(b=parameters.get_tck_beam()[j]*1e3, d=beam_depth*1e3, hf=tck_top*1e3 ,fck=fck) 
            ab_, av_, p_ = design.beam_bending_and_shear('beam (sag)', 'local_1', select_rib3[j], select_rib2[j], 'abs', output)
            ab1, av1, p5 = ab1+ab_, av1+av_, p5+p_

        volume_steel_ribs = (ab1 * 1e-6 * len_x) + (av1 * 1e-6 * lv1 * len_x/0.2)
        volume_steel[i] = (np.sum((as0, as1, as2, as3)) * 1e-6 * len_x * len_y) + 2 * ((ab0 * 1e-6 * len_y) + (av0 * 1e-6 * lv0 * len_y/0.2)) + volume_steel_ribs 
        penalty[i] = np.sum((p0, p1, p2, p3, p4, p5)) 

    if not output: enablePrint() 

    return np.max(volume_steel), volume_concrete, np.max(penalty)


#=========================================
# optimize with Platypus  
#=========================================

def objective_ribbed(parameters):

    """ 
    carries out the meshing, analysis and design and then computes the cost 

    """

    nodes, faces = mesh_gmsh(parameters)

    # tic()
    mdl, select = model_compas_fea(nodes, faces, parameters)
    # toc()

    volume_steel, volume_concrete, p = design_python(mdl, select, parameters)
    
    density_steel = 7850 # kg/m3 
    density_concrete = 2400 # kg/m3 

    mass_concrete = volume_concrete * density_concrete # kg 
    mass_steel = volume_steel * density_steel # kg 

    fck = parameters.get_fck()
    cement_type = parameters.get_cement_type()

    # A brief guide to calculating embodied carbon 2020 (IStructE)    
    carbon_s = 0.684 # kgCO2e/kg A1-A3 (material production) - UK: BRC EPD
    carbon_c = cement_type[str(fck)]

    # See Concept_v4_5 (TCC) 
    cost_c = 145 # pounds/m3 
    cost_s = 0.98 # pounds/kg 
    cost_form = 66 # pounds/m2 
    
    cost = (cost_c * volume_concrete) + (cost_s * mass_steel) + (cost_form * parameters.get_len_x() * parameters.get_len_y())
    carbon = (carbon_c * mass_concrete) + (carbon_s * mass_steel)

    print([cost + p, carbon + p])

    return [cost + p, carbon + p]


class Optimize_ribbed(Problem):

    def __init__(self):
        super().__init__(5+10, 2) # 3 decision variables, 2 objectives, 0 constraints     
        self.types[:] = [var0, var1, var2, var3, var4] + var5

    def evaluate(self, solution):
        beam_depth = solution.variables[0] * 1e-3
        tck_beam = solution.variables[1] * 1e-3
        tck_top = solution.variables[2] * 1e-3
        fck = cement[solution.variables[3]]
        nrib = solution.variables[4]
        tck_rib = [solution.variables[i+5] * 1e-3 for i in range(nrib)] # unpack rib thicknesses 

        parameters = Parameters(beam_depth=beam_depth, tck_beam=tck_rib+[tck_beam], tck_top=tck_top, fck=fck, len_x=9, len_y=9, nrib=nrib)
        solution.objectives[:] = objective_ribbed(parameters)


if __name__ == "__main__":

    name = 'mesh_ribbed' 
    myid = 'james'
    fitpath = 'C:/Users/'+myid+'/mres/compas_fea/data/'+name+'_fitdata.csv'
    ccpath = 'C:/Users/'+myid+'/mres/compas_fea/data/'+name+'_ccdata.csv'
    output = False # algorithm output 
    
    # test 
    tck_beam = 0.001 * np.array([223.5699548, 114.7287772, 251.5695032, 0, 1000, 0, 0, 0, 0, 0])
    parameters = Parameters(beam_depth=0.653, tck_beam=list(tck_beam)+[0.163], tck_top=0.093, fck=30, len_x=9, len_y=9, nrib=3)
    objective_ribbed(parameters)

    cement = [20, 25, 28, 30, 32, 35, 40, 45, 50] # cement types 
    var0 = Real(80, 900) # -> x1: beam depth (all) 
    var1 = Real(80, 600) # -> x2: beam tck (edge beams) 
    var2 = Real(80, 600) # -> x3: slab tck 
    var3 = Integer(0, 8)  # -> x4: fck 
    var4 = Integer(2, 10)  # -> x5: nrib 
    var5 = [Real(80, 600) for i in range(10)] # -> x6: rib tck 

    function_evals = 2000
    population_size, crossover, mutation = 10, 0.6, 0.01

    # run optimization 
    if '-opt' in sys.argv: 

        tic()
        
        algorithm = NSGAII(Optimize_ribbed(), population_size=population_size, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip())) 
        algorithm.crossover_probability = crossover  # crossover rate
        algorithm.mutation_probability = mutation  # mutation rate  

        # hyperparam study 
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

        variable0 = [solution.variables[0:3] for solution in algorithm.result if solution.objectives[0] < 1e98]
        variable3 = [var3.decode(solution.variables[3]) for solution in algorithm.result if solution.objectives[0] < 1e98]
        variable4 = [var4.decode(solution.variables[4]) for solution in algorithm.result if solution.objectives[0] < 1e98]
        variable5 = [solution.variables[5:16] for solution in algorithm.result if solution.objectives[0] < 1e98]

        variables = [var0 + [var3] + [var4] + var5 for var0, var3, var4, var5 in zip(variable0, variable3, variable4, variable5)]
        ccdata = np.column_stack((objectives, variables, np.ones(len(variables)) * runtime))
        save_to_csv(ccpath, ccdata, name)

        # fitness_vs_iterations data
        fitdata = np.concatenate((archives[:,0,:], archives[:,1,:]), axis=1)
        save_to_csv(fitpath, fitdata, name)


    if '-plot' in sys.argv:

        # fitness_vs_iterations data 
        fitdata = load_from_cvs(fitpath)
        iters, fitdata = clean_fitness_data(fitdata)
        # plot_fitdata(iters, fitdata)

        # cost/carbon data
        ccdata = load_from_cvs(ccpath)
        plot_ccdata(ccdata, cement, ribbed=True)
