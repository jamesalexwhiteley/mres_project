# type: ignore
from compas_fea.utilities import process_data
import numpy as np
import math 

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Design():

    """
    Initialises Design object [to apply eurocode design equations] to mdl.results

    Parameters
        ----------
        mdl : dict
            compas_fea model 
        step : str 
            step to extract results from

    """

    def __init__(self, mdl, step):
        
            self.b = None
            self.d = None
            self.As = None
            self.hf = None
            self.fyk = None
            self.fck = None
            self.rho = None 
            self.Asw = None 
            self.axial_force = None
            self.moment = None 
            self.shear_force = None 
            self.torsional_moment = None
            self.membrane_stress = None 
            self.node_count = mdl.node_count()
            self.sf3 = mdl.get_element_results(step, 'sf3', elements='all')
            self.sf4 = mdl.get_element_results(step, 'sf4', elements='all')
            self.sf5 = mdl.get_element_results(step, 'sf5', elements='all')
            self.sm1 = mdl.get_element_results(step, 'sm1', elements='all')
            self.sm2 = mdl.get_element_results(step, 'sm2', elements='all')
            self.sxx = mdl.get_element_results(step, 'sxx', elements='all')
            self.syy = mdl.get_element_results(step, 'syy', elements='all')
            self.sxy = mdl.get_element_results(step, 'sxy', elements='all')
            self.element_nodes = [mdl.elements[i].nodes for i in range(len(mdl.elements))]
        
        
    def set_design_properties(self, b, d, hf=None, fck=30, fyk=500, fywd=500, cover=30): 

        """ 
        flexure, singly reinforced beam 

        Parameters
        ----------
        fck : float
            characteristic concrete strength (MPa)
        fyk : float
            characteristic steel strength (MPa)
        fywd : float
            design steel strength for shear reinforcement (MPa)
        b, d, cover : float 
            width, effective depth, cover (mm)

        """
    
        # vRdmax = {"fck": (vRdmax,cot2.5, vRdmax,cot1.0)}. See HowToEC2
        vRdmax_dict = {"20": (2.54, 3.68), 
                       "25": (3.10, 4.50), 
                       "28": (3.43, 4.97),
                       "30": (3.64, 5.28),
                       "32": (3.84, 5.58),
                       "35": (4.15, 6.02),
                       "40": (4.63, 6.72),
                       "45": (5.08, 7.38),
                       "50": (5.51, 8.00)}

        self.fyk = fyk
        self.fck = fck
        self.fywd = fywd
        self.b = b
        self.d = d - cover - 10 # minus 10 is approx, rebar diam not specified
        self.hf = hf
        self.cover = cover
        self.vRdmax = vRdmax_dict[str(fck)]

        # try: 
        #     if self.d + cover + 10 < 80: 
        #         raise ValueError('min depth of 80mm needed for fire')
        # except ValueError as ve:
        #     print('ValueError:', ve)

        # try: 
        #     if not 20 <= self.fck <= 50:
        #         raise ValueError('outside 20 < fck < 50')
        # except ValueError as ve:
        #     print('ValueError:', ve) 


    #=========================================
    # Strength 
    #=========================================

    def bending(self, penalty=0):

        """ 
        flexure, singly reinforced beam 
        
        Returns
        -------
        Med, As 
            design moment (kNm), area rebar (mm2/m)

        """

        Med = self.moment
        Med /= 1000 # Nm -> kNm
        fck, fyk = self.fck, self.fyk
        b, d = self.b, self.d
        delta = 1

        # try:
        #     if not 0.1 < b/d < 10: 
        #         raise ValueError('b/d outside 0.1 < x < 10. Are you sure units are correct?')
        # except ValueError as ve:
        #     print('ValueError:', ve)    

        K = np.abs(Med) * 1e6 / (fck * b * d**2)
        K2 = 0.6*delta - 0.18*delta**2 - 0.21 

        try:
            if K > K2: 
                raise ValueError('K>K2 compression reinforcement required')
        except ValueError as ve:
            print('ValueError:', ve)
            penalty = 1e99

        z = d/2 * (1 + np.sqrt(1 - 3.529 * K))
        z = min(0.95 * d, z)
        As = np.abs(Med) * 1e6 / (0.87 * fyk * z)
        rho = As * 100 / (b * d)

        self.As = As
        self.rho = rho

        return Med, As, penalty
    

    def bending_tbeam(self, penalty=0):

        """ 
        flexure, singly reinforced beam 
        
        Returns
        -------
        Med, As 
            design moment (kNm), area rebar (mm2/m)

        """

        Med = self.moment
        Med /= 1000 # Nm -> kNm
        fck, fyk, cover = self.fck, self.fyk, self.cover
        b, d, hf = self.b, self.d, self.hf
        delta = 1

        beff = 1500
        
        K = np.abs(Med) * 1e6 / (fck * beff * d**2)
        K2 = 0.6*delta - 0.18*delta**2 - 0.21 
        
        z = d/2 * (1 + np.sqrt(1 - 3.529 * K))
        z = min(0.95 * d, z)
        x = 2.5 * (d - z)

        if x < 1.25 * hf:
            _, As, penalty = self.bending()

        else: 
            Mrf = 0.57 * fck * (beff * b) * hf * (d - 0.5 * hf)
            Kf = (np.abs(Med) * 1e6 - Mrf) / (fck * b * d**2)

            try:
                if Kf > K2: 
                    raise ValueError('Kf>K2 compression reinforcement required')
            except ValueError as ve:
                print('ValueError:', ve)
                penalty = 1e99
                
            As = (Mrf / ((fyk / 1.15) * (d - 0.5 * hf))) + ((np.abs(Med) * 1e6 - Mrf) / ((fyk / 1.15) * (z)))
            rho = As * 100 / (b * d)

            self.As = As
            self.rho = rho  
                                                            
        return Med, As, penalty
            

    def shear_without_reinforcement(self, penalty=0):

        """ 
        shear, without reinforcement 

        Returns
        -------
        Ved : float 
            design shear force (kN)

        """
        Ved = self.shear_force
        Ved /= 1000 # N -> kN
        b, d, fck = self.b, self.d, self.fck

        ved = np.abs(Ved) * 1e3 / (0.9 * b * d)
        k = np.min((1 + np.sqrt(200 / d), 2))

        # try: 
        #     if self.As == None: 
        #         raise ValueError('Flexure design must be carried out before shear design')
        # except ValueError as ve:
        #     print('ValueError:', ve)
        
        pl = np.min((self.As / (b * d), 0.02))
        vRdc = 0.12 * k * (100 * pl * fck)**(1/3) 
        vRdcmin = 0.035 * k**1.5 * np.sqrt(fck)
        if vRdc < vRdcmin: vRdc = vRdcmin

        try: 
            if ved > vRdc: 
                raise ValueError('Shear capacity ({:.2f} kN) exceeded'.format(vRdc * 1e-3 * (0.9 * b * d)))
        except ValueError as ve:
            print('ValueError:', ve)
            penalty = 1e99

        self.Asw = 0

        return Ved, penalty


    def shear_with_reinforcement(self, penalty=0):

        """ 
        shear, with reinforcement 

        Returns
        -------
        Ved : float 
            design shear force (kN)
        Asw : float 
            area shear reinforcement (mm2/200mm)

        """
        Ved = self.shear_force
        Ved /= 1000 # N -> kN
        b, d, fck = self.b, self.d, self.fck

        ved = np.abs(Ved) * 1e3 / (0.9* b * d) # kN/mm2

        if ved < self.vRdmax[0]: 
            cot_theta = 2.5 
        else:
            try: 
                if ved < self.vRdmax[1]:
                    raise ValueError('ved < vRd,max cot_theta=1.0')
            except ValueError as ve:
                print('ValueError:', ve)
                penalty = 1e99

            try: 
                theta = 0.5 * math.asin(ved / (0.2 * fck * (1-fck/250)))
                cot_theta = 1/math.tan(theta)
            except ValueError as ve:
                cot_theta = 2.5
                print('ValueError: ved too high (math domain error)')        
                penalty = 1e99 

        Asw = 200 * ved * b / (self.fywd * cot_theta)
 
        self.Asw = Asw

        return Ved, Asw, penalty
    

    def torsion(self):

        """ 
        torsion, additive to other forces  
     
        Returns
        -------
        As, Asw 
            area tensile reinforcement (mm2), area shear reinforcement (mm2/m)

        """

        Ted = self.torsional_moment  
        Ted /= 1000 # Nm -> kNm
        b, d, fyk = self.b, self.d, self.fyk

        t = b * d / (2 * (b + d)) 
        Ak = (d - t) * (b - t) 
        uk = 2 * ((d - t) + (b - t))

        As = Ted * 1e6 * uk / (2 * Ak * (fyk/1.15))

        tau = Ted * 1e6 / (2 * t * Ak)
        Ved = tau * t * (d - t)
        Asw = 200 * Ved / (0.9 * d * (fyk/1.15))

        return Ted, As, Asw
    
    def membrane(self, penalty=0):

        """
        check membrane compression < crushing strength 

        """

        membrane_stress = self.membrane_stress / 1e6 # N/m2 -> N/mm2

        try: 
            if np.abs(membrane_stress) > (self.fck / 1.5):
                raise ValueError('shell fails in crushing')
        except ValueError as ve:
            print('ValueError:', ve)      
            penalty = 1e99 

        return penalty
    
    def tie(self):

        """
        calculate reinforcement to resist tie force  

        """

        self.As = self.axial_force / (0.87 * self.fyk)
    

    def detail(self, Ast=0, Aswt=0, penalty=0):

        """ 
        min and max reinforcement quantities in bending and shear
        
        """

        Asb, Aswb = self.As, self.Asw
        As, Asw = Asb + Ast, Aswb + Aswt

        # check min longitudinal reinforcement      
        fctm = 0.3 * self.fck**(2/3) 
        Asmin = 0.26 * fctm * 0.9 * self.b * self.d / self.fyk 
        if As < Asmin: As = Asmin 

        # check max longitudinal reinforcement 
        try: 
            if self.rho > 4: 
                raise ValueError('max bending reinforcement ratio is 4%Ac')
        except ValueError as ve:
            print('ValueError:', ve) 
            penalty = 1e99 

        # check min shear reinforcement 
        if Asw < 101: Asw = 101 # Asw / 2 per bar (min H8)
    
        # check max shear reinforcement 
        try: 
            if Asw > 2513:
                raise ValueError('max area shear reinforcement exceeded (for 200mm spacings)')
        except ValueError as ve:
            print('ValueError:', ve)
            penalty = 1e99

        return As, Asw, penalty


    def slab_bending_and_shear(self, name, direction, element_ids, b_iptype, v_iptype, output=True, shell=False):

        """ 
        sets the moments and shear forces to design to and calls design functions 

            note: best to call bending_and_shear() rather than either separately, 
            since As from bending() needed in shear() 

        Abaqus documentation: "23.6.7 Three-dimensional conventional shell element library"
        SF4 Transverse shear force per unit width in local 1-direction 
        SF5 Transverse shear force per unit width in local 2-direction 

        SM1 Bending moment force per unit width about local 2-axis.
        SM2 Bending moment force per unit width about local 1-axis.

        """

        # choose direction 
        if direction == 'local_1': 
            b_field, v_field = 'sm1','sf4'
            
        if direction == 'local_2': 
            b_field, v_field = 'sm2', 'sf5'
        
        # set bending moment 
        if b_field == 'sm1':
            _, elm_data = process_data(self.sm1, 'element', b_iptype, None, self.element_nodes, self.node_count)
        elif b_field == 'sm2':
            _, elm_data = process_data(self.sm2, 'element', b_iptype, None, self.element_nodes, self.node_count)

        if b_iptype == 'max':
            self.moment = np.max((elm_data[element_ids]))
        elif b_iptype == 'min':
            self.moment = np.min((elm_data[element_ids]))

        # set shear force 
        if v_field == 'sf4':
            _, elm_data = process_data(self.sf4, 'element', v_iptype, None, self.element_nodes, self.node_count)
        elif v_field == 'sf5':
            _, elm_data = process_data(self.sf5, 'element', v_iptype, None, self.element_nodes, self.node_count)

        if v_iptype == 'max' or v_iptype == 'abs':
            self.shear_force = np.max((elm_data[element_ids]))
        elif v_iptype == 'min':
            self.shear_force = np.min((elm_data[element_ids]))

        # set membrane force
        if shell: 

            if b_field == 'sm2': 
                _, elm_data = process_data(self.sxx, 'element', 'min', None, self.element_nodes, self.node_count)
                self.membrane_stress = np.min((elm_data[element_ids])) # compressive stress  

            elif b_field == 'sm1':
                _, elm_data = process_data(self.syy, 'element', 'min', None, self.element_nodes, self.node_count)
                self.membrane_stress = np.min((elm_data[element_ids]))

        # design (constraints) 
        Med, _, p0 = self.bending()

        Ved, p1 = self.shear_without_reinforcement()

        As, _, p2 = self.detail()

        if shell: p0 += self.membrane()

        if output: 
            print( name + ' ' + direction + ':    ' + b_iptype + ' moment ' + b_field + ' = ' "{:.2f}".format(Med) + ' kNm  ' + '(Asprov = ' + "{:.2f}".format(As) + ' mm2/m)    ' + \
            v_iptype + ' shear ' + v_field + ' = ' "{:.2f}".format(Ved) + ' kN' )
        
        return As, np.sum((p0, p1, p2)) 
    

    def beam_bending_and_shear(self, name, direction, b_element_ids, v_element_ids, v_iptype, output=True):

        """ 
        in plane moment must be ascertained from in-plane stress components at ips

        """

        # choose direction 
        if direction == 'local_1': 
            b_field, v_field, t_field = 'sxx', 'sf3', 'sm2'

        if direction == 'local_2': 
            b_field, v_field, t_field = 'syy', 'sf3', 'sm1'

        # get bending stress 
        if b_field == 'sxx': 
            _, elm_data = process_data(self.sxx, 'element', 'max', None, self.element_nodes, self.node_count)
            lower_sigx = np.max((elm_data[b_element_ids])) # bending stress at bottom of cross section 

            _, elm_data = process_data(self.sxx, 'element', 'min', None, self.element_nodes, self.node_count)
            upper_sigx = np.min((elm_data[b_element_ids])) # bending stress at top of cross section 

        elif b_field == 'syy':
            _, elm_data = process_data(self.syy, 'element', 'max', None, self.element_nodes, self.node_count)
            lower_sigx = np.max((elm_data[b_element_ids]))

            _, elm_data = process_data(self.syy, 'element', 'min', None, self.element_nodes, self.node_count)
            upper_sigx = np.min((elm_data[b_element_ids]))  

        # set shear force 
        _, elm_data = process_data(self.sf3, 'element', v_iptype, None, self.element_nodes, self.node_count)

        if v_iptype == 'max' or v_iptype == 'abs':
            self.shear_force = np.max((elm_data[v_element_ids]))
        elif v_iptype == 'min':
            self.shear_force = np.min((elm_data[v_element_ids]))

        # set torsional moment 
        if t_field == 'sm2':
            _, elm_data = process_data(self.sm2, 'element', 'abs', None, self.element_nodes, self.node_count)
        elif t_field == 'sm1':
            _, elm_data = process_data(self.sm1, 'element', 'abs', None, self.element_nodes, self.node_count)

        self.torsional_moment = np.max((elm_data[v_element_ids]))

        # set bending moment 
        p_ = 0

        # try: 
        #     if upper_sigx > 0:
        #         raise ValueError('tension at top of beam')
        # except ValueError as ve:
        #     print('ValueError:', ve)
        #     print(upper_sigx)
        #     print(lower_sigx)
        #     p_ = 1e99

        # try: 
        #     if lower_sigx < 0:
        #         raise ValueError('compression at bottom of beam')
        # except ValueError as ve:
        #     print('ValueError:', ve)
        #     p_ = 1e99

        if upper_sigx > 0 and lower_sigx > 0:
            # design as tie 

            self.axial_force = 0.5 * (np.abs(upper_sigx) + np.abs(lower_sigx)) * self.b * self.d
            Med, p0 = 0, 0  
            self.tie()

        elif upper_sigx < 0 and lower_sigx < 0:
            # design as strut

            print('(!) strut exists (!)') # for testing, should not be possible

        elif upper_sigx > 0 and lower_sigx < 0:
            # design as beam 
            
            print('(!) hogging in downstand beam at midspan (!)') # for testing, should not be possible  

        elif upper_sigx < 0 and lower_sigx > 0:
            # design as beam 

            upper_sigx, lower_sigx = np.abs(upper_sigx), np.abs(lower_sigx)

            y = self.d*1e-3 * upper_sigx / (upper_sigx + lower_sigx) # top of section to NA

            self.moment =   (upper_sigx * y * self.b*1e-3 / 2) * 2/3 * y + \
                            (lower_sigx * (self.d*1e-3 - y) * self.b*1e-3 / 2) * 2/3 * (self.d*1e-3 - y)

            Med, _, p0 = self.bending_tbeam()

        # design (constraints)
        Ved, _, p1 = self.shear_with_reinforcement()

        Ted, Ast, Aswt = self.torsion()

        As, Asw, p2 = self.detail(Ast, Aswt)

        if output: 
            print( name + ' ' + direction + ':    ' + 'moment ' + str(b_field) + '* = ' "{:.2f}".format(Med) + '    '
                  + 'torsion ' + str(t_field) + ' = ' "{:.2f}".format(Ted) + ' kNm  ' + '(Asprov = ' + "{:.2f}".format(As) + ' mm2/m)    ' 
                  + ' shear ' + v_field + ' = ' "{:.2f}".format(Ved) + ' kN  ' + '(Asw /200mm = ' + "{:.2f}".format(Asw) + ' mm2)' )

        return As, Asw, np.sum((p_, p0, p1, p2)) 

