import os
import json
import shutil
from collections import OrderedDict

import h5py
import numpy as np
# from numba import jit
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from utils.customclass import NormalizeActivation


class Conditions:
    """
    This class is used to set and store all conditions for parallel scheil model calculation.

    All functions in this class are set as 'classmethod'. For each high-dimensional phase diagram, only one series
    conditions is permitted.

    Conditions will be saved in dict. The consists of each condition is key: (var_type: str, value). For now, to types
    are appointed.

    'CHANGING' type: tuple('CHANGING', list([start, end, step]))
        It stored appointed changing conditions of changing variables.

    'FIX' type: tuple('FIX', value)
        It stored appointed fix value of fix variables.

    Simple constraints will be checked by function 'check_conditions'. But the value will not be checked rigorously.
    """

    fix_el_num = 0  # The number of fix variables
    chg_el_num = 0  # The number of changing variables
    t_step = None
    con_dict = dict()  # Store all conditions
    xn_cons, xn_chg_cons = None, None  # Store conditions orderly
    shape = None  # Store the shape of grid

    @classmethod
    def set_condition(cls, *args: tuple):
        """
        Set conditions for system.

        Parameters
        ----------
        args: tuple
            all conditions need to set.

        Notes
        -----
        The format of args is rigorous. The format is: (var: str, var_type: str, value).
        var_type == 'CHANGING':
            value: list([start, end, step])

        var_type == 'FIX':
            value: float

        Examples
        --------
        >>> Conditions.set_condition(('Fe', 'FIX', 0.2), ('Cr', 'CHANGING', [0.1, 0.3, 0.01]))

        """
        for con in args:
            if con[0].upper() in cls.con_dict:
                raise KeyError('{0} is already set'.format(con[0].upper()))
            else:
                cls.con_dict[con[0].upper()] = (con[1].upper(), con[2])
                if con[1].upper() == 'FIX':
                    cls.fix_el_num += 1
                elif con[1].upper() == 'CHANGING':
                    cls.chg_el_num += 1
                else:
                    raise TypeError(
                        'The type of {0} should belong to either \'FIX\' or \'CHANGING\'.'.format(con[0].upper()))

    @classmethod
    def change_condition(cls, *args: tuple):
        """
        Change set conditions for system.

        Parameters
        ----------
        args: tuple
            all conditions need to change.

        Note
        ----
        The format of args is rigorous. The format is: (var: str, var_type: str, value).
        var_type == 'CHANGING':
            value: list([start, end, step])

        var_type == 'FIX':
            value: float

        Examples
        --------
        >>> Conditions.change_condition(('Fe', 'FIX', 0.1), ('Cr', 'CHANGING', [0.1, 0.3, 0.05]))

        """
        for con in args:
            if con[0].upper() not in cls.con_dict:
                raise KeyError('{0} condition should be set before change'.format(con[0].upper()))
            else:
                con_type, val = cls.con_dict.pop(con[0].upper())
                if con_type == 'FIX':
                    cls.fix_el_num -= 1
                elif con_type == 'CHANGING':
                    cls.chg_el_num -= 1

                cls.set_condition(con)

    @classmethod
    def check_conditions(cls, elements):
        """
        Check if conditions match the dimensions of system.

        Parameters
        ---------
        elements: list
            The list of all elements in system.

        """

        # One is depending variable and the others need to be fixed or changing.
        if cls.fix_el_num + cls.chg_el_num != len(elements) - 1:
            raise ValueError('The number of fix variables and changing variables should be equal to number of '
                             'total variables - 1.')

        if cls.t_step is None:
            raise ValueError('Temperature step should be set.')

        cls.xn_order_cons(elements)

    @classmethod
    def set_t_step(cls, t_step):
        """
        Set time step for system.

        Parameters
        ----------
        t_step: float
            The time step for system.

        Examples
        --------
        >>> Conditions.set_t_step(4)

        """
        cls.t_step = t_step

    @classmethod
    def xn_order_cons(cls, elements):
        """
        Organize OrderedDict for conditions.

        Returns
        -------
        xn_cons: OrderedDict
            Store conditions of all variables orderly.
        xn_chg_cons: OrderedDict
            Store conditions of CHANGING variables orderly.
        """
        xn_cons = OrderedDict()
        xn_chg_cons = OrderedDict()
        cons_dict = cls.con_dict

        for var in elements:
            each_var = []
            if var in cons_dict:
                if cons_dict[var][0] == 'CHANGING':
                    each_var.append(cons_dict[var][0])
                    each_var.append(cons_dict[var][1])
                    # Ensuring it match the generated grid.
                    each_var.append(
                        int((cons_dict[var][1][1] - cons_dict[var][1][0]) / cons_dict[var][1][2] + 1e-4) + 1)

                    xn_cons[var] = each_var
                    xn_chg_cons[var] = each_var
                elif cons_dict[var][0] == 'FIX':
                    each_var.append(cons_dict[var][0])
                    each_var.append(cons_dict[var][1])

                    xn_cons[var] = each_var

            else:
                xn_cons[var] = ['DEPEND']  # Depending variable is announced here

        shape = [value[2] for (key, value) in xn_chg_cons.items()]
        cls.xn_cons, cls.xn_chg_cons, cls.shape = xn_cons, xn_chg_cons, shape

    @classmethod
    def condition_storage(cls, path):
        """
        Store conditions in json file.

        Parameters
        ----------
        path: str
            The path to store conditions.
        """
        attr_keys = ['xn_cons', 'xn_chg_cons', 'shape', 't_step']
        all_cons = {key: getattr(cls, key) for key in attr_keys}

        with open(path, 'w') as f:
            json.dump(all_cons, f)


class CombinedModel:
    """
    This class is used to load all models for Scheil calculation. The classification model and composition models are
    organized in this class.

    """
    def __init__(self, proj_root_path):
        self.__proj_root_path = proj_root_path

        self.classification_model = None
        self.composition_models = []

        self.load_all_models()

    def load_all_models(self):
        """
        Load all models for Scheil calculation.

        Notes
        -----
        The classification model and composition models are loaded from 'classification model' and 'composition model'
        in organized path according to the project root path.

        """
        phase_kinds_path = os.path.join(self.__proj_root_path, 'phase kinds.json')
        with open(phase_kinds_path, 'r') as f:
            phase_kinds = json.load(f)['phase kinds']

        cl_path = os.path.join(self.__proj_root_path, 'classification model', 'classification model')
        if os.path.exists(cl_path):
            self.classification_model = load_model(cl_path)
        else:
            raise FileNotFoundError('Classification model is not found')

        cp_path = os.path.join(self.__proj_root_path, 'composition model')
        if os.path.exists(cp_path):
            for i in range(len(phase_kinds)):
                model = load_model(os.path.join(cp_path, f'composition model{i}'),
                                   custom_objects={'NormalizeActivation': NormalizeActivation})
                self.composition_models.append(model)
        else:
            raise FileNotFoundError('Composition models are not found')


def int_chg_var_arr(chg_var_arr):
    """
    Make inputs grid into integer according to start, end, and step of variables.

    """
    for idx, (key, value) in enumerate(Conditions.xn_chg_cons.items()):
        chg_var_arr[:, idx] = (chg_var_arr[:, idx] - value[1][0]) / value[1][
            2] + 0.4  # In case a float like 2.9999998

    chg_var_arr = np.asarray(chg_var_arr, dtype=int)

    return chg_var_arr


class ScheilCalculation(CombinedModel):
    """
    This class is used to perform Scheil calculation.

    The parallel calculation and single or batch calculation are supported. They are conflicting with each other.
    It recommended to use different instances for different calculation modes.
    """
    def __init__(self, proj_root_path, task_name):
        super().__init__(proj_root_path)

        # Some information for Scheil calculation
        self.__proj_root_path = proj_root_path
        self.task_name = task_name
        self.elements = None
        self.elements_num = None
        self.phase_kinds = None
        self.phase_num = None
        self.T_range = None
        self.t_step = None

        # Used for parallel calculation
        self.depend_var_idx = None
        self.grid_input = None
        self.val_comp_num = None
        self.cha_var_arr = None
        self.liquid_composition = None
        self.only_liquid_array = None
        self.left_liquid = None
        self.reach_solidification = None
        self.solid_frac = None
        self.solid_phase_frac = None  # The sum of solid phase fraction
        self.solidification_data = None
        self.T_norm = None
        self.int_t = None
        self.t_step_norm = None
        self.t_step_num = None

        # Single cal data is important for visualization
        self.single_calc_data = None

        # Figures
        self.ax = None


        with open(os.path.join(proj_root_path, 'base infor.json'), 'r') as f:
            base_infor = json.load(f)
            self.elements = base_infor['elements']
            self.elements_num = len(self.elements)
            self.T_range = base_infor['T_range']
            self.delta_t = self.T_range[1] - self.T_range[0]

        with open(os.path.join(proj_root_path, 'phase kinds.json'), 'r') as f:
            self.phase_kinds = json.load(f)['phase kinds']
            self.phase_num = len(self.phase_kinds)

        self.liquid_index = self.phase_kinds.index('LIQUID')

        if not os.path.exists(os.path.join(proj_root_path, 'all tasks')):
            os.makedirs(os.path.join(proj_root_path, 'all tasks'))

    @staticmethod
    def set_condition(*args: tuple):
        """
        Set conditions for system.

        Parameters
        ----------
        args: tuple
            all conditions need to set.

        Note
        ----
        The format of args is rigorous. The format is: (var: str, var_type: str, value).

        var_type == 'CHANGING':
            value: list([start, end, step])

        var_type == 'FIX':
            value: float

        Examples
        --------
        >>> ScheilCalculation.set_condition(('Fe', 'FIX', 0.2), ('Cr', 'CHANGING', [0.1, 0.3, 0.01]))

        """

        Conditions.set_condition(*args)

    @staticmethod
    def change_condition(*args):
        """
        Change set conditions for system.

        Parameters
        ----------
        args: tuple
            all conditions need to change.

        Note
        ----
        The format of args is rigorous. The format is: (var: str, var_type: str, value).

        var_type == 'CHANGING':
            value: list([start, end, step])

        var_type == 'FIX':
            value: float

        Examples
        --------
        >>> ScheilCalculation.change_condition(('Fe', 'FIX', 0.1), ('Cr', 'CHANGING', [0.1, 0.3, 0.05]))

        """
        Conditions.change_condition(*args)

    @staticmethod
    def set_t_step(t_step):
        """
        Set time step for system.

        Parameters
        ----------
        t_step: float
            The temperature step for system.

        Examples
        --------
        >>> ScheilCalculation.set_t_step(4)

        """
        Conditions.set_t_step(t_step)

    def check_condition(self):
        """
        Check the dimension constraints for high-dimensional phase diagram. If pass check, conditions will be
        stored automatically.

        Notes
        -----
        Task folder and conditions.json will be created and stored if conditions check pass. This is an important step
        before the scheil calculation.

        An overwrite check will be performed if the task already exists. This mode is to prevent the loss of data and
        convenient for the user to adjust the program.

        xn_cons, xn_chg_cons, and __shape will be assigned if pass check.

        """
        Conditions.check_conditions(self.elements)

        if not os.path.exists(os.path.join(self.__proj_root_path, 'all tasks', self.task_name)):
            os.makedirs(os.path.join(self.__proj_root_path, 'all tasks', self.task_name))
        else:
            print('The task already exists')
            overwrite = input('Do you want to overwrite it? (y/n): ')
            if overwrite.lower() == 'y':
                try:
                    shutil.rmtree(os.path.join(self.__proj_root_path, 'all tasks', self.task_name))
                except Exception as e:
                    print(f"Error: {e}")

                os.makedirs(os.path.join(self.__proj_root_path, 'all tasks', self.task_name))
            else:
                raise FileExistsError('The task already exists')

        Conditions.condition_storage(os.path.join(self.__proj_root_path, 'all tasks', self.task_name, 'conditions.json'))

    def generate_inputs(self):
        """
        According to Conditions generate inputs array.

        Notes
        -----
        self.grid_input will be assigned. It is all data need to calculate.

        The memory occupation seems not to eliminate within the function. Some redundant variables are deleted to
        release memory.

        """
        grid = []
        for var in self.elements:
            if var in Conditions.con_dict:
                if Conditions.con_dict[var][0] == 'FIX':
                    grid.append(np.atleast_1d(np.array(Conditions.con_dict[var][1])))
                elif Conditions.con_dict[var][0] == 'CHANGING':
                    grid.append(np.atleast_1d(np.arange(Conditions.con_dict[var][1][0],
                                              Conditions.con_dict[var][1][1] + 1e-5,
                                              Conditions.con_dict[var][1][2])))
            else:
                self.depend_var_idx = self.elements.index(var)
                grid.append(np.atleast_1d(np.array(0)))

        grid = np.meshgrid(*grid, indexing='ij')
        # Concat flatten array for each variable
        grid_2d = np.concatenate([np.expand_dims(each_var_arr.reshape(-1), axis=1) for each_var_arr in grid], axis=1)
        del grid

        # Check mass
        over_mass = np.sum(grid_2d, axis=1) > 1 + 1e-5
        valid_grid = np.delete(grid_2d, over_mass, axis=0)
        del grid_2d, over_mass

        # Calculate dependent component mass
        valid_grid[:, self.depend_var_idx] = abs(np.reshape(1 - np.sum(valid_grid, axis=1), newshape=(-1)))

        self.grid_input = valid_grid
        self.val_comp_num = self.grid_input.shape[0]

    def integer_inputs(self):
        """
        Make the values of CHANGING condition row of input array into integers, for storage and search
        algorithms.

        """
        chg_var_idx = [Conditions.xn_cons[var][0] == 'CHANGING' for var in self.elements]
        chg_var_arr = self.grid_input[:, chg_var_idx]

        cha_var_arr = int_chg_var_arr(chg_var_arr)

        self.cha_var_arr = cha_var_arr

    def initialize_scheil(self, independent_calc=False):
        """
        Initialize Scheil calculation.
        
        Parameters
        ----------
        independent_calc: bool
            If True, the liquid composition will be generated independently. Otherwise, the liquid composition will be
            generated according to the conditions.

        """
        if not independent_calc:
            self.generate_inputs()
            self.integer_inputs()
            self.liquid_composition = self.grid_input

        self.left_liquid = np.full((self.val_comp_num, 1), 1, dtype='float64')
        self.t_step_norm = Conditions.t_step / self.delta_t
        self.t_step_num = int(self.delta_t / Conditions.t_step + 2.4)
        self.solid_frac = np.zeros((self.val_comp_num, self.t_step_num), dtype=np.float64)
        self.reach_solidification = np.zeros((self.val_comp_num, 1), dtype=np.int)
        self.solidification_data = [[] for _ in range(self.val_comp_num)]

    def phase_type_predict(self, inputs):
        """
        Predict phase type for each composition.
        
        Parameters
        ----------
        inputs: np.ndarray
            The input array for classification model.
            
        Returns
        -------
        phase_classify_result: np.ndarray
            The phase classification result.

        """
        phase_classify_result = self.classification_model.predict(inputs)
        phase_classify_result = (phase_classify_result > 0.5).astype(int)

        return phase_classify_result

    def only_liquid(self, class_result):
        """
        Get the only liquid array.
        
        Parameters
        ----------
        class_result: np.ndarray
            The phase classification result.
        
        Notes
        -----
        The only liquid array is used to determine if the composition is only liquid. The values of only liquid 
        composition are 1, others are 0.
        
        Returns
        -------
        only_liquid_array: np.ndarray
            The only liquid array.
            
        """
        only_liquid_array = np.zeros(shape=(class_result.shape[0], 1))
        for i in range(class_result.shape[0]):
            if class_result[i, self.liquid_index] == 1 and sum(class_result[i, :]) == 1:
                only_liquid_array[i, 0] = 1

        return only_liquid_array

    def parallel_calc_scheil(self, independent_calc=False):
        """
        Parallel Scheil calculation.
        
        Notes
        -----
        This is core algorithm for Scheil calculation. It provides a parallel calculation for all compositions which
        largely increase the efficiency of calculation. The algorithm is well explained in the paper

        'xxxx xxxx'.

        The order of the functions in this function is important.
        
        Parameters
        ----------
        independent_calc: bool
            If True, the liquid composition will be generated independently. Otherwise, the liquid composition will be
            generated according to the conditions.
        
        """

        self.initialize_scheil(independent_calc=independent_calc)
        self.T_norm = 1
        while self.T_norm > 0 + 1e-5:
            print(self.T_norm)
            self.int_t = int(self.T_norm / self.t_step_norm + 0.4)  # Index to store data

            # For each step, get classify result, only_liquid_array, phase_frac_array, and composition result
            classify_input = np.concatenate((np.full((self.val_comp_num, 1), self.T_norm, dtype=np.float64),
                                             self.liquid_composition), axis=1)
            class_result = self.phase_type_predict(classify_input)
            only_liquid_array = self.only_liquid(class_result)
            composition_result = self.phase_composition_predict(class_result, classify_input)
            phase_frac_array = self.calculate_phase_fraction(class_result, composition_result)

            # The update of solid information should be the first step after calculation, because the
            # reach_solidification will be updated accordingly.
            self.update_solid_frac(phase_frac_array, only_liquid_array)
            self.solidification_data_storage(class_result, composition_result, phase_frac_array, only_liquid_array)

            # The update of left liquid will also update the reach_solidification. It must after the update of solid,
            # because the solid fraction and other information should use the liquid mole of last step.
            self.update_left_liq(phase_frac_array, only_liquid_array)

            # The update of reach_solidification should be the last step.
            self.update_reach_solidification(class_result)

            # The update of liquid composition for the next calculation.
            self.update_liquid_composition(only_liquid_array, composition_result)

            self.T_norm -= self.t_step_norm

    def phase_composition_predict(self, class_result, inputs):
        """
        Predict composition for each phase.

        Notes
        -----
        To improve the efficiency of calculation, the generated phase will be extracted and calculated.

        Parameters
        ----------
        class_result: np.ndarray
            The phase classification result.
        inputs: np.ndarray
            The input array for composition model.

        Returns
        -------
        composition_result: np.ndarray
            The composition result.

        """

        composition_result = np.zeros(shape=(self.val_comp_num, self.phase_num, self.elements_num), dtype=np.float64)
        for i in range(self.phase_num):
            phase_result = class_result[:, i] == 1  # Extract if generated the phase
            if any(phase_result):
                phase_index = np.where(phase_result == 1)[0]
                comp_inputs = inputs[phase_index, :]
                composition_i = self.composition_models[i].predict(comp_inputs)
                composition_result[phase_index, i, :] = composition_i

        return composition_result

    def calculate_phase_fraction(self, class_result, composition_result):
        """
        Calculate phase fraction for each phase.

        Parameters
        ----------
        class_result: np.ndarray
            The phase classification result.
        composition_result: np.ndarray
            The composition result.

        Returns
        -------
        phase_frac_array: np.ndarray
            The phase fraction array.
        """
        phase_frac_array = np.zeros(shape=(self.val_comp_num, self.phase_num), dtype=np.float64)
        for i in range(self.val_comp_num):
            element_num = composition_result.shape[-1]
            each_phase_composition_matrix = np.zeros(shape=(element_num, class_result.shape[1]))

            # Build equation matrix
            for j in range(self.phase_num):
                if class_result[i, j] == 1:
                    each_phase_composition_matrix[:, j] = composition_result[i, j, :]

            # Solve the equation with the least square method
            phase_frac_matrix_i, residuals, rank, s = np.linalg.lstsq(each_phase_composition_matrix,
                                                                      self.liquid_composition[i, :],
                                                                      rcond=None)

            phase_frac_array[i, :] = phase_frac_matrix_i

        phase_frac_array = np.clip(phase_frac_array, 0, 1)  # Clip
        phase_frac_array = phase_frac_array / (phase_frac_array.sum(axis=1, keepdims=True) + 1e-8)  # Normalize

        return phase_frac_array

    def update_solid_frac(self, phase_frac_array, only_liquid_array):
        """
        Update solid fraction for each composition.

        Parameters
        ----------
        phase_frac_array: np.ndarray
            The phase fraction array.
        only_liquid_array: np.ndarray
            The only liquid array.

        """
        # Sum of solid for the step
        self.solid_phase_frac = np.sum(np.delete(phase_frac_array, self.liquid_index, axis=1), axis=1)

        for i in range(self.val_comp_num):
            if self.reach_solidification[i, 0] == 1:  # Stop update if reach solidification
                continue
            elif only_liquid_array[i, 0] == 1:  # Keep the solid fraction of last step if only liquid
                self.solid_frac[i, self.int_t] = self.solid_frac[i, self.int_t + 1]
            else:
                self.solid_frac[i, self.int_t] = (self.solid_frac[i, self.int_t + 1] + self.left_liquid[i, 0] *
                                                  self.solid_phase_frac[i])

                # After Clip and Normalize, the result of Liquid as standard and Solid as standard are the same.

                # Liquid as standard
                # self.solid_frac[i, self.int_t] = (self.solid_frac[i, self.int_t + 1] + self.left_liquid[i, 0] *
                #                                   (1 - phase_frac_array[i, self.liquid_index]))

    def update_left_liq(self, phase_frac_array, only_liquid_array):
        """
        Update left liquid for each composition.

        Parameters
        ----------
        phase_frac_array: np.ndarray
            The phase fraction array.
        only_liquid_array: np.ndarray
            The only liquid array.
        """
        for i in range(self.val_comp_num):
            # Stop update if reach solidification or only liquid
            if self.reach_solidification[i, 0] == 1 or only_liquid_array[i, 0] == 1:
                continue
            else:
                self.left_liquid[i, 0] *= phase_frac_array[i, self.liquid_index]

    def update_reach_solidification(self, phase_result):
        """
        Update reach solidification for each composition.

        Notes
        -----
        Two conditions will be checked to determine if reach solidification. (OR condition)

        The first condition is the left liquid is less than 0.001.

        The second condition is result of the phase
        classification contains no liquid.

        Parameters
        ----------
        phase_result: np.ndarray
            The phase classification result.

        """
        for i in range(self.val_comp_num):
            if self.reach_solidification[i, 0] == 1:
                continue
            elif self.left_liquid[i, 0] <= 0.001 or phase_result[i, self.liquid_index] == 0:
                self.reach_solidification[i, 0] = 1

    def update_liquid_composition(self, only_liquid_array, composition_result):
        """
        Update liquid composition for next step calculation.

        Parameters
        ----------
        only_liquid_array: np.ndarray
            The only liquid array.
        composition_result: np.ndarray
            The composition result.

        """
        for i in range(self.val_comp_num):
            # No update if reach solidification or only liquid
            if self.reach_solidification[i, 0] == 1 or only_liquid_array[i, 0] == 1:
                continue
            else:
                self.liquid_composition[i, :] = composition_result[i, self.liquid_index, :]

    def solidification_data_storage(self, class_result, composition_result, phase_frac_array, only_liquid_array):
        """
        Store solidification data for each composition.

        Notes
        -----
        All data is structured with list and dict, which is convenient for json file storage.

        Parameters
        ----------
        class_result: np.ndarray
            The phase classification result.
        composition_result: np.ndarray
            The composition result.
        phase_frac_array: np.ndarray
            The phase fraction array.
        only_liquid_array: np.ndarray
            The only liquid array.

        """

        phase_type = self.get_type(class_result, only_liquid_array)  # Phase type of each step

        for i in range(self.val_comp_num):
            # Only update when not reach solidification and not only liquid
            if self.reach_solidification[i, 0] == 0 and only_liquid_array[i, 0] == 0:
                storage_dict = {'T': self.T_norm, 'phase type': phase_type[i, 0],
                                'solid frac': self.solid_frac[i, self.int_t] - self.solid_frac[i, self.int_t + 1]}

                # Organize each phase information
                phase_dict = {}
                for j in range(self.phase_num):
                    if class_result[i, j] == 1:
                        phase_dict[self.phase_kinds[j]] = [phase_frac_array[i, j]*self.left_liquid[i, 0],
                                                           composition_result[i, j, :].tolist()]
                storage_dict['phases'] = phase_dict

                self.solidification_data[i].append(storage_dict)

    def get_type(self, class_result, only_liquid_array):
        phase_type = np.full((self.val_comp_num, 1), 'NONE', dtype='U100')
        for i in range(self.val_comp_num):
            if self.reach_solidification[i, 0] == 0 and only_liquid_array[i, 0] == 0:
                each_type = ''
                for j in range(self.phase_num):
                    if j != self.liquid_index and class_result[i, j] == 1:
                        each_type += self.phase_kinds[j] + '+'

                phase_type[i, 0] = each_type[:-1]

        return phase_type

    def data_storage(self):
        """
        Store solidification data in json file.

        Notes
        -----
        The data will be stored in 'solidification data.json' in the task folder.

        This file will be used for further analysis and visualization. The size of the file is large.

        """
        data_path = os.path.join(self.__proj_root_path, 'all tasks', self.task_name, 'solidification data.json')
        with open(data_path, 'w') as f:
            json.dump(self.solidification_data, f)

        with h5py.File(os.path.join(self.__proj_root_path, 'all tasks', self.task_name, 'hashed data index.h5'), 'w') as f:
            f.create_dataset('hashed data index', data=self.cha_var_arr)
            f.close()

    def extract_single_calculation(self, chg_var_composition: dict):
        """
        Extract single calculation from Scheil calculation.

        Notes
        -----
        The extract data will not exactly match the input composition. The data is the calculated data closest to the
        input composition.

        Parameters
        ----------
        chg_var_composition: dict
            The composition of changing variables.

        Returns
        -------
        single_calculation: dict
            The single calculation of Scheil calculation.

        Examples
        --------
        >>> chg_var_composition = {'Fe': 0.2, 'Ni': 0.1, 'Cr': 0.1}
        >>> single_calculation = ScheilCalculation.extract_single_calculation(chg_var_composition)
        """
        chg_var_composition = {key.upper(): value for key, value in chg_var_composition.items()}

        index = []
        for (key, value) in Conditions.xn_chg_cons.items():
            index.append(int((chg_var_composition[key] - value[1][0]) / value[1][2] + 0.4))

        target_row = np.array(index)  # replace with your target row
        indices = np.where((self.cha_var_arr == target_row).all(axis=1))[0]

        if self.solidification_data[indices[0]]:
            single_calc_data = self.solidification_data[indices[0]]
        else:
            raise ValueError('The calculation is not found.')

        self.single_calc_data = single_calc_data

    def single_comp_calculation(self, composition, t_step=4):
        """
        Single composition calculation.

        Parameters
        ----------
        composition: list
            Target composition for calculation.

        Examples
        --------
        >>> ScheilCalculation.single_comp_calculation([0.2, 0.1, 0.1, 0.6])

        """

        self.set_t_step(t_step)
        self.liquid_composition = np.atleast_2d(np.array(composition))
        self.val_comp_num = 1

        self.parallel_calc_scheil(independent_calc=True)

        self.single_calc_data = self.solidification_data[0]

    def batch_comp_calculation(self, compositions, t_step=4):
        """
        Batch compositions calculation.

        Parameters
        ----------
        compositions: list
            Target compositions for calculation.
        t_step: float
            The temperature step for calculation.

        Examples
        --------
        >>> ScheilCalculation.batch_comp_calculation([[0.2, 0.1, 0.1, 0.6], [0.3, 0.1, 0.1, 0.5]])

        """
        self.set_t_step(t_step)
        self.liquid_composition = np.array(compositions)
        self.val_comp_num = len(compositions)

        # self.initialize_scheil()
        self.parallel_calc_scheil(independent_calc=True)

        return self.solidification_data

    def draw_solidification_curve(self):
        """
        Draw solidification curve for single calculation data.

        Notes
        -----
        Only single calculation data is supported. The data should be extracted by extract_single_calculation method or
        use single_comp_calculation method.

        """
        solid_fraction = []
        y_temperature = []
        label = []
        sum_solid_frac = 0

        # Build each axis data
        for i in self.single_calc_data:
            sum_solid_frac += i['solid frac']
            solid_fraction.append(sum_solid_frac)
            y_temperature.append(i['T'] * self.delta_t + self.T_range[0] - 273.15)
            label.append(i['phase type'])

        unique_labels = set(label)

        color_names = [
            '#1f77b4',  # muted blue
            '#ff7f0e',  # safety orange
            '#2ca02c',  # cooked asparagus green
            '#d62728',  # brick red
            '#9467bd',  # muted purple
            '#8c564b',  # chestnut brown
            '#e377c2',  # raspberry yogurt pink
            '#7f7f7f',  # middle gray
            '#bcbd22',  # curry yellow-green
            '#17becf',  # blue-teal
            '#aec7e8',  # light blue
            '#ffbb78',  # light orange
            '#98df8a',  # light green
            '#ff9896',  # light red
            '#c5b0d5',  # light purple
        ]

        color_names = color_names[:len(unique_labels)]
        label_to_color = dict(zip(unique_labels, color_names))  # Assign color to each phase type

        plt.figure(figsize=(5, 2.5))
        for the_label in unique_labels:
            x_subset = [solid_fraction[i] for i in range(len(solid_fraction)) if label[i] == the_label]
            y_subset = [y_temperature[i] for i in range(len(y_temperature)) if label[i] == the_label]
            plt.scatter(x_subset, y_subset, color=label_to_color[the_label],
                        label='LIQUID+' + the_label)  # The same format with Thermo-Calc

        plt.xlabel('Solid fraction', fontsize=14, fontweight='bold')
        plt.ylabel('Temperature (C)', fontsize=14, fontweight='bold')
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.gca().spines['top'].set_visible('gray')
        plt.gca().spines['right'].set_visible('gray')
        plt.gca().spines['bottom'].set_color('gray')
        plt.gca().spines['left'].set_color('gray')

        plt.legend()
        plt.show()

        return label, y_temperature

    def draw_comp_change_in_phase(self, phase, ele_list:list, new_fig=True, x_axis='solid fraction',
                                  y_axis='composition'):
        """
        Draw composition change in phase for single calculation data.

        Parameters
        ----------
        phase: str
            The target phase.
        ele_list: list
            The list of elements for visualization.
        new_fig: bool
            If True, a new figure will be created. Otherwise, the data will be added to the current figure.
        x_axis: str
            The x-axis for visualization. The default is 'solid fraction'.
        y_axis: str
            The y-axis for visualization. The default is 'composition'.

        """

        ele_fraction = [[] for _ in range(len(ele_list))]  # To support multiple elements
        t_ax = []
        s_ax = []
        solid_frac = 0
        for i in self.single_calc_data:
            solid_frac += i['solid frac']
            phase_dict = i['phases']
            if phase.upper() in phase_dict:
                for j, ele in enumerate(ele_list):
                    ele_fraction[j].append(phase_dict[phase][1][self.elements.index(ele)])

                t_ax.append(i['T'] * self.delta_t + self.T_range[0] - 273.15)
                s_ax.append(solid_frac)

        distri_dict = {'solid fraction': s_ax, 'composition': ele_fraction, 'temperature': t_ax}
        x_axis = x_axis.lower()
        y_axis = y_axis.lower()

        x_ax = distri_dict[x_axis]
        y_ax = distri_dict[y_axis]

        if new_fig:
            fig = plt.figure(figsize=(5, 2.5))
            ax = fig.add_subplot(111)
            if x_axis == 'composition':
                for (ele, ele_frac) in zip(ele_list, x_ax):
                    ax.plot(ele_frac, y_ax, label=ele)
            else:
                for (ele, ele_frac) in zip(ele_list, y_ax):
                    ax.plot(x_ax, ele_frac, label=ele)

            plt.legend()
            plt.show()
            self.ax = ax
        else:
            if x_axis == 'composition':
                for (ele, ele_frac) in zip(ele_list, x_ax):
                    self.ax.plot(ele_frac, y_ax, label=ele)
            else:
                for (ele, ele_frac) in zip(ele_list, y_ax):
                    self.ax.plot(x_ax, ele_frac, label=ele)

            plt.legend()
            plt.show()
