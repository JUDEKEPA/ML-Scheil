import itertools
import os, json, h5py
import random
from collections import OrderedDict

import numpy as np

from utils.analyzing_tool import *
from core.scheil_calc import ScheilCalculation


class HDLine:
    """
    Linear path for graded functionally metal design. The algorithm is based on the paper:
    
    "xxxx xxxx"
    
    """
    def __init__(self, point1, point2, shape):
        """
        
        Parameters
        ----------
        point1 : np.ndarray
            The starting point of the line
        point2 : np.ndarray
            The ending point of the line
        shape : tuple
            The shape of the composition space.
        
        """
        self.start_point = point1
        self.end_point = point2
        self.shape = shape
        self.dim = len(point1)

        self.vector = self.end_point - self.start_point

        self.points = None
        self.pts_around_line = None

    def points_on_line(self, point_num=20):
        point_weight = np.linspace(0, 1, point_num)
        points = np.outer(1 - point_weight, self.start_point) + np.outer(point_weight, self.end_point)

        self.points = points

    def points_around_line(self):
        """
        Using the points on the line to find the points around the line.
        
        Notes
        -----
        The points around the line are the points that are calculated index that is the nearest to the line.
        
        """
        pts_around_line = np.zeros((self.points.shape[0] * (2 ** self.dim), self.dim), dtype=int)
        for i, point in enumerate(self.points):
            small_int = point.astype(int)  # Lower nearest integer
            large_int = small_int + 1  # Upper nearest integer

            # Pair of lower and upper nearest integer, the box for the point on different dimension.
            pairs = list(zip(small_int, large_int))  
            combinations = np.array(list(itertools.product(*pairs)))
            
            pts_around_line[i * (2 ** self.dim):(i + 1) * (2 ** self.dim)] = combinations

        pts_around_line = self.valid_points(pts_around_line)

        self.pts_around_line = np.unique(pts_around_line, axis=0)

    def valid_points(self, pts_around_line):
        row_to_delete = np.full((pts_around_line.shape[0], self.dim), False, dtype=bool)
        for i in range(self.dim):
            row_to_delete[:, i] = np.logical_or(pts_around_line[:, i] < 0, pts_around_line[:, i] >= self.shape[i])

        return np.delete(pts_around_line, np.where(row_to_delete.any(axis=1)), axis=0)


class ScheilPhad:
    """
    The class for Scheil phase diagram.
    
    The path design is based on phase-type phase diagram.
    """
    def __init__(self, proj_root_path, task_name):
        """
        
        Notes
        -----
        Data loaded from exist task.
        
        """
        self.__proj_root_path = proj_root_path
        self.__task_name = task_name

        base_path = os.path.join(self.__proj_root_path, 'all tasks', self.__task_name)

        with open(os.path.join(base_path, 'conditions.json'), 'r') as f:
            self.conditions = json.load(f, object_pairs_hook=OrderedDict)

        with open(os.path.join(base_path, 'solidification data.json'), 'r') as f:
            self.solidification_data = json.load(f)

        with h5py.File(os.path.join(base_path, 'hashed data index.h5'), 'r') as f:
            self.hashed_index = f['hashed data index'][:]

        # base information of all data
        self.phase_type = None
        self.path_type = None
        self.sys_phase_type = None
        self.sys_path_type = None
        self.sys_path_num = None

        # two types of phase diagram
        self.high_dim_phase_type_phad = None
        self.high_dim_path_type_phad = None
        
        # valid points for path design
        self.valid_points = None
        self.allowed_phase = None
        
        # data for validation
        self.data_for_validation = None
        
        self.data_simplify()
        self.build_high_dim_phad()

    @property
    def phase_infor(self):
        return self.sys_phase_type

    @property
    def path_infor(self):
        return self.sys_path_type

    def data_simplify(self):
        """
        Initialize phase_type and path for the Scheil phase diagram, before high-dimensional array.
        
        """
        phase_type = []
        path_type = []

        for each_data in self.solidification_data:
            phase_type_each = []
            path_type_each = []
            for step in each_data:
                if step['phase type'] not in path_type_each:
                    path_type_each.append(step['phase type'])

                for phase in step['phases'].keys():
                    if phase not in phase_type_each and phase != 'LIQUID':
                        phase_type_each.append(phase)

            phase_type.append(phase_type_each)
            path_type.append(path_type_each)

        self.phase_type = phase_type
        self.path_type = path_type

        self.get_sys_phase_type()
        self.get_sys_path_type()

    def get_sys_phase_type(self):
        """
        All phase types in the calculated result.
        
        """
        sys_phase_type = []
        for each_phase_type in self.phase_type:
            for phase in each_phase_type:
                if phase not in sys_phase_type:
                    sys_phase_type.append(phase)

        self.sys_phase_type = sys_phase_type

    def get_sys_path_type(self):
        """
        All path types in the calculated result.
        
        Notes
        -----
        [''] is the failed calculation.
        
        """
        sys_path_type = []
        for each_path_type in self.path_type:
            if each_path_type not in sys_path_type and each_path_type != ['']:
                sys_path_type.append(each_path_type)

        sys_path_type, sys_path_num = sys_each_path_num(self.path_type, sys_path_type)

        self.sys_path_type = sys_path_type
        self.sys_path_num = sys_path_num

    def build_high_dim_phad(self):
        """
        Build high-dimensional phase-type and path-type phase diagram.
        
        Notes
        -----
        The core function for the Scheil phase diagram. This function made the data into high-dimensional array.
        
        """
        shape = self.conditions['shape']
        high_dim_phase_type_phad = np.empty(shape=shape, dtype=object)
        high_dim_path_type_phad = np.empty(shape=shape, dtype=object)
        
        for (index, each_phase_type, each_phase_path) in zip(self.hashed_index, self.phase_type, self.path_type):
            high_dim_phase_type_phad[tuple(index)] = each_phase_type
            high_dim_path_type_phad[tuple(index)] = each_phase_path

        self.high_dim_phase_type_phad = high_dim_phase_type_phad
        self.high_dim_path_type_phad = high_dim_path_type_phad

    def endpoint_path(self, path1, path2):
        """
        Get the data points for two paths.

        Parameters
        ----------
        path1 : list
            The first path.
        path2 : list
            The second path.

        Notes
        -----
        The path here means the solidification process path.

        Returns
        -------
        path1_data_points : np.ndarray
            The data points for the first path.
        path2_data_points : np.ndarray
            The data points for the second path.

        Examples
        --------
        >>> path1_data_points, path2_data_points = ScheilPhad.endpoint_path(['BCC_B2'], ['FCC_L12'])
        >>> path1_data_points, path2_data_points = ScheilPhad.endpoint_path(['BCC_B2', 'BCC_B2+FCC_L12'], ['FCC_L12'])

        """
        path1_data_index = []
        path2_data_index = []

        for i, each_path in enumerate(self.path_type):
            if each_path == path1:
                path1_data_index.append(i)
            elif each_path == path2:
                path2_data_index.append(i)

        path1_data_points = self.hashed_index[path1_data_index]
        path2_data_points = self.hashed_index[path2_data_index]

        return path1_data_points, path2_data_points

    def design_path(self, path1, path2, trial_num=10000, phase_not_suspend=None, scan_points=20):
        """
        Design the composition path between two solidification paths.
        
        Notes
        -----
        The algorithm is based on the paper:
        
        "xxxx xxxx"
        
        Parameters
        ----------
        path1 : list
            The first path.
        path2 : list
            The second path.
        trial_num : int
            The number of trials.
        phase_not_suspend : list
            The other allowed phase.
        scan_points : int
            The number of points on the line scan.
        
        Examples
        --------
        >>> ScheilPhad.design_path(['BCC_B2'], ['FCC_L12'])
        >>> ScheilPhad.design_path(['BCC_B2', 'BCC_B2+FCC_L12'], ['FCC_L12'])
        >>> ScheilPhad.design_path(['BCC_B2', 'BCC_B2+FCC_L12'], ['FCC_L12'], trial_num=20000, phase_not_suspend=['SIGMA'])
        
        """
        if phase_not_suspend is None:
            phase_not_suspend = list()

        # Find exist phase in two paths, and the allow these phases for the path design
        exist_phase = []
        for step in itertools.chain(path1, path2):
            phases = step.split('+')
            for phase in phases:
                if phase not in exist_phase:
                    exist_phase.append(phase)

        allowed_phase = []

        for phase in itertools.chain(exist_phase, phase_not_suspend):
            if phase not in allowed_phase:
                allowed_phase.append(phase)
                
        # Extract target data points
        path1_data_points, path2_data_points = self.endpoint_path(path1, path2)

        # get random pair and random linear path
        path1_random_pts_idx = [random.randint(0, path1_data_points.shape[0]-1) for _ in range(trial_num)]
        path2_random_pts_idx = [random.randint(0, path2_data_points.shape[0]-1) for _ in range(trial_num)]

        path1_random_pts = path1_data_points[path1_random_pts_idx]
        path2_random_pts = path2_data_points[path2_random_pts_idx]

        valid_points = np.zeros((2, trial_num, path1_data_points.shape[1]), dtype=int)

        valid_path_count = 0

        # Check each linear path
        for i, (point1, point2) in enumerate(zip(path1_random_pts, path2_random_pts)):
            print(i)

            if any(point1 == 0) or any(point2 == 0):
                continue

            line = HDLine(point1, point2, self.conditions['shape'])
            line.points_on_line(scan_points)
            line.points_around_line()
            
            # If any not allowed phase exists
            if self.validate_points_around_line(line.pts_around_line, allowed_phase):
                valid_points[0, valid_path_count] = point1
                valid_points[1, valid_path_count] = point2

                valid_path_count += 1

        valid_points = valid_points[:, :valid_path_count]

        self.valid_points = valid_points
        self.allowed_phase = allowed_phase

    def validate_points_around_line(self, points_around_line, allowed_phase):
        """
        Validate the points around the line.
        
        Parameters
        ----------
        points_around_line : np.ndarray
            The points around the line.
        allowed_phase : list
            The allowed phase.
            
        """
        for point in points_around_line:  # FAILED calculation is not considered
            phase_type = self.high_dim_phase_type_phad[tuple(point)]
            if phase_type is None:
                continue

            for phase in phase_type:
                if phase not in allowed_phase:
                    return False

        return True

    def tc_validate_storage(self, path_num_for_validation=50, points_num_on_line_for_validation=20):
        """
        Store the data for validation using developed Thermo-Calc TC-Toolbox API on MatLab.
        
        Parameters
        ----------
        path_num_for_validation : int
            The number of designed paths for validation.
            
        """
        points_for_validation = self.valid_points[:, :path_num_for_validation]
        allowed_phase = self.allowed_phase

        data_for_validation = dict()
        data_for_validation['allowed_phase'] = allowed_phase
        data_for_validation['data_num'] = path_num_for_validation

        for i in range(path_num_for_validation):
            point1 = points_for_validation[0, i]
            point2 = points_for_validation[1, i]

            line = HDLine(point1, point2, self.conditions['shape'])
            line.points_on_line(points_num_on_line_for_validation)

            # Reverse the hashed index to real value before storing
            point1 = self.reverse_to_real_val(point1)
            point2 = self.reverse_to_real_val(point2)
            points = self.reverse_to_real_val(line.points)

            name = f'path{i}'
            data_for_validation[name] = dict()
            data_for_validation[name]['endpoint1'] = point1.tolist()
            data_for_validation[name]['endpoint2'] = point2.tolist()
            data_for_validation[name]['validate_points'] = points.tolist()

        storage_path = os.path.join(self.__proj_root_path, 'all tasks', self.__task_name, 'data to validate.json')
        with open(storage_path, 'w') as f:
            json.dump(data_for_validation, f)

        self.data_for_validation = data_for_validation

    def reverse_to_real_val(self, hashed_index):
        """
        Reverse the hashed index to real value.
        
        """
        hashed_index = np.atleast_2d(hashed_index)

        xn_chg_cons = self.conditions['xn_chg_cons']
        xn_cons = self.conditions['xn_cons']

        reversed_array = np.zeros((hashed_index.shape[0], len(xn_cons)), dtype=float)

        chg_var = [key for key in xn_chg_cons.keys()]
        all_var = [key for key in xn_cons.keys()]

        for (key, value) in xn_cons.items():
            if value[0] == 'CHANGING':
                index = chg_var.index(key)
                reversed_array[:, all_var.index(key)] = hashed_index[:, index] * value[1][2] + value[1][0]
            elif value[0] == 'FIX':
                reversed_array[:, all_var.index(key)] = value[1]
            elif value[0] == 'DEPEND':
                depend_idx = all_var.index(key)

        reversed_array[:, depend_idx] = 1 - np.sum(reversed_array, axis=1)

        return reversed_array
            

