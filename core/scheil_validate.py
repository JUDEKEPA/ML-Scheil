import itertools
import os, json, h5py
import random
from collections import OrderedDict

import numpy as np

from utils.analyzing_tool import *
from core.scheil_calc import ScheilCalculation


class ScheilValidate(ScheilCalculation):
    """
    Validation with ML-models.

    All result data structure is the same with the structure calculated in MatLab API. For easier comparison.

    """
    def __init__(self, proj_root_path, task_name):
        super().__init__(proj_root_path, task_name)
        self.__proj_root_path = proj_root_path

        self.all_data_for_validation = None
        self.allowed_phase = None
        self.path_num = None
        self.all_validation_result = None
        self.valid_path = None

    def load_validate_data(self, data_path=None):
        """
        Load the data for validation. The data structure for MatLab API is also available for this part.

        """
        if data_path is None:
            data_path = os.path.join(self.__proj_root_path, 'all tasks', self.task_name, 'data to validate.json')

        with open(data_path, 'r') as f:
            self.all_data_for_validation = json.load(f)

        self.allowed_phase = self.all_data_for_validation['allowed_phase']
        self.path_num = len(self.all_data_for_validation) - 1

    def validate_all(self):
        """
        Validate all paths.

        Notes
        -----
        The validation result will be stored in self.all_validation_result.
        """
        self.all_validation_result = dict()
        self.valid_path = dict()

        for i in range(self.path_num):
            single_path_data = self.all_data_for_validation[f'path{i}']
            single_path_validation_result, all_allowed = self.single_path_validate(single_path_data)

            self.all_validation_result[f'path{i}'] = single_path_validation_result

            if all_allowed:
                self.valid_path[f'path{i}'] = single_path_validation_result

    def single_path_validate(self, single_path_data, t_step=4):
        """
        Validate a single path.

        Parameters
        ----------
        single_path_data : dict
            The data for a single path.
        t_step : int
            The temperature step for calculation.

        """
        compositions = np.array(single_path_data['validate_points'])
        single_path_result = self.batch_comp_calculation(compositions, t_step)

        single_path_validation_result = dict()

        valid_path = True

        for i, (comp, each_point) in enumerate(zip(compositions, single_path_result)):
            phases_in_solid, phase_fracs_in_solid = overview_solidification(each_point)

            point_result = dict()
            point_result['composition'] = comp.tolist()
            point_result['phases_in_solid'] = phases_in_solid
            point_result['phase_fracs_in_solid'] = phase_fracs_in_solid

            single_path_validation_result[f'point{i+1}'] = point_result

            if not valid_path:
                continue
            else:
                valid_path = self.if_all_allowed_phase(single_path_validation_result)

        return single_path_validation_result, valid_path

    def if_all_allowed_phase(self, single_path_validation_result):
        """
        Check if all phases in solid are allowed.

        Parameters
        ----------
        single_path_validation_result : dict
            The validation result for a single path.

        """
        for point in single_path_validation_result:
            phases_in_solid = single_path_validation_result[point]['phases_in_solid']

            for phase in phases_in_solid:
                if phase not in self.allowed_phase:
                    return False

        return True


