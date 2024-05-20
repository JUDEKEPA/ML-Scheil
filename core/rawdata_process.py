import os, json
import numpy as np
import pandas as pd


class CreateDataset:
    """
    Rawdata structure:

    | T | Element 1 | Element 2 | ... | Element n | generated phase 1 | composition | generated phase 2 | ... |

    For example:
    | 1000 | 0.2 | 0.3 | 0.1 | 0.4 | 'LIQUID' | 0.2 | 0.3 | 0.01 | 0.49 | 'BCC_B2' | 0.2 | 0.3 | 0.5 | 0 |
    | 1100 | 0.2 | 0.3 | 0.1 | 0.4 | 'LIQUID' | 0.2 | 0.3 | 0.01 | 0.49 |

    """
    def __init__(self, proj_root_path, elements_list=None, T_range=None):
        """
        Initialize the CreateDataset class.

        Parameters
        ----------
        proj_root_path : str
            The root path of the project.
        elements_list : list
            The list of elements.
        T_range : list
            The temperature range.

        Notes
        -----
        The proj_root_path is the root path of the project. All data and models will be stored in this path.

        """
        self.__proj_root_path = proj_root_path
        self.elements_list = None
        self.T_range = None

        if os.path.exists(self.__proj_root_path):
            base_path = os.path.join(self.__proj_root_path, 'all raw data')

            # Load former data and information if exists
            if os.path.exists(os.path.join(base_path, 'all data.npy')):
                self.all_data = np.load(os.path.join(base_path, 'all data.npy'), allow_pickle=True)
                self.exist_data = True
            else:
                self.exist_data = False

            # Load former information if exists
            if os.path.exists(os.path.join(self.__proj_root_path, 'base infor.json')):
                with open(os.path.join(self.__proj_root_path, 'base infor.json'), 'r') as f:
                    infor = json.load(f)
                    self.elements_list = infor['elements']
                    self.T_range = infor['T_range']
                    self.elements_num = len(self.elements_list)

        # Create a new project
        else:
            self.elements_list = list(map(str.upper, elements_list))  # Capitalize all elements
            self.elements_num = len(elements_list)
            self.T_range = T_range

            base_infor = {
                            'elements': elements_list,
                            'T_range': T_range
                          }

            os.makedirs(self.__proj_root_path)
            os.chdir(self.__proj_root_path)
            os.makedirs('all raw data')

            with open(os.path.join(self.__proj_root_path, 'base infor.json'), 'w') as f:
                json.dump(base_infor, f)

        if self.elements_list is None or self.T_range is None:  # Check if the elements list and T_range are assigned
            raise ValueError("For a new project or an empty project, elements list and T_range must be assigned.")

        self.phase_kinds = None
        self.inputs = None
        self.outputs_original = None

        self.classification_outputs = None
        self.discrete_dataset = None

    def add_xlsx_data(self, path_list):
        """
        Add data from xlsx files.

        Parameters
        ----------
        path_list: list
            The list of paths of xlsx files.

        Notes
        -----
        If the data already exists, the new data will be appended to the existing data. The data will be stored in the
        all_data attribute.

        """
        # Read existing data
        if self.exist_data:
            for path in path_list:
                data_temp = pd.read_excel(path)
                data_temp = np.array(data_temp)
                self.all_data = np.append(self.all_data, data_temp, axis=0)
        else:
            self.all_data = np.array(pd.read_excel(path_list.pop(0)))
            for path in path_list:
                data_temp = pd.read_excel(path)
                data_temp = np.array(data_temp)
                self.all_data = np.append(self.all_data, data_temp, axis=0)

            self.exist_data = True

    def complete_add(self):
        """
        Save the all data.

        """
        storage_path = os.path.join(self.__proj_root_path, 'all raw data', 'all data.npy')
        np.save(storage_path, self.all_data)

    def create_dataset(self):
        """
        Create the dataset.

        """
        self.create_inputs()
        self.outputs_original = self.all_data[:, self.elements_num + 1:]
        self.find_phase_kinds()
        self.create_classification_outputs()
        self.create_discrete_dataset()

    def create_inputs(self):
        inputs = self.all_data[:, :self.elements_num+1]
        inputs[:, 0] = (inputs[:, 0] - self.T_range[0]) / (self.T_range[1] - self.T_range[0])  # Normalize the temperature
        inputs = inputs.astype('float64')

        self.inputs = inputs

    def find_phase_kinds(self):
        phase_kinds = []
        for i in range(self.outputs_original.shape[0]):
            for j in range(0, self.outputs_original.shape[1], self.elements_num+1):
                if not (self.outputs_original[i, j] in phase_kinds):
                    phase_kinds.append(self.outputs_original[i, j])

        phase_kinds.pop(phase_kinds.index(0))  # remove 0
        phase_kinds.sort()  # sort the phase kinds, keep the same order

        phase_kinds_js = {'phase kinds': phase_kinds}
        with open(os.path.join(self.__proj_root_path, 'phase kinds.json'), 'w') as f:
            json.dump(phase_kinds_js, f)

        self.phase_kinds = phase_kinds

    def create_classification_outputs(self):
        outputs = []
        for i in range(self.outputs_original.shape[0]):
            temp = [0 for _ in range(len(self.phase_kinds))]
            for j in range(0, self.outputs_original.shape[1], self.elements_num + 1):
                if self.outputs_original[i, j] == 0:
                    break
                else:
                    index = self.phase_kinds.index(self.outputs_original[i, j])
                    temp[index] = 1

            outputs.append(temp)

        outputs = np.array(outputs)
        self.classification_outputs = outputs

    def create_discrete_dataset(self):
        phase_num = len(self.phase_kinds)
        each_phase_dataset = []
        for i in range(phase_num):
            temp_inputs = []
            temp_outputs = []
            for j in range(self.outputs_original.shape[0]):
                temp = []
                for k in range(0, self.outputs_original.shape[1], self.elements_num+1):
                    if self.outputs_original[j, k] == self.phase_kinds[i]:
                        temp_inputs.append(self.inputs[j, :])
                        for w in range(self.elements_num):
                            temp.append(self.outputs_original[j, k+w+1])

                        temp_outputs.append(temp)
                        break

            each_phase = []
            temp_inputs = np.array(temp_inputs)
            temp_outputs = np.array(temp_outputs)

            each_phase.append(temp_inputs)
            each_phase.append(temp_outputs)

            each_phase_dataset.append(each_phase)

        self.discrete_dataset = each_phase_dataset

