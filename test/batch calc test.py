from core.scheil_calc import ScheilCalculation


proj_root_path = 'YOUR PROJECT ROOT PATH'

# 1. Set the project root path and task name
# Though the task name is not used in this snippet, it is still required to create the ScheilCalculation object.
# This may be changed in the future.
FeCrNiMn_models = ScheilCalculation(proj_root_path, 'jawfaoei')

all_composition_need_to_calculate = [[0.1, 0.1, 0.2, 0.6], [0.2, 0.1, 0.1, 0.6], [0.6, 0.2, 0.1, 0.1]] # Just an example

FeCrNiMn_models.batch_comp_calculation(all_composition_need_to_calculate, t_step=4)

