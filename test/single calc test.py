from core.scheil_calc import ScheilCalculation


proj_root_path = 'YOUR PROJECT ROOT PATH'

# 1. Set the project root path and task name
# Though the task name is not used in this snippet, it is still required to create the ScheilCalculation object.
# This may be changed in the future.
FeCrNiMn_models = ScheilCalculation(proj_root_path, 'jawfaoei')

FeCrNiMn_models.single_comp_calculation([0.05, 0.31666, 0.31666, 1-0.05-0.31666-0.31666], t_step=4)

FeCrNiMn_models.draw_solidification_curve()
#FeCrNiMn_models.draw_comp_change_in_phase('LIQUID', ['FE', 'NI', 'CR', 'MN'])


