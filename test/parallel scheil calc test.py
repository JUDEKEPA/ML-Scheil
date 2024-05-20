from core.scheil_calc import ScheilCalculation


# 1. Set the project root path and task name
proj_root_path = 'YOUR PROJECT ROOT PATH'
FeCrNiMn_models = ScheilCalculation(proj_root_path, 'task name')

# 2. Set the calculation conditions
FeCrNiMn_models.set_condition(['FE', 'CHANGING', [0, 1, 0.01]], ['Ni', 'CHANGING', [0, 1, 0.01]],
                              ['Cr', 'changing', [0, 1, 0.01]])

FeCrNiMn_models.set_t_step(4)
FeCrNiMn_models.check_condition()

# 3. Run the calculation
FeCrNiMn_models.parallel_calc_scheil()

# 4. Save the calculation results
FeCrNiMn_models.data_storage()

