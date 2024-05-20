from core.scheil_validate import ScheilValidate


proj_root_path = 'YOUR PROJECT ROOT PATH'
task_name = 'The task that finished calculation'

FeCrNiMn_validate = ScheilValidate(proj_root_path, task_name)

FeCrNiMn_validate.load_validate_data()
FeCrNiMn_validate.validate_all()
