from core.rawdata_process import CreateDataset
from core.ml_models import CompositionModels


proj_root_path = 'YOUR PREFFERED PATH'

FeNiCrMn_dataset = CreateDataset(proj_root_path)
FeNiCrMn_dataset.create_dataset()

comp_model = CompositionModels()
comp_model.cp_prepare_for_training(FeNiCrMn_dataset.discrete_dataset)

# Mode 1
comp_model.cp_train_model(bayesian_opt=True, max_evals=3)

# Mode 2
comp_model.cp_train_model(bayesian_opt=False, epoch=900, first_layer=320, second_layer=440, third_layer=360)

# This is separate model training test. If you want to store the model, you can use the following code.
comp_model.cp_save_model('YOUR PREFFERED PATH')
