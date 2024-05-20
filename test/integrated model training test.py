from core.ml_models import TrainModels


# This allows automatically integrated model training.
proj_root_path = 'YOUR PROJECT ROOT PATH'
FeNiCrMn_model = TrainModels(proj_root_path)

FeNiCrMn_model.auto_integrate_train(max_evals=10)
FeNiCrMn_model.save_all_models()
