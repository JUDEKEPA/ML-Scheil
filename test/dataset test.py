from core.rawdata_process import CreateDataset

proj_root_path = 'YOUR PREFFERED PATH'

FeNiCrMn_dataset = CreateDataset(proj_root_path, ['Fe', 'Ni', 'Cr', 'Mn'], [1200, 2200])

# An example data can be downloaded from xxx
FeNiCrMn_dataset.add_xlsx_data(['Data path1', 'Data path2', 'Data path3'])
FeNiCrMn_dataset.complete_add()
