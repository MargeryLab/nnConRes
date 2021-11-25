import shutil, os

raw_path = '/home/dell/WinDisk/nnConRes/nnUNet_raw_data_base/nnUNet_raw_data/Task001_RectalCancer/labelsTs'
Data_path = '/home/dell/WinDisk/Data/nnUNet_raw_data_base/nnUNet_raw_data/Task001_RectalCancer/labelsTs'

des_data_ls = os.listdir(raw_path)
for ID in os.listdir(raw_path):
    try:
        shutil.copy(os.path.join(Data_path, ID), raw_path)
    except:
        shutil.copy(os.path.join('/home/dell/WinDisk/Data/nnUNet_raw_data_base/nnUNet_raw_data/Task001_RectalCancer/labelsTr', ID), raw_path)
