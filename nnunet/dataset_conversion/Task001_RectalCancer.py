#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from nnunet.paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import SimpleITK as sitk


def convert_for_submission(source_dir, target_dir):
    """
    I believe they want .nii, not .nii.gz
    :param source_dir:
    :param target_dir:
    :return:
    """
    files = subfiles(source_dir, suffix=".nii.gz", join=False)
    maybe_mkdir_p(target_dir)
    for f in files:
        img = sitk.ReadImage(join(source_dir, f))
        out_file = join(target_dir, f[:-7] + ".nii")
        sitk.WriteImage(img, out_file)


if __name__ == "__main__":
    base = "/media/margery/4ABB9B07DF30B9DB/DARA-SIRRUNRUN/NEW12Month/RectalCancerDataFull"

    task_id = 1
    task_name = "RectalCancer"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_patient_names = []
    test_patient_names = []
    train_patients = subfiles(join(base, "for3Ddata/source"), join=False)
    for p in train_patients:
        image_file = join(base, "for3Ddata/source", p)
        label_file = join(base, 'for3Ddata/label', p)
        shutil.copy(image_file, join(imagestr, p[:-7] + "_0000.nii.gz"))
        shutil.copy(label_file, join(labelstr, p[:-7] + ".nii.gz"))
        train_patient_names.append(p[:-7])

    test_patients = subfiles(join(base, "for3DdataTest/source"), join=False)
    for p in test_patients:
        image_file = join(base, "for3DdataTest/source", p)
        label_file = join(base, 'for3DdataTest/label', p)
        shutil.copy(image_file, join(imagests, p[-7] + "_0000.nii.gz"))
        shutil.copy(label_file, join(labelsts, p[:-7] + ".nii.gz"))
        test_patient_names.append(p[:-7])


    json_dict = OrderedDict()
    json_dict['name'] = "RectalCancer"
    json_dict['description'] = "RectalCancer"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "sir run run Hospital"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MRI",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "tumor",
        "2": "rectalWall",
    }
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
