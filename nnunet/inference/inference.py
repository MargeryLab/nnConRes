import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import binary
from sklearn.neighbors import KDTree
from scipy import ndimage


def read_nii(path):
    itk_img = sitk.ReadImage(path)
    spacing = np.array(itk_img.GetSpacing())
    return sitk.GetArrayFromImage(itk_img), spacing


def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())


def process_label(label):
    tumor = label == 1
    rectal = label == 2

    return tumor, rectal


'''    
def hd(pred,gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = binary.dc(pred, gt)
        hd95 = binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
'''


def hd(pred, gt):
    # labelPred=sitk.GetImageFromArray(lP.astype(np.float32), isVector=False)
    # labelTrue=sitk.GetImageFromArray(lT.astype(np.float32), isVector=False)
    # hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    # hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    # return hausdorffcomputer.GetAverageHausdorffDistance()
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = binary.hd95(pred, gt)
        print(hd95)
        return hd95
    else:
        return 0


def test(fold):
    path = '../../nnUNet_raw_data_base/nnUNet_raw_data/Task001_RectalCancer'
    label_list = sorted(glob.glob(os.path.join(path, 'labelsTs', '*nii.gz')))
    infer_list = sorted(glob.glob(os.path.join(path, 'inferTs', fold, '*nii.gz')))
    print("loading success...")
    print(len(label_list))
    print(len(infer_list))
    Dice_tumor = []
    Dice_rectal = []

    file = path +'/'+ 'inferTs/' + fold
    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(os.path.join(path, 'inferTs') + '/dice.txt', 'w')

    for label_path, infer_path in zip(label_list, infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label, spacing = read_nii(label_path)
        infer, spacing = read_nii(infer_path)
        label_tumor, label_rectal = process_label(label)
        infer_tumor, infer_rectal = process_label(infer)

        Dice_tumor.append(dice(infer_tumor, label_tumor))
        Dice_rectal.append(dice(infer_rectal, label_rectal))

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('Dice_rv: {:.4f}\n'.format(Dice_tumor[-1]))
        fw.write('Dice_rectal: {:.4f}\n'.format(Dice_rectal[-1]))
        fw.write('*' * 20 + '\n')

    fw.write('*' * 20 + '\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_rv' + str(np.mean(Dice_tumor)) + '\n')
    fw.write('Dice_rectal' + str(np.mean(Dice_rectal)) + '\n')
    fw.write('*' * 20 + '\n')

    dsc = []
    dsc.append(np.mean(Dice_tumor))
    dsc.append(np.mean(Dice_rectal))

    fw.write('DSC:' + str(np.mean(dsc)) + '\n')
    print('done')


if __name__ == '__main__':
    fold = 'output'
    test(fold)
