import numpy as np
import os
import SimpleITK as sitk
import torch
from modeling.SASR_Unet_3D import *
import pandas as pd
from utils.metrics import FWIou
from sklearn.metrics import accuracy_score
from medpy import metric
from scipy.ndimage import zoom as sci_zoom


os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def load_dcm_image(case_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(case_path)
    reader.SetFileNames(dicom_names)
    sitk_img = reader.Execute()
    img = sitk.GetArrayFromImage(sitk_img)
    origin = sitk_img.GetOrigin()
    spacing = sitk_img.GetSpacing()
    direction = sitk_img.GetDirection()
    img = normalize(img)
    return img.astype('float32'), np.asarray(origin), np.asarray(spacing), direction, sitk_img

def zoom(img, new_shape, new_spacing, old_spacing, origin, direction, interpolation):
    sitk_img = sitk.GetImageFromArray(img)
    sitk_img.SetOrigin(origin)
    sitk_img.SetSpacing(old_spacing)
    sitk_img.SetDirection(direction)

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolation)
    resample.SetOutputDirection(sitk_img.GetDirection())
    resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_shape)
    newimage = resample.Execute(sitk_img)
    return sitk.GetArrayFromImage(newimage)

def normalize(npzarray):
    '''
    preprocessing
    '''
    maxHU = 400.0
    minHU = -1000.     #origin HU is -1200~0
    MEAN_VALUE = 0 # for luna is 0.33
    STD = 1
    npzarray[npzarray>maxHU] = maxHU
    npzarray[npzarray<minHU] = minHU
    npzarray_normalized = (npzarray-minHU) / (maxHU-minHU)
#    npzarray_normalized = (npzarray_normalized - MEAN_VALUE) / STD
    return npzarray_normalized


def dice_coef(input, target):
    # N = target.size(0)

    input_flat = input
    target_flat = target

    intersection = input_flat * target_flat

    if np.sum(input_flat)==0 and np.sum(target_flat)==0:
        return 'error'
    dice = 2 * (np.sum(intersection) / (np.sum(input_flat) + np.sum(target_flat)))

    return dice

def vtow_coord(voxelCoord, origin, spacing):
    '''
    To transform the voxel coordinates to world coordinates
    '''
    worldCoord = np.multiply(voxelCoord, spacing) + origin
    worldCoord = np.around(worldCoord).astype(int)
    return worldCoord.tolist()

def accuracy(y_true, y_pred):
    input_flat = y_pred.flatten()
    target_flat = y_true.flatten()
    acc = accuracy_score(target_flat, input_flat)
    return acc

if __name__=='__main__':

    test_csv = '/home2/LUNG_DATA/LIDC-IDRI/test_nodules_new.csv'
    data_path = '/home2/LUNG_DATA/npy_48_LIDC/image/LIDC-IDRI/2018_07_21/'
    label_path = '/home2/LUNG_DATA/npy_48_LIDC/label/LIDC-IDRI/2018_07_21/'

    nodule_list = np.array(pd.read_csv(test_csv))


    model = SASR_Unet(in_channels=1, out_channels=1, final_sigmoid=True, sr=True)

    model = nn.DataParallel(model)
    model.cuda()

    checkpoint = torch.load('./run/nodule/Unet--SR_48_1025_unetsr_SAM_sigmoid_LIDC/experiment_3/checkpoint_0.34856232615598176.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    count = 0
    dice_all = []
    dice_4 = []
    dice_above_4 = []

    iou_all = []
    iou_4 = []
    iou_above_4 = []
    FWIOU_all = []
    FWIOU_4 = []
    FWIOU_above_4 = []
    mae_all = []
    mae_4 = []
    mae_above_4 = []
    dice_detect=0
    count_detect=0

    seg_path = '/home2/ywli/DSRL/nodule_seg_result/unet/'
    result = []
    for idx in range(nodule_list.shape[0]):#
        nodule = nodule_list[idx, :]
        name = nodule[1]
        coordx = int(nodule[2])

        coordy = int(nodule[3])
        coordz = int(nodule[4])
        dia = nodule[5]

        img_patch = np.load(
            data_path + name + '_' + str(coordx) + '_' + str(coordy) + '_' + str(
                coordz) + '.npy')
        mask_patch = np.load(
            label_path + name + '_' + str(coordx) + '_' + str(coordy) + '_' + str(
                coordz) + '.npy')

        patch_img_arr = np.zeros([1, 1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]], dtype='float32')
        patch_img_arr[0,0,:,:,:] = img_patch
        patch_img_arr = torch.from_numpy(patch_img_arr.astype(np.float32)).cuda()
        ###############################################################resize predict ##################################################
        input_patch_img_arr = patch_img_arr

        with torch.no_grad():
            patch_predict, output_sr, fea_seg, fea_sr,offset_seg, offset_sr  = model(input_patch_img_arr)

        patch_pred = patch_predict.cpu().numpy()

        patch_predict = (patch_pred[0, 0, :, :, :] > 0.5).astype(np.uint8)

        zoom_seq_org = [1/2, 1/2, 1/2]
        patch_predict_org = sci_zoom(patch_predict, zoom_seq_org, order=0)

        dice_nodule = dice_coef(patch_predict_org, mask_patch)
        iou = metric.jc(patch_predict_org, mask_patch)
        FWIOU = FWIou(mask_patch, patch_predict_org)
        mae = np.mean(abs(patch_predict_org.astype(np.float32) - mask_patch.astype(np.float32)))

        if dice_nodule == 'error':
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        else:
            if dice_nodule > 0:
                dice_detect = dice_detect + dice_nodule
                count_detect += 1
            dice_all.append(dice_nodule)
            iou_all.append(iou)
            FWIOU_all.append(FWIOU)
            mae_all.append(mae)
            count += 1
            if dia < 4:
                dice_4.append(dice_nodule)
                iou_4.append(iou)
                FWIOU_4.append(FWIOU)
                mae_4.append(mae)

            else:
                dice_above_4.append(dice_nodule)
                iou_above_4.append(iou)
                FWIOU_above_4.append(FWIOU)
                mae_above_4.append(mae)

        result.append([name, coordx, coordy, coordz, dia, dice_nodule])

    print('dice_all:', dice_all)
    print('count:', count)

    print('dice <4:', np.mean(np.array(dice_4)))
    print('iou <4:', np.mean(np.array(iou_4)))
    print('acc <4:', np.mean(np.array(FWIOU_4)))
    print('mae <4:', np.mean(np.array(mae_4)))
    print(' ')


    print('dice >4:', np.mean(np.array(dice_above_4)))
    print('iou >4:', np.mean(np.array(iou_above_4)))
    print('acc >4:', np.mean(np.array(FWIOU_above_4)))
    print('mae >4:', np.mean(np.array(mae_above_4)))

    print(' ')

    print('dice avg: ', np.mean(np.array(dice_all)))
    print('iou avg: ', np.mean(np.array(iou_all)))
    print('acc avg: ', np.mean(np.array(FWIOU_all)))
    print('mae avg: ', np.mean(np.array(mae_all)))

