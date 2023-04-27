import os
from IPython import embed
import torchio as tio
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import csv
# from preprocess import get_mhd_files
# import pyvista as pv
import torch
import random
import torch.nn.functional as F 
import torch.nn as nn
from time import time




def name2coord(mhd_name, root_dir='/public_bme/data/xiongjl/det'):
    # * 输入name，输出这个name所对应着的gt坐标信息
    xyz = []
    whd = []
    # csv_file_dir = 'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord.csv'
    csv_file_dir = os.path.join(root_dir, 'csv_file', 'AT_afterlungcrop.csv')
    with open(csv_file_dir, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            
            if row[0] == mhd_name:
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
                # radius = float(row[5])
                # result.append((x, y, z, radius))
                w = float(row[4])
                h = float(row[5])
                d = float(row[6])
                xyz.append((x, y, z))
                whd.append((w, h, d))
    # print(f'xyz : {xyz}, whd : {whd}  in name2coord func')
    return xyz, whd





def resize_data(name, root_dir, new_shape=(512, 512, 256)):
    # time_1 = time()
    path = '/public_bme/data/xiongjl//det//nii_data_resample_seg_crop//{}_croplung.nii.gz'.format(name)
    origin_coords, origin_whd = name2coord(name)
    image = tio.ScalarImage(path)
    affine = np.eye(4)
    shape = image.shape[1:]
    # affine = image.affine
    scale = np.array(new_shape) / np.array(image.shape[1:])
    new_coords = []
    new_whd = []
    for i in range(len(origin_coords)):
        new_coords.append(origin_coords[i] * scale)
        # new_whd.append((coord[-1], coord[-1], coord[-1]) * scale)
        new_whd.append(origin_whd[i] * scale)

    # create the 1.mask and 2.whd and 3.offset and the 4.image
    mask = create_mask(new_coords, new_shape) # 0.0s no save is so fast
    whd = create_whd(coordinates=new_coords, whd=new_whd, shape=new_shape)
    offset = create_offset(coordinates=new_coords, shape=new_shape)
    # npy2nii(name, mask, suffix='resize_mask', resample=True, affine=affine)
    # npy2nii(name, whd, suffix='resize_whd', resample=True, affine=affine)
    # npy2nii(name, offset, suffix='resize_offset', resample=True, affine=affine)
    
    # hmap_dir = 'D:\Work_file\det\\npy_data\\{}_hmap.npy'.format(name)
    hmap_dir = os.path.join(root_dir, 'npy_data', '{}_hmap.npy'.format(name))
    # hmap_dir = '/public_bme/data/xiongjl/npy_data/{}_hmap.npy'.format(name)
    if os.path.isfile(hmap_dir):
        hmap = np.load(hmap_dir)
    else:
        hmap = create_hmap(coordinates=new_coords, shape=new_shape, save=True, hmap_dir=hmap_dir)
        # hmap = resize_and_normalize(hmap, new_size=(tuple(np.array(new_shape) // 4)))
        # hmap = generate_heatmap_ing(coordinates=new_coords, shape=new_shape, whd=new_whd, reduce=4, save=None, hmap_dir=hmap_dir)

    # npy2nii(name, hmap, suffix='resize_hmap', resample=True, affine=affine)
    # input_data = seg_3d_name(name, root_dir)
    # input_data = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0).float()
    input_data = image.data.unsqueeze(0).float()
    input_resize = F.interpolate(input_data, size=new_shape).squeeze(0).squeeze(0).numpy()
    input_resize = (input_resize - input_data.numpy().min()) / (input_data.numpy().max() - input_data.numpy().min() + 1e-8)
    # npy2nii(name, input_resize, suffix='resize_image', resample=True, affine=affine)
    
    dict = {}
    dict['hmap'] = hmap
    dict['offset'] = offset
    dict['mask'] = mask
    dict['input'] = input_resize
    dict['new_whd'] = new_whd
    dict['new_coords'] = new_coords
    dict['name'] = name
    dict['scale'] = scale
    dict['origin_whd'] = origin_whd
    dict['origin_coords'] = origin_coords
    dict['whd'] = whd
    # print(f'whd : {origin_whd}')
    # embed()
    return dict

def resize_and_normalize(input_array, new_size):

    tensor = torch.tensor(input_array)
    
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif len(tensor.shape) == 4:
        tensor = tensor.unsqueeze(0)
    else:
        print('the numpy dim != 3 or 4')
    
    resized_tensor = F.interpolate(tensor, size=new_size, mode='trilinear', align_corners=True)
    resized_tensor = resized_tensor.squeeze(0).squeeze(0)
    normalized_tensor = (resized_tensor - resized_tensor.min()) / (resized_tensor.max() - resized_tensor.min() + 1e-8)
    return normalized_tensor.numpy()



def create_mask(coordinates, shape, reduce=4, save=False, name=''):
    
    arr = np.zeros(tuple(np.array(shape) // reduce)) 
    for coord in coordinates:
        x, y, z = coord
        x = x / reduce
        y = y / reduce
        z = z / reduce 
        arr[int(x)][int(y)][int(z)] = 1
    if save:
        np.save('/public_bme/data/xiongjl/det//npy_data//{}_mask.npy'.format(name), arr)
    
    return arr

def create_whd(coordinates, whd, shape, reduce=4, save=False):
    
    arr = np.zeros(tuple(np.insert(np.array(shape) // reduce, 0, 3)))
    for i in range(len(coordinates)):
        x, y, z = coordinates[i]
        x = x / reduce
        y = y / reduce
        z = z / reduce 
        arr[0][int(x)][int(y)][int(z)] = whd[i][0]
        arr[1][int(x)][int(y)][int(z)] = whd[i][1]
        arr[2][int(x)][int(y)][int(z)] = whd[i][2]
    if save:
        np.save('array.npy', arr)
    
    return arr


def create_offset(coordinates, shape, reduce=4, save=False):
    arr = np.zeros(tuple(np.insert(np.array(shape) // reduce, 0, 3)))
    for coord in coordinates:
        x, y, z = coord
        x = x / reduce
        y = y / reduce
        z = z / reduce 
        arr[0][int(x)][int(y)][int(z)] = x - int(x)
        arr[1][int(x)][int(y)][int(z)] = y - int(y)
        arr[2][int(x)][int(y)][int(z)] = z - int(z)
    if save:
        np.save('array.npy', arr)
    return arr


# * load time is 0.09s
def create_hmap(coordinates, shape, reduce=4, save=None, hmap_dir=''): # 1.37s, if save :4.33s
    arr = np.zeros(tuple(np.array(shape) // reduce))
    for coord in coordinates:
        x, y, z = coord
        arr[int(x / reduce)][int(y / reduce)][int(z / reduce)] = 1
    # time_si = time()
    arr = gaussian_filter(arr, sigma=3)
    # print('time of si is {}'.format(time() - time_si))
    if arr.max() == arr.min():
        if save != None:
            np.save(hmap_dir, arr)
        return arr
    else:
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        if save != None:
            np.save(hmap_dir, arr)
        return arr






def npy2nii(name, image_npy, root_dir='/public_bme/data/xiongjl/det', suffix='', resample=None, affine=''):
    # csv_dir = 'D:\\Work_file\\det_LUNA16_data\\annotations_athcoord.csv'
    csv_dir = os.path.join(root_dir, 'annotations_pathcoord.csv')
    df = pd.read_csv(csv_dir)
    df = df[df['seriesuid'] == name]
    
    mhd_path = str(df[['path']].values[0])[2:-2]
    image = tio.ScalarImage(mhd_path)
    if resample != None:
        if affine == '':
            print("affine isn't be given")
    else:
        affine = image.affine
    
    if isinstance(image_npy, np.ndarray):
        image_npy = torch.from_numpy(image_npy)
    if len(image_npy.shape) == 3:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy.unsqueeze(0), affine=affine)
        image_nii.save('./nii_temp/{}_{}.nii'.format(name, suffix))
    elif len(image_npy.shape) == 4:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy, affine=affine)
        image_nii.save('./nii_temp/{}_{}.nii'.format(name, suffix))
    else: 
        print('DIM ERROR : npy.dim != 3 or 4')

    # image_nii.save('./nii_temp/{}_image.nii'.format(name))
    # return print('save done')






if __name__ == '__main__':
    name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.129567032250534530765928856531'
    dict = resize_data(name, '')
