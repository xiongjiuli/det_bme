import csv
from utils_v2 import resize_data
from model.resnet import CenterNet
import torch
import numpy as np
import torch
from torch import nn
from torchvision.ops import nms
import torchio as tio
from IPython import embed
import csv
from collections import defaultdict



def npy2nii(name, image_npy, suffix=''):
    image_npy = image_npy.cpu()
    affine = np.array([[0.7, 0, 0, 0], [0, 0.7, 0, 0], [0, 0, 1.2, 0], [0, 0, 0, 1]])
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
    elif len(image_npy.shape) == 5:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy[0, :, :, :,:], affine=affine)
        image_nii.save('./nii_temp/{}_{}.nii'.format(name, suffix))
    else: 
        print('DIM ERROR : npy.dim != 3 or 4 or 5')

# def read_and_convert_data(names):
#     file_path = '/public_bme/data/xiongjl/det/AT_afterlungcrop.csv'
#     result = defaultdict(list)
#     with open(file_path, 'r') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             name = row['seriesuid']
#             if name in names:
#                 x = float(row['x'])
#                 y = float(row['y'])
#                 z = float(row['z'])
#                 w = float(row['w'])
#                 h = float(row['h'])
#                 d = float(row['d'])
                
#                 x1 = x - w / 2
#                 y1 = y - h / 2
#                 z1 = z - d / 2
#                 x2 = x + w / 2
#                 y2 = y + h / 2
#                 z2 = z + d / 2
                
#                 result[name].append((x1, y1, z1, x2, y2, z2))
#     return result


def centerwhd_2nodes(centers, dimensions_list):
    result = []
    for center, dimensions in zip(centers, dimensions_list):

        x, y, z = center
        # embed()
        length, width, height = dimensions
        x1 = x - length/2
        y1 = y - width/2
        z1 = z - height/2
        x2 = x + length/2
        y2 = y + width/2
        z2 = z + height/2
        result.append([x1, y1, z1, x2, y2, z2])
    return result



def read_names_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过第一行
        names = [row[0] for row in reader]
    return list(names)

def calculate_iou(box1, box2):
    # 计算两个框的交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    z1 = max(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    y2 = min(box1[4], box2[4])
    z2 = min(box1[5], box2[5])
    intersection = max(0, x2 - x1) * max(0, y2 - y1) * max(0, z2 - z1)
    
    # 计算两个框的并集
    box1_volume = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    box2_volume = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    union = box1_volume + box2_volume - intersection
    
    # 计算IoU
    iou = intersection / union
    return iou


def calculate_accuracy_and_recall(predicted_boxes, ground_truth_boxes, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # for image_index in range(len(predicted_boxes)):
        # in a image 
    image_predicted_boxes = predicted_boxes#[image_index]
    image_ground_truth_boxes = ground_truth_boxes#[image_index]
    matched_ground_truth_boxes = [False] * len(image_ground_truth_boxes)
    
    for predicted_box in image_predicted_boxes:
        max_iou = 0
        max_iou_index = -1
        for ground_truth_index, ground_truth_box in enumerate(image_ground_truth_boxes):
            # embed()
            iou = calculate_iou(predicted_box, ground_truth_box)
            if iou > max_iou:
                max_iou = iou
                max_iou_index = ground_truth_index
        
        if max_iou >= iou_threshold:
            if not matched_ground_truth_boxes[max_iou_index]:
                true_positives += 1
                matched_ground_truth_boxes[max_iou_index] = True
            else:
                false_positives += 1
        else:
            false_positives += 1
    
    false_negatives += len(image_ground_truth_boxes) - sum(matched_ground_truth_boxes)

    if true_positives == 0:
        accuracy = 0
        recall = 0
    else:
        accuracy = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
    
    return accuracy, recall



def pool_nms(heat, kernel = 3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool3d(heat, (kernel, kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep



def decode_bbox(pred_hms, pred_whds, pred_offsets, scale, confidence, reduce, cuda):

    pred_hms = pool_nms(pred_hms)

    heat_map    = pred_hms[0, :, :, :].cpu()
    pred_whd     = pred_whds[0, :, :, :].cpu()
    pred_offset = pred_offsets[0, :, :, :].cpu()

    mask = torch.from_numpy(np.where(heat_map > confidence, 1, 0)).squeeze(1).bool()
    # mask[0, 50, 60, 30] = 1
    indices = np.argwhere(mask == 1)
    # indices += 1
    # print(indices)
    # embed()
    
    centers = []
    whds = []
    for i in range(indices.shape[1]):
        coord = indices[1 :, i]
        x = coord[0].cpu()
        y = coord[1].cpu()
        z = coord[2].cpu()
        
        offset_x = pred_offset[0, x, y, z]
        offset_y = pred_offset[1, x, y, z]
        offset_z = pred_offset[2, x, y, z]

        w = pred_whd[0, x, y, z] / scale[0]
        h = pred_whd[1, x, y, z] / scale[1]
        d = pred_whd[2, x, y, z] / scale[2]

        center = ((x + offset_x) * reduce, (y + offset_y) * reduce, (z + offset_z) * reduce) / scale
        origin_whd = (w, h, d) 
        # print(f'center: {center}, origin_whd : {origin_whd}')

        centers.append(center)
        whds.append(origin_whd)
    predicted_boxes = centerwhd_2nodes(centers, dimensions_list=whds)
    # print(f'predicted_boxes is : {predicted_boxes}')

    return predicted_boxes




def test_luna16(name, model_path):

    # file_path = ''
    root_dir = '/public_bme/data/xiongjl/det'
    # names = read_names_from_csv(file_path)
    test_batch = resize_data(name, root_dir, new_shape=(512, 512, 256))

    model = CenterNet('resnet50', 1)
    # model = CenterNet(config.backbone_name, 3)
    model.cuda()

    model.load_state_dict(torch.load(model_path)['model'])

    with torch.no_grad():
        # 清理缓存
        torch.cuda.empty_cache()
        image = test_batch['input']
        image = torch.from_numpy(image)
        image = image.unsqueeze(0).unsqueeze(0).cuda()
        name = test_batch['name']
        coords = test_batch['origin_coords']
        whd = test_batch['origin_whd']
        scale = test_batch['scale']
        # embed()
        pred_hmap, pred_whd, pred_offset = model(image)
        npy2nii(name=name[-4 :], image_npy=pred_hmap, suffix='pred_hmap')
        npy2nii(name=name[-4 :], image_npy=image, suffix='input_image')

        predicted_boxes = decode_bbox(pred_hmap, pred_whd, pred_offset, scale, confidence=0.7, reduce=4, cuda=True)
        # ground_truth_boxes = read_and_convert_data(name)
        # embed()
        ground_truth_boxes = centerwhd_2nodes(centers=coords, dimensions_list=whd)
        # embed()
        accuracy, recall = calculate_accuracy_and_recall(predicted_boxes, ground_truth_boxes, iou_threshold=0.3)
        
        print(accuracy, recall)
        print(name[-4:])
        
        print(predicted_boxes)
        print(ground_truth_boxes)
        print('==================================')
        # logger.info('Epoch: %d, accuracy: %d, recall: %.4f', step, accuracy, recall)/
        # embed()




if __name__ == '__main__':

    model_path = '/public_bme/data/xiongjl/det/save/best_model-240.pt'
    file_path = '/public_bme/data/xiongjl/det/csv_file/test_names.csv'
    names = read_names_from_csv(file_path)
    # embed()
    for name in names:
        # embed()
        test_luna16(name, model_path)

