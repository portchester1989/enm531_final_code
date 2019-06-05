from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from collections import namedtuple, defaultdict
import logging
import torch
import math
import shutil
import dill

data_path = os.path.join('data')
train_root_dir = os.path.join(data_path, 'train')
val_root_dir = os.path.join(data_path, 'test')
blacklist = defaultdict(list)
blacklist[os.path.join(train_root_dir, 'fruit_fly_volumes.npz')] = [14, 74]

'''
train_samples = []
for path in list(map(lambda f : os.path.join(train_root_dir, f), os.listdir(train_root_dir))):
    data = np.load(path)
    data_vol = data['volume']
    data_label = data['label']
    d_blacklist = blacklist[path]
    for i in range(data_vol.shape[0]):
        if i not in d_blacklist:
            train_samples.append((data_vol, data_label, i))

val_samples = []
for path in list(map(lambda f : os.path.join(val_root_dir, f), os.listdir(val_root_dir))):
    data = np.load(path)
    data_vol = data['volume']
    data_label = data['label']
    d_blacklist = blacklist[path]
    for i in range(data_vol.shape[0]):
        if i not in d_blacklist:
            val_samples.append((data_vol, data_label, i))
'''

class NeuronDataset(Dataset):
    def __init__(self,mode = 'train',use_deep_cut = True):
        super().__init__()
        '''
        self.transform_img = transform_img
        self.transform_both = transform_both
        self.transform_norm = transform_norm
        self.shape = (preproc.img_size, preproc.img_size)
        self.samples = samples
        '''
        if mode == 'train':
            images = np.load('../../svs_images/training_imgs.npz')
            box_labels = np.load('../../svs_images/box_labels.npz')
            box_coords = np.load('../../svs_images/box_coords.npz')
            if use_deep_cut:
                masks = np.load('../../svs_images/train_masks.npz')
                masks = [mask for mask in masks.values()]
            else:
                masks = []
        else:
            images = np.load('../../svs_images/val_imgs.npz')
            box_labels = np.load('../../svs_images/val_box_labels.npz')
            box_coords = np.load('../../svs_images/val_box_coords.npz')     
            if use_deep_cut:
                masks = np.load('../../svs_images/val_masks.npz')
                masks = [mask for mask in masks.values()]
        fg_images = images['fg_patches']
        bg_images = images['bg_patches']
        labels = [label for label in box_labels.values()]
        
        if not use_deep_cut:    
            for i,coord in enumerate(box_coords.values()):
                label = labels[i] + 1
                mask = np.zeros((256,256))
                for j,box in enumerate(coord):
                    y1,x1 = np.floor(box[:2]).astype(np.uint8)
                    y2,x2 = np.ceil(box[2:]).astype(np.uint8)
                    mask[y1:y2,x1:x2] = label[j]
                masks.append(mask)
        #concatenate fg and bg
        bg_masks = [np.zeros((256,256)) for i in range(len(bg_images))]
        self.images = np.concatenate([fg_images,bg_images],axis = 0)
        self.images = np.transpose(self.images,(0,3,1,2)) / 255
        self.masks = np.stack(masks + bg_masks)
                                 
    def __len__(self):
        return len(self.masks)

    def __getitem__(self, i):
        img = self.images[i]
        label = self.masks[i]
        '''
        index = self.samples[i][2]
        img = self.samples[i][0][index]
        label = self.samples[i][1][index]
        if self.transform_both:
            img_label = np.array(self.transform_both(Image.fromarray(np.stack((img, label), axis=-1))))
            img = Image.fromarray(img_label[:,:,0])
            label = Image.fromarray(img_label[:,:,1])
        if self.transform_img:
            img = self.transform_img(img)
        img_label = self.transform_norm(Image.fromarray(np.stack((img, label), axis=-1)))
        '''
        
        return {'data':img,'label':label}

def get_data(use_deep_cut = True):
    trn_data = NeuronDataset('train',use_deep_cut)
    val_data = NeuronDataset('val',use_deep_cut) 
    #shape is HW or HWC
    #shape = trn_data.shape
    input_channels = 3 
    #assert shape[0] == shape[1], "not expected shape = {}".format(shape)
    

    return [{'input_channels': input_channels, 'num_classes': 3}, 
            trn_data, val_data]

def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('scene')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.

class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def iou(pred, target, n_classes=3):
    ious = []
    #pred = torch.round(pred.view(-1)).long()
    pred = torch.argmax(pred,1).view(-1).long()
    target = torch.round(target.view(-1)).long()

    # Ignore IoU for background class ("0")
    for cls in range(0, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored

        pred_inds = pred == (cls + 1)
        target_inds = target == (cls + 1)
        # Cast to long to prevent overflows       
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - \
                intersection

        if union == 0:
            ious.append(0.)  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.average(ious)

def save_item(item, f_dir, names):
    filename = os.path.join(f_dir, names[0]+'.pth.tar')
    torch.save(item, filename, pickle_module=dill)
    for name in names[1:]:
        other_filename = os.path.join(f_dir, name+'.pth.tar')
        shutil.copyfile(filename, other_filename)

class ExpFinderSchedule(namedtuple('ExpFinderSchedule', ('lr_start', 'lr_end', 'nb_iters_train'))):
    def __call__(self, t):
        r = t / self.nb_iters_train
        return self.lr_start * (self.lr_end / self.lr_start) ** r

class PiecewiseLinearOrCos(namedtuple('PiecewiseLinear', ('knots', 'vals', 'is_cos'))):
    def __call__(self, t):
        if t <= self.knots[0]:
            return float(self.vals[0])
        if t >= self.knots[-1]:
            return float(self.vals[-1])
        for k in range(len(self.knots)):
            if t < self.knots[k]:
                break

        if self.is_cos[k-1]:
            a = self.vals[k-1]
            b = self.vals[k]
            c = self.knots[k-1]
            d = self.knots[k]

            assert abs(c-d) > 1e-200
            
            return math.cos((t-c)*math.pi/(c-d))*(a-b)/2.+(b-a)/2+a
        else:    
            return np.interp([t], self.knots, self.vals)[0]
