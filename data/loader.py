"""
Central entrypoint for dataset loading.
Supports loading as pixel, line, or cube.
"""

from typing import Tuple


import numpy as np
from .preprocessing import preprocess_spectra
import os
from pycocotools.coco import COCO
from scipy.signal import savgol_filter
import scipy
from scipy.signal import medfilt
import cv2 as cv2

def load_hsi_numpy(path):
    loaded = np.load(path+'.npy')[...,5:247]
    
    return loaded

def load_calibrations(dark_path,white_path):
    return load_hsi_numpy(dark_path),load_hsi_numpy(white_path)
def get_class_vector(class_list,attr):
    label_vector = []
    for c in class_list:
        if(attr[c]>0):
            label_vector.append(1)
        else:
            label_vector.append(0)
    return label_vector
def get_materials_present(attr):
    materials_present = []
    for c in attr:
        if(attr[c]>0):
            materials_present.append(c)
    return materials_present
def get_split_file_list(split_list_path,):
    split_list = []
    with open(split_list_path) as f:
        split_list = f.read().splitlines()

    return split_list
def get_split_dict(coco,split_list):
    split_dict = {}
    for img_id in coco.getImgIds():
            img_info = coco.loadImgs(img_id)[0]
            fname = img_info["file_name"].split('_')[0]
            if fname in split_list:
                split_dict[img_info["id"]] = fname

    return split_dict
def stream_dataset(cfg: dict, mode: str = "pixel",split='train'):
    split = cfg[split+'_split_file']

    #read from config.yaml the parameters
    dataset_base = cfg['root_dir']
    coco=COCO(os.path.join(dataset_base,'labels/smartex_annotations_cocostyle.json'))
    splitlist = get_split_file_list(os.path.join(dataset_base,split))
    
    split_dict = get_split_dict(coco,splitlist)
    dark,white = load_calibrations(os.path.join(dataset_base,'calib_mean','black'),os.path.join(dataset_base,'calib_mean','white'))
    if(mode=='pixel'):
        return stream_pixels(cfg['pixel_level'],coco,split_dict,dark,white)
    elif(mode=='line'):
        if(split=='test'):
            return stream_lines(cfg['line_level'],coco,split_dict,dark,white,line_stride=20)
        else:
            return stream_lines(cfg['line_level'],coco,split_dict,dark,white)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
def load_dataset(cfg: dict, mode: str = "pixel",split='train') -> Tuple:
    """
    Load hyperspectral data according to the selected mode.
    mode: 'pixel' | 'line' | 'cube'
    Returns: X_train, y_train, X_test, y_test, meta_test
    """

    split = cfg[split+'_split_file']

    #read from config.yaml the parameters
    dataset_base = cfg['root_dir']
    coco=COCO(os.path.join(dataset_base,'labels/smartex_annotations_cocostyle.json'))
    splitlist = get_split_file_list(os.path.join(dataset_base,split))
    
    split_dict = get_split_dict(coco,splitlist)
    dark,white = load_calibrations(os.path.join(dataset_base,'calib_mean','black'),os.path.join(dataset_base,'calib_mean','white'))
    if(mode=='pixel'):
        return load_dict_pixels(cfg['pixel_level'],coco,split_dict,dark,white)
    elif(mode=='line'):
        return load_dict_lines(cfg['line_level'],coco,split_dict,dark,white)
    elif(mode=='cube'):
        return load_dict_cubes(cfg,coco,split_dict,dark,white)
    else:
        raise ValueError(f"Unknown mode: {mode}")
def load_dict_cubes(cfg,coco,split_dict,dark,white) -> Tuple:
    raise NotImplementedError("Cube loading not implemented yet.")
def load_dict_lines(cfg,coco,split_dict,dark,white) -> Tuple:
    class_list = cfg['class_list']
    class_to_idx = {cls: i for i, cls in enumerate(class_list)}
    idx_to_class = {i: cls for i, cls in enumerate(class_list)}
    dataset_base = cfg['root_dir']
    
    X_all, y_all = [], []
    print('Loading dataset...')
    
    print('class list:',class_list)
    for train_id in split_dict.keys():
        print('current file:',train_id)
        ann_ids = coco.getAnnIds(imgIds=[train_id])
        anns = coco.loadAnns(ann_ids)
        ann = [ann for ann in anns if ann['category_id']==2][0]
        #print(ann)
        ann_vector = get_class_vector(class_list,ann['attributes'])
        
        materials_present = get_materials_present(ann['attributes'])
        
        if cfg['filter_mode']=='pure_only':
            if(np.array(ann_vector).sum()>1):
                
                continue
        elif cfg['filter_mode']=='mixed_only':
            if(np.array(ann_vector).sum()<=1):
                
                continue
        elif cfg['filter_mode']=='all':
            pass
        
        #check if all materials present are in class_list
        if not all(item in class_list for item in materials_present):
            
            continue
        file_name = split_dict[train_id]
        mask = cv2.imread(os.path.join(dataset_base,'labels/masks','textile',f"{file_name}.png"),0)
        #print(mask.shape)
        hsi_data = load_hsi_numpy(os.path.join(dataset_base,'data/hsi',f"{file_name}"))
       
        hsi_data = preprocess_spectra(cfg,hsi_data,dark,white)
        for line in range(0,hsi_data.shape[1],10):
            
            if int(np.sum(mask[:,line])/255)<50:
                continue
            
            count = 0
           
            valid_idx = np.where(mask[:, line]>0)[0]
           
                
            # Randomly shuffle and take first N pixels
            np.random.shuffle(valid_idx)
            sel_idx = valid_idx[: min(cfg.get("n_samples_per_class",50), valid_idx.size)]
            X_line = hsi_data[sel_idx, line, :]  # (N, B)
            
            #maintain the shape (N, B)
            X_all.append(X_line)
            y_all.append(ann_vector )
            
        
        
       
        
        
       
        
    X_all = np.array(X_all)
    y_all = np.vstack(y_all)
    print('Final dataset shape:',X_all.shape,y_all.shape)
    
    return X_all,y_all
def load_dict_pixels(cfg,coco,split_dict,dark,white) -> Tuple:
    class_list = cfg['class_list']
    class_to_idx = {cls: i for i, cls in enumerate(class_list)}
    idx_to_class = {i: cls for i, cls in enumerate(class_list)}
    dataset_base = cfg['root_dir']
    
    X_all, y_all = [], []
    print('Loading dataset...')
    
    print('class list:',class_list)
    for train_id in split_dict.keys():
        ann_ids = coco.getAnnIds(imgIds=[train_id])
        anns = coco.loadAnns(ann_ids)
        ann = [ann for ann in anns if ann['category_id']==2][0]
        #print(ann)
        ann_vector = get_class_vector(class_list,ann['attributes'])
        
        materials_present = get_materials_present(ann['attributes'])
        if('elastan' in materials_present and 'polyester' in materials_present):
            print("found polyester and elastan together, skipping   ")
            continue
        if cfg['filter_mode']=='pure_only':
            if(np.array(ann_vector).sum()>1):
                
                continue
        
        #check if all materials present are in class_list
        if not all(item in class_list for item in materials_present):
            
            continue
    
        print(ann['attributes'])
        file_name = split_dict[train_id]
        mask = cv2.imread(os.path.join(dataset_base,'labels/masks','textile',f"{file_name}.png"),0)
        #print(mask.shape)
        hsi_data = load_hsi_numpy(os.path.join(dataset_base,'data/hsi',f"{file_name}"))
        
        
        hsi_data = preprocess_spectra(cfg,hsi_data,dark,white)
        valid_idx = mask > 0
        masked = hsi_data[valid_idx]
        #np.random.shuffle(masked)
        sampled = masked[: cfg.get("n_samples_per_class", 200)]
        
        print(sampled.shape)
        X_all.extend(sampled)
        y_all.append( np.tile(ann_vector, (sampled.shape[0], 1)) )
        print("# of samples so far",len(y_all))
    X_all = np.array(X_all)
    y_all = np.vstack(y_all)
    print('Final dataset shape:',X_all.shape,y_all.shape)
    
    return X_all,y_all

def calc_means(x,group_size=5, groups_to_keep=10):
    """
    Calculate means for normalization.
    """
    n = x.shape[0]
    p = x.shape[1]
    x_shuffled = x[np.random.permutation(n)]
    num_groups = n //group_size
    x_trimmed = x_shuffled[:num_groups * group_size]
    x_groups = x_trimmed.reshape(num_groups, group_size, p)
    
    group_means = x_groups.mean(axis=1)
    #print(x_groups.shape,group_means.shape)
    result = group_means[:groups_to_keep]
    return result

    
def stream_lines(cfg,coco,split_dict,dark,white,line_stride=10):
    class_list = cfg['class_list']
    class_to_idx = {cls: i for i, cls in enumerate(class_list)}
    idx_to_class = {i: cls for i, cls in enumerate(class_list)}
    dataset_base = cfg['root_dir']
    
    X_all, y_all = [], []
    print('Loading dataset...')
    
    print('class list:',class_list)
    file_count = 0
    for train_id in split_dict.keys():
        ann_ids = coco.getAnnIds(imgIds=[train_id])
        anns = coco.loadAnns(ann_ids)
        ann = [ann for ann in anns if ann['category_id']==2][0]
        #print(ann)
        ann_vector = get_class_vector(class_list,ann['attributes'])
        
        materials_present = get_materials_present(ann['attributes'])
        if('elastan' in materials_present and 'polyester' in materials_present):
            print("found polyester and elastan together, skipping   ")
            continue
        if cfg['filter_mode']=='pure_only':
            if(np.array(ann_vector).sum()>1):
                
                continue
        elif cfg['filter_mode']=='mixed_only':
            if(np.array(ann_vector).sum()<=1):
                
                continue
        elif cfg['filter_mode']=='all':
            pass
        
        #check if all materials present are in class_list
        if not all(item in class_list for item in materials_present):
            
            continue
        print(ann['attributes'])
        res_dict = {}
        print('current file:',split_dict[train_id])
        file_count += 1
        #copy percentages from ann['attributes'] to res_dict only for non-zero classes
        res_dict = {cls: ann['attributes'][cls] for cls in class_list if ann['attributes'][cls]>0}
        file_name = split_dict[train_id]
        mask = cv2.imread(os.path.join(dataset_base,'labels/masks','textile',f"{file_name}.png"),0)
        #print(mask.shape)
        hsi_data = load_hsi_numpy(os.path.join(dataset_base,'data/hsi',f"{file_name}"))
       
        hsi_data = preprocess_spectra(cfg,hsi_data,dark,white)
        
        for line in range(0,hsi_data.shape[1],line_stride):
            
            if int(np.sum(mask[:,line])/255)<50:
                continue
            
            count = 0
            
            valid_idx = np.where(mask[:, line]>0)[0]
           
            valid_line = hsi_data[valid_idx, line, :]

            # Randomly shuffle and take first N pixels
            #np.random.shuffle(valid_idx)
            sel_idx = valid_idx[: min(cfg.get("n_samples_per_class",50), valid_idx.size)]
            X_line = hsi_data[sel_idx, line, :]  # (N, B)
            #X_line = calc_means(valid_line,group_size=5, groups_to_keep=cfg.get("n_samples_per_class",50))
            if(X_line.shape[0]<24):
                continue
            #maintain the shape (N, B)
            yield X_line, ann_vector,file_name,res_dict
            #X_all.append(X_line)
            #y_all.append(ann_vector )
    print('total files processed:',file_count)
            
def stream_pixels(cfg,coco,split_dict,dark,white):
    class_list = cfg['class_list']
    class_to_idx = {cls: i for i, cls in enumerate(class_list)}
    idx_to_class = {i: cls for i, cls in enumerate(class_list)}
    dataset_base = cfg['root_dir']
    
    X_all, y_all = [], []
    print('Loading dataset...')
    
    print('class list:',class_list)
    file_count = 0
    for train_id in split_dict.keys():
        
        ann_ids = coco.getAnnIds(imgIds=[train_id])
        anns = coco.loadAnns(ann_ids)
        ann = [ann for ann in anns if ann['category_id']==2][0]
        #print(ann)
        ann_vector = get_class_vector(class_list,ann['attributes'])
        
        materials_present = get_materials_present(ann['attributes'])
        if('elastan' in materials_present and 'polyester' in materials_present):
            print("found polyester and elastan together, skipping   ")
            continue
        if cfg['filter_mode']=='pure_only':
            if(np.array(ann_vector).sum()>1):
                
                continue
        
        #check if all materials present are in class_list
        if not all(item in class_list for item in materials_present):
            
            continue
        print('current file:',split_dict[train_id])
        file_count += 1
        print(ann['attributes'])
        file_name = split_dict[train_id]
        mask = cv2.imread(os.path.join(dataset_base,'labels/masks','textile',f"{file_name}.png"),0)
        #print(mask.shape)
        hsi_data = load_hsi_numpy(os.path.join(dataset_base,'data/hsi',f"{file_name}"))
        
        
        hsi_data = preprocess_spectra(cfg,hsi_data,dark,white)
        valid_idx = mask > 0
        masked = hsi_data[valid_idx]
        #np.random.shuffle(masked)
        sampled = masked[: cfg.get("n_samples_per_class", 500)]
        #print(sampled.shape)
        #sampled = calc_means(masked,groups_to_keep=cfg.get("n_samples_per_class", 200))
        
        yield sampled, np.tile(ann_vector, (sampled.shape[0], 1))
        #X_all.extend(sampled)
        #y_all.append( np.tile(ann_vector, (sampled.shape[0], 1)) )
    
    #X_all = np.array(X_all)
    #y_all = np.vstack(y_all)
    #print('Final dataset shape:',X_all.shape,y_all.shape)
    print('total files processed:',file_count)
    #return X_all,y_all

