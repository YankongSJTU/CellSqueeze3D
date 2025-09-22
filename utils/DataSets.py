import torch
import glob
import pickle
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import torchvision.models as models
from torch.utils.data.dataset import Dataset 
from PIL import ImageFilter
import random 
import os
import torch.utils.data
def is_typical_nuclei_image(image, white_threshold=160, max_white_ratio=0.5):
   
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be RGB (H, W, 3).")
    white_mask = np.all(image >= white_threshold, axis=2)
    
    white_ratio = np.mean(white_mask)
    return white_ratio < max_white_ratio
class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, data, size=80, max_cells=100000):
        self.X_nucpatch = data['x_nucpatch']  
        self.X_samplenames = data['x_imgname']  
        self.X_segmentnames = data['x_segmentname']  
        self.X_labels = data['x_tumor']  
        self.X_nucpos = data['x_nucpatch_pos']  # 
        self.X_nuc_radius = data['x_nuc_radius'] 
        tumor_labels = list(set(self.X_labels))
        self.label_map = {
            "dataBLCA": 1, "dataBRCA1": 2, "dataCESC": 3, "dataCHOL": 4, "dataCOAD": 5,
            "dataDLBC": 6, "dataESCA": 7, "dataGBM": 8, "dataHNSC": 9, "dataKICH": 10,
            "dataKIRC": 11, "dataKIRP": 12, "dataLGG": 13, "dataLIHC": 14, "dataLUAD": 15,
            "dataLUSC": 16, "dataOV": 17, "dataPAAD": 18, "dataPRAD": 19, "dataREAD": 20,
            "dataSTAD": 21, "dataTHCA": 22, "dataTHYM": 23, "dataUCEC1": 0,
            "BRCA1": 2, "KIRC": 11
        }
        self.labels = [self.label_map[x] for x in self.X_labels]
        self.size = size
        self.max_cells = max_cells  # 最大细胞数量
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])
        
    def __len__(self):
        return len(self.X_samplenames)
    
    def __getitem__(self, index):
        images_nps = self.X_nucpatch[index]  # [cell number, 56, 56, 3]
        x_samplename = self.X_samplenames[index]
        x_segname = self.X_segmentnames[index]
        x_nucpatchpos = self.X_nucpos[index]
        x_nuc_radius = self.X_nuc_radius[index]
        num_cells = images_nps.shape[0]
        label = self.labels[index]  # 获取标签
        
        if num_cells == 0:
            return torch.zeros((0, 3, self.size, self.size)), x_segname, x_samplename, 0, label, torch.zeros((1, 2)), torch.zeros(1)
        
        filtered_images = []
        filtered_positions = []
        filtered_radii = []
        
        for cell_idx, cell_patch in enumerate(images_nps):
            radius = x_nuc_radius[cell_idx]
            if is_typical_nuclei_image(cell_patch) and 2 <= radius <= 25:
                filtered_images.append(cell_patch)
                filtered_positions.append(x_nucpatchpos[cell_idx])
                filtered_radii.append(x_nuc_radius[cell_idx])
                
        num_cells = len(filtered_images)
        if num_cells == 0:
            return torch.zeros((0, 3, self.size, self.size)), x_segname, x_samplename, 0, label, torch.zeros((1, 2)), torch.zeros(1)
            
        if num_cells > self.max_cells:
            selected_cells = np.random.choice(num_cells, self.max_cells, replace=False)
            filtered_images = [filtered_images[i] for i in selected_cells]
            filtered_positions = [filtered_positions[i] for i in selected_cells]
            filtered_radii = [filtered_radii[i] for i in selected_cells]
        
        # Convert filtered_images to tensor
        transformed_images = []
        for img in filtered_images:
            img_pil = Image.fromarray(img.astype('uint8'))
            transformed_img = self.transform(img_pil)
            transformed_images.append(transformed_img)
        
        if len(transformed_images) > 0:
            image_tensor = torch.stack(transformed_images)
        else:
            image_tensor = torch.zeros((0, 3, self.size, self.size))
            
        if len(filtered_positions) == 0:
            filtered_positions = np.zeros((1, 2), dtype=np.float32)
            filtered_radii = np.zeros(1, dtype=np.float32)
        else:
            filtered_positions = np.array(filtered_positions, dtype=np.float32)
            filtered_radii = np.array(filtered_radii, dtype=np.float32)
            
        return (image_tensor, x_segname, x_samplename, len(filtered_images), label, 
                torch.from_numpy(filtered_positions), torch.from_numpy(filtered_radii))
