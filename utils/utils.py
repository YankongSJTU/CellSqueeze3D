import math
from scipy.optimize import minimize
import random
import torch.nn.functional as F
import plotly.graph_objects as go
import albumentations
from mpl_toolkits.mplot3d import Axes3D
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from sklearn.cluster import MiniBatchKMeans
import joblib
import os
import pickle
import re
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from torch.nn import init, Parameter
from torch.utils.data._utils.collate import *
from torch.utils.data.dataloader import default_collate
import torch_geometric
import albumentations
from albumentations.augmentations import transforms
from tqdm import tqdm
from math import atan2, pi
from albumentations.core.composition import Compose
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset  # For custom datasets
import cv2
import torch.utils.data
from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_auc_score
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class ImgDatasetmask(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir,segment_dir,num_classes=1, transform=None):
        self.img_ids = img_ids
        self.num_classes = num_classes
        self.transform = transform
        self.img_dir = img_dir
        self.segment_dir = segment_dir
    def __len__(self):
        return len(self.img_ids)
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_id = os.path.basename(img_id).split('.')[0]
        img = cv2.imread(self.img_dir+ "/"+img_id+".png",-1 )
        img = img.astype(np.float32)
        imgmask = cv2.imread(self.segment_dir+ "/"+img_id+".png",0 )
        return imgmask,img

def histeq2(im,nbr_bins):
        im2=np.float32(im-im.min())*np.float32(nbr_bins)/np.float32(im.max()-im.min())
        return im2

def get_patch_featuresPred(opt,imgfilelists):
    segment_dir=opt.datadir+"/"+opt.nuc_seg_dir
    img_dir=opt.datadir+"/"+opt.image_dir
    batchsize = len(imgfilelists) if opt.allpatch == "all" else min(opt.piecenumber, len(imgfilelists))
    img_transform = Compose([
        albumentations.Resize(opt.patchsize, opt.patchsize),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    img_dataset = ImgDatasetmask(img_ids=imgfilelists,img_dir = img_dir,segment_dir=segment_dir,transform=img_transform )
    img_loader = torch.utils.data.DataLoader(img_dataset,batch_size=batchsize,shuffle=False,drop_last=True)
    objregions=[]
    allobjregions=[]
    allobjregions_pos=[]
    allobjregions_no=[]
    allobjregions_radius=[]
  
    for input,  meta in tqdm(img_loader, total=len(img_loader)):
        for i in range(batchsize):
            objregions,poses,radius=find_obj_contour(opt,np.uint8(input.numpy()[i]),np.uint8(meta.numpy()[i]))
            if len(objregions)==0:
                continue
            for j in range(len(objregions)):
                allobjregions.append(objregions[j])
                allobjregions_radius.append(radius[j])
                allobjregions_pos.append(poses[j])
            allobjregions_no.append(len(poses))
            continue
    return(allobjregions,allobjregions_pos,allobjregions_no,allobjregions_radius)
def find_obj_contour(opt,grayimg,rawimg):
    _,grayimg = cv2.threshold(np.uint8(grayimg),250,1,cv2.THRESH_BINARY)
    mindistance=5
    kval=3
    npones=7
    distance = ndi.distance_transform_edt(np.uint8(grayimg))
    _, thresh = cv2.threshold(np.uint8(distance),0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    distance=np.multiply(distance,1-thresh)
    m=255-histeq2(distance,255)
    distance=cv2.GaussianBlur(255-m,(kval,kval),1)
    local_maxi = peak_local_max(distance, min_distance=mindistance, footprint=np.ones((npones,npones), dtype=np.bool_), labels=np.uint8(grayimg))
    mask=np.zeros(distance.shape,dtype=bool)
    mask[tuple(local_maxi.T)]=True
    markers,_=ndi.label(mask)
    labels = watershed(-distance, markers, mask=np.uint8(grayimg))
    m=np.uint8(histeq2(labels,255))
    maxval=np.max(labels)
    positions=[]
    nucleiradius=[]
    nucleiregion=[]
    for j in range(maxval):
        i=j+1
        tmplabel=labels.copy()
        tmplabel[tmplabel!=i]=0
        tmplabel=histeq2(tmplabel,255)
        contours,hier=cv2.findContours(np.uint8(tmplabel),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>0:
            (x,y),radius=cv2.minEnclosingCircle(contours[0])
            if radius>3:
                tmp=rawimg[max(0,int(y-radius)):min(opt.patchsize,int(y+radius)),max(0,int(x-radius)):min(opt.patchsize,int(x+radius))]
                tmp2=cv2.resize(tmp,(56,56),interpolation=(cv2.INTER_CUBIC))
                nucleiregion.append(tmp2)
                nucleiradius.append(radius)
                positions.append([x,y])
    return(nucleiregion,positions,nucleiradius)
def getCellData(opt, pat_name, pat2img):
    x_samplename,x_imgname,x_segmentname, x_nuc_patches,x_nuc_patches_pos,x_nuc_patches_no,x_nuc_patches_radius =[], [],[],[],[],[],[]
    x_samplename=pat_name
    x_imgname = [opt.datadir + "/" + opt.image_dir + "/" + img for img in pat2img[pat_name]]
    x_segmentname = [opt.datadir + "/" + opt.nuc_seg_dir + "/" + img for img in pat2img[pat_name]]
    patchfeatures,patchfeatures_pos,patchfeatures_no,patchfeatures_radius=get_patch_featuresPred(opt,pat2img[pat_name])
    x_nuc_patches.append(np.array(patchfeatures))
    x_nuc_patches_pos.append(np.array(patchfeatures_pos))
    x_nuc_patches_no.append(np.array(patchfeatures_no))
    x_nuc_patches_radius.append(np.array(patchfeatures_radius))
    return x_samplename,x_segmentname,x_imgname,x_nuc_patches,x_nuc_patches_pos,x_nuc_patches_no,x_nuc_patches_radius
def mixed_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    transposed = zip(*batch)
    if elem_type is torch_geometric.data.data.Data:
        return [Batch.from_data_list(samples, []) for samples in transposed]
    else:
        return [default_collate(samples) for samples in transposed]
def parse_gpuids(opt):
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
    return opt
def normalize_image(image):
    min_val = image.min()
    max_val = image.max()
    return (image - min_val) / (max_val - min_val)
def generate_random_image(diffusion_model, device, image_size=(3, 80, 80)):
    diffusion_model.eval()
    num_steps = diffusion_model.timesteps  # 
    with torch.no_grad():
        generated_img = torch.randn((1, *image_size), device=device)
        for t in reversed(range(0, num_steps)):  # t 的范围确保在 [0, timesteps-1]
            t_tensor = torch.tensor([t], device=device)
            beta_t = diffusion_model.beta_schedule[t].to(device)  # 计算 beta_t
            alpha_t = 1 - beta_t  # 计算 alpha_t
            alpha_bar_t = torch.cumprod(1 - diffusion_model.beta_schedule, dim=0)[t]  # 计算 alpha_bar_t
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            noise_pred = diffusion_model(generated_img, t_tensor)  # 预测噪声
            generated_img = (generated_img - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
            if t > 0:
                noise = torch.randn_like(generated_img, device=device)
                generated_img += torch.sqrt(1 - alpha_bar_t) * noise
        generated_img = generated_img.squeeze(0).cpu().detach().numpy()
        generated_img = np.transpose(generated_img, (1, 2, 0))
        generated_img = (generated_img - generated_img.min()) / (generated_img.max() - generated_img.min())
        generated_img = (generated_img * 255).astype(np.uint8)
        cv2.imwrite("generated_cell.png", generated_img)
        plt.figure(figsize=(5, 5))
        plt.imshow(generated_img)
        plt.axis("off")
        plt.savefig("generated_cell_visualization.png")
        plt.close()
        print("随机生成的细胞图像已保存：generated_cell.png")

def visualize_slices(he_image, simulated_slice, batch_idx,imgname="img.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(he_image, cmap='gray')
    axes[0].set_title(f"Original HE Image - Batch {batch_idx}")
    axes[0].axis("off")
    axes[1].imshow(simulated_slice, cmap='gray')
    axes[1].set_title(f"Simulated Slice - Batch {batch_idx}")
    axes[1].axis("off")
    plt.savefig(imgname, bbox_inches='tight')
def visualize_3d_cells(cells):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    for cell in cells:
        x, y, z = cell['center']
        radius = cell['radius']
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = x + radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = y + radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = z + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='b', alpha=0.888888885)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1]) 
    ax.set_title("3D Cell Distribution")
    ax.view_init(elev=30, azim=45)
    plt.show()
    plt.savefig('tmpcell3D.png', bbox_inches='tight')
def visualize_combined_canuse(he_image, cells, imgname="tmp3D.png"):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(he_image, cmap='gray')
    ax1.set_title("Real HE Image")
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title("3D Cell Distribution")
    vertices = []
    faces = []
    colors = []  # List to store color information for each vertex
    face_offset = 0
    cmap = get_cmap('viridis')
    z_values = [cell['center'][2] for cell in cells]
    z_min, z_max = min(z_values), max(z_values)
    z_normalized = [(z - z_min) / (z_max - z_min) for z in z_values]
    light_dir = np.array([1, 1, 1])
    light_dir = light_dir / np.linalg.norm(light_dir)
    for idx, cell in enumerate(cells):
        x, y, z = cell['center']
        radius = cell['radius']
        color = cmap(z_normalized[idx])[:3]  # Get RGB color from colormap
        base_color = cmap(z_normalized[idx])[:3] 
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x_sphere = x + radius * np.outer(np.cos(u), np.sin(v)).flatten()
        y_sphere = y + radius * np.outer(np.sin(u), np.sin(v)).flatten()
        z_sphere = z + radius * np.outer(np.ones(np.size(u)), np.cos(v)).flatten()
        normals = np.vstack([x_sphere - x, y_sphere - y, z_sphere - z]).T
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        light_intensity = np.clip(np.dot(normals, light_dir), 0, 1)
        sphere_colors = base_color * light_intensity[:, np.newaxis]
        sphere_vertices = np.vstack([x_sphere, y_sphere, z_sphere]).T
        # Generate faces
        num_u = len(u)
        num_v = len(v)
        sphere_faces = []
        for i in range(num_u - 1):
            for j in range(num_v - 1):
                idx = i * num_v + j
                sphere_faces.append([idx + face_offset, idx + num_v + face_offset, idx + num_v + 1 + face_offset])
                sphere_faces.append([idx + face_offset, idx + num_v + 1 + face_offset, idx + 1 + face_offset])
        vertices.extend(sphere_vertices)
        faces.extend(sphere_faces)
        colors.extend(sphere_colors)
        face_offset += len(sphere_vertices)
    vertices = np.array(vertices)
    faces = np.array(faces)
    colors = np.array(colors)
    mesh = meshio.Mesh(
        points=vertices,
        cells=[("triangle", faces)],
        point_data={"colors": colors}  # Add color information as vertex attributes
    )
    mesh.write("cells_3d.ply")
from matplotlib.cm import get_cmap
import trimesh
def visualize_combined(he_image, cells, imgid="tmp3D"):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(he_image, cmap='gray')
    ax1.set_title("Real HE Image")
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title("3D Cell Distribution")
    vertices = []
    faces = []
    colors = []  # List to store color information for each vertex
    face_offset = 0
    cmap = get_cmap('viridis')
    z_values = [cell['center'][2] for cell in cells]
    z_min, z_max = min(z_values), max(z_values)
    z_normalized = [(z - z_min) / (z_max - z_min) for z in z_values]
    light_dir = np.array([1, 1, 1])
    light_dir = light_dir / np.linalg.norm(light_dir)
    for idx, cell in enumerate(cells):
        x, y, z = cell['center']
        radius = cell['radius']
        base_color = cmap(z_normalized[idx])[:3]  # Get RGB color from colormap
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x_sphere = x + radius * np.outer(np.cos(u), np.sin(v)).flatten()
        y_sphere = y + radius * np.outer(np.sin(u), np.sin(v)).flatten()
        z_sphere = z + radius * np.outer(np.ones(np.size(u)), np.cos(v)).flatten()
        sphere_vertices = np.vstack([x_sphere, y_sphere, z_sphere]).T
        # Generate sphere faces
        num_u = len(u)
        num_v = len(v)
        sphere_faces = []
        for i in range(num_u - 1):
            for j in range(num_v - 1):
                idx = i * num_v + j
                sphere_faces.append([idx + face_offset, idx + num_v + face_offset, idx + num_v + 1 + face_offset])
                sphere_faces.append([idx + face_offset, idx + num_v + 1 + face_offset, idx + 1 + face_offset])
        # Calculate normals and apply shading
        normals = np.vstack([x_sphere - x, y_sphere - y, z_sphere - z]).T
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        light_intensity = np.clip(np.dot(normals, light_dir), 0, 1)
        sphere_colors = base_color * light_intensity[:, np.newaxis]
        # Add to global vertices, faces, and colors
        vertices.extend(sphere_vertices)
        faces.extend(sphere_faces)
        colors.extend(sphere_colors)
        face_offset += len(sphere_vertices)
    vertices = np.array(vertices)
    faces = np.array(faces)
    colors = np.array(colors)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
    mesh.export(imgid+".cells_3d.ply")
def generate_slice(cells, z_slice, image_shape, draw_boundary=False):
    slice_image = np.ones(image_shape, dtype=np.uint8) * 255
    overlay = np.zeros((image_shape[0], image_shape[1], 4), dtype=np.uint8)  # RGBA image
    for cell in cells:
        x, y, z = cell['center']
        R = cell['radius']
        if isinstance(R, torch.Tensor):
            R = R.detach().cpu().numpy()
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
        if abs(z - z_slice) <= R:
            radius_2d = int(np.sqrt(R**2 - (z - z_slice)**2))  # Compute the 2D radius
            cv2.circle(overlay, (int(x), int(y)), radius_2d, (255, 0, 0, 204), -1)  # Blue with 80% opacity
            if draw_boundary:
                cv2.circle(overlay, (int(x), int(y)), int(radius_2d), (0, 0, 255, 255), 1)  # Red boundary
    alpha = overlay[:, :, 3] / 255.0  # Extract the alpha channel
    for c in range(3):  # Blend RGB channels
        slice_image[:, :, c] = slice_image[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha
    return slice_image
def custom_collate_fn(batch):
    x_nucpatches,x_segnames, x_samplenames, num_cells_list, labels, poses, radiuses = zip(*batch)

    max_cells = max(num_cells_list)
    padded_patches = []
    padded_poses = []  # 用于存储填充后的 poses
    padded_radiuses = []  # 用于存储填充后的 poses
    masks = []

    for x_nucpatch, pose, radius, num_cells in zip(x_nucpatches, poses, radiuses, num_cells_list):
        padding = torch.zeros((max_cells - num_cells, x_nucpatch.shape[1], x_nucpatch.shape[2], x_nucpatch.shape[3]))
        padded_patch = torch.cat([x_nucpatch, padding], dim=0)
        padded_patches.append(padded_patch)

        pose_padding = torch.zeros((max_cells - num_cells, 2))
        padded_pose = torch.cat([pose, pose_padding], dim=0)
        padded_poses.append(padded_pose)

        radius_padding = torch.zeros(max_cells - num_cells)
        padded_radius = torch.cat([radius, radius_padding], dim=0)
        padded_radiuses.append(padded_radius)

        mask = torch.cat([torch.ones(num_cells, dtype=torch.bool), torch.zeros(max_cells - num_cells, dtype=torch.bool)])
        masks.append(mask)


    batched_patches = torch.stack(padded_patches)  # [batch_size, max_cells, 3, size, size]
    batched_poses = torch.stack(padded_poses)  # [batch_size, max_cells, 2]
    batched_radiuses = torch.stack(padded_radiuses)  # [batch_size, max_cells]
    batched_masks = torch.stack(masks)  # [batch_size, max_cells]
    batched_samplenames = list(x_samplenames)
    batched_segnames = list(x_segnames)
    batched_labels = torch.tensor(labels, dtype=torch.long)  # 假设标签是整数类型


    return batched_patches, batched_masks,batched_segnames, batched_samplenames, batched_labels, batched_poses, batched_radiuses

