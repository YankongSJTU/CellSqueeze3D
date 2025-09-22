import torch
import pickle
import numpy as np
from utils.DataSets import *
from mpl_toolkits.mplot3d import Axes3D
from utils.utils import *
from torchvision.models import vgg16
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from utils.bound import *
import argparse
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from skimage.metrics import structural_similarity as ssim
import trimesh
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import cv2
from scipy.spatial import KDTree

os.environ["QT_QPA_PLATFORM"] = "offscreen"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--time_steps', type=int, default=30, help='iteration steps')  
    parser.add_argument('--data', type=str, default="./data/traindata.pkl", help="Data file")
    parser.add_argument('--num_workers', type=int, default=10, help="Number of parallel workers")
    parser.add_argument('--save_freq', type=int, default=10, help="Save frequency for results")
    opt = parser.parse_known_args()[0]
    return opt

opt = parse_args()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cpu"

def fast_overlap_check(cells):
    """KDTree for neighbors"""
    positions = np.array([cell['center'] for cell in cells])
    radii = np.array([cell['radius'] for cell in cells])
    tree = KDTree(positions)
    overlap_pairs = tree.query_pairs(np.max(radii) * 2)  # find overlap pairs
    overlaps = []
    for i, j in overlap_pairs:
        d = np.linalg.norm(positions[i] - positions[j])
        if d < radii[i] + radii[j]:
            overlaps.append((i, j, radii[i] + radii[j] - d))
    return overlaps

def initialize_cells(nuclei_info, z_max=50):
    cells = []
    for i, nucleus in enumerate(nuclei_info):
        x, y = nucleus['center']
        r_nucleus = nucleus['radius']
        
        for attempt in range(100):  #  
            z = np.random.uniform(0, z_max)
            r = np.random.uniform(1.5 * r_nucleus, 2.5 * r_nucleus)
            new_cell = {
                'id': i,
                'center': (x, y, z),
                'radius': r,
                'nucleus_radius': r_nucleus
            }
            overlap = False
            for other in cells:
                d_2d = np.linalg.norm(np.array([x, y]) - np.array(other['center'][:2]))
                if d_2d < (r_nucleus + other['nucleus_radius']):
                    theta = np.random.uniform(0, 2*np.pi)
                    min_d = r_nucleus + other['nucleus_radius']
                    x = other['center'][0] + min_d * np.cos(theta)
                    y = other['center'][1] + min_d * np.sin(theta)
                    break
            if not overlap:
                cells.append(new_cell)
                break
        else:
            cells.append({
                'id': i,
                'center': (x, y, z_max / 2),
                'radius': 2.5 * r_nucleus,
                'nucleus_radius': r_nucleus
            })
    return cells
def gpu_vectorized_fitness(positions, cells, z_max=50, device='cpu'):
    """gpu-accelerated fitness function"""
    n_particles = positions.shape[0]
    n_cells = len(cells)
    positions = torch.tensor(positions, dtype=torch.float32, device=device)
    z_values = positions[:, :n_cells]
    r_values = positions[:, n_cells:]
    r_nucleus = torch.tensor([c['nucleus_radius'] for c in cells], 
                            dtype=torch.float32, device=device)
    r_min = 1.5 * r_nucleus
    r_max = 2.5 * r_nucleus
    r_min_violation = torch.sum(torch.relu(r_min - r_values), dim=1)
    r_max_violation = torch.sum(torch.relu(r_values - r_max), dim=1)
    z_violation = torch.sum(torch.relu(-z_values) + torch.relu(z_values - z_max), dim=1)
    penalty = 5e4 * (r_min_violation + r_max_violation) + 1e4 * z_violation
    centers = torch.stack([
        torch.tensor([c['center'][0], c['center'][1]], device=device).repeat(n_particles, 1)
        for c in cells
    ])
    centers = torch.cat([centers, z_values.t.unsqueeze(-1)], dim=-1)
    for i in range(n_cells):
        for j in range(i+1, n_cells):
            dist = torch.norm(centers[i] - centers[j], dim=1)
            min_dist = r_nucleus[i] + r_nucleus[j]
            penalty += 1e6 * torch.relu(min_dist - dist)
    return penalty.cpu().numpy()
def vectorized_fitness(positions, cells, z_max=50):
    """ """
    n_particles = positions.shape[0]
    n_cells = len(cells)
    z_values = positions[:, :n_cells]
    r_values = positions[:, n_cells:]
    r_nucleus = np.array([c['nucleus_radius'] for c in cells])
    r_min = 1.5 * r_nucleus
    r_max = 2.5 * r_nucleus
    r_min_violation = np.sum(np.maximum(r_min - r_values, 0), axis=1)
    r_max_violation = np.sum(np.maximum(r_values - r_max, 0), axis=1)
    z_violation = np.sum(np.maximum(-z_values, 0) + np.maximum(z_values - z_max, 0), axis=1)
    penalty = 5e4 * (r_min_violation + r_max_violation) + 1e4 * z_violation
    for p_idx in range(n_particles):
        centers = np.array([(c['center'][0], c['center'][1], z) 
                          for c, z in zip(cells, z_values[p_idx])])
        tree = KDTree(centers)
        neighbor_pairs = list(tree.query_pairs(np.max(r_values[p_idx]) * 2))
        for i, j in neighbor_pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            min_distance = cells[i]['nucleus_radius'] + cells[j]['nucleus_radius']
            
            if d < min_distance:
                penalty[p_idx] += 1e6 * (min_distance - d)
    return penalty
def visualize_iteration(cells, params, iteration, n_cells):
    """visualize steps"""
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    z_values = params[:n_cells]
    radii = params[n_cells:]
    ax1.scatter(
        [c['center'][0] for c in cells],
        [c['center'][1] for c in cells],
        z_values,
        c='r', s=10
    )
    for i in np.random.choice(len(cells), 5):   
        x, y,_ = cells[i]['center']
        z = z_values[i]
        r = radii[i]
        u = np.linspace(0, 1.5 * np.pi, 15)
        v = np.linspace(0, np.pi, 15)
        xx = x + r * np.outer(np.cos(u), np.sin(v))
        yy = y + r * np.outer(np.sin(u), np.sin(v))
        zz = z + r * np.outer(np.ones(np.size(u)), np.cos(v))

        ax1.plot_surface(xx, yy, zz, color='b', alpha=0.1)

    ax1.set_title(f"Iteration {iteration}\nZ Distribution")
    ax2 = fig.add_subplot(122)
    nucleus_radii = [c['nucleus_radius'] for c in cells]
    ax2.scatter(nucleus_radii, radii, alpha=0.6)
    ax2.plot([0, max(nucleus_radii)*5], [0, max(nucleus_radii)*5], 'k--')
    ax2.set_xlabel("Nucleus Radius")
    ax2.set_ylabel("Optimized Radius")
    ax2.set_title("Radius Relationship")

    plt.tight_layout()
    plt.savefig(f"pso_iter_{iteration:04d}.png")
    plt.close()

def pso_optimize_cells(cells, imgid, max_iter=38, n_swarm=50,
                      w_start=0.9, w_end=0.4,
                      c1_start=2.5, c1_end=1.0,
                      c2_start=1.0, c2_end=2.6,
                      z_max=40, visual_freq=10):
    if not cells:
        raise valueerror("empty cells list provided")
    n_cells = len(cells)
    if any('nucleus_radius' not in c for c in cells):
        raise valueerror("missing nucleus_radius in cell data")
    dim = n_cells * 2
    particles_pos = np.zeros((n_swarm, dim))
    particles_vel = np.random.uniform(-1, 1, (n_swarm, dim)) * 0.1
    nucleus_radii = np.array([c['nucleus_radius'] for c in cells])
    
    for i in range(n_swarm):
        particles_pos[i, :n_cells] = np.random.uniform(0, z_max)
        particles_pos[i, n_cells:] = np.random.uniform(nucleus_radii * 1.5, nucleus_radii * 3.0)
    pbest_pos = particles_pos.copy()
    pbest_fitness = np.full(n_swarm, np.inf)
    gbest_pos = None
    gbest_fitness = np.inf
    history = []
    def get_params(iter_ratio):
        w = w_start - (w_start - w_end) * iter_ratio
        c1 = c1_start - (c1_start - c1_end) * iter_ratio
        c2 = c2_start + (c2_end - c2_start) * iter_ratio
        return w, c1, c2
    best_fitness = []
    with tqdm(total=max_iter, desc=f"pso optimizing {imgid}") as pbar:
        for iter in range(max_iter):
            iter_ratio = iter / max_iter
            w, c1, c2 = get_params(iter_ratio)
            current_fitness = vectorized_fitness(particles_pos, cells, z_max)
            improved = current_fitness < pbest_fitness
            pbest_pos[improved] = particles_pos[improved]
            pbest_fitness[improved] = current_fitness[improved]
            if np.min(current_fitness) < gbest_fitness:
                gbest_idx = np.argmin(current_fitness)
                gbest_pos = particles_pos[gbest_idx].copy()
                gbest_fitness = current_fitness[gbest_idx]
            
            history.append(gbest_fitness)
            best_fitness.append(gbest_fitness)
            
            r1 = np.random.rand(n_swarm, dim)
            r2 = np.random.rand(n_swarm, dim)
            particles_vel = w * particles_vel \
                + c1 * r1 * (pbest_pos - particles_pos) \
                + c2 * r2 * (gbest_pos - particles_pos)
            
            particles_pos += particles_vel
            particles_pos[:, :n_cells] = np.clip(particles_pos[:, :n_cells], 0, z_max)
            particles_pos[:, n_cells:] = np.clip(particles_pos[:, n_cells:], nucleus_radii * 1.5, nucleus_radii * 2.5)
           # if visual_freq > 0 and iter % visual_freq == 0:
           #     visualize_iteration(cells, gbest_pos, iter, n_cells)
            pbar.update(1)
            pbar.set_postfix({'fitness': gbest_fitness})
    
    optimized_cells = []
    for i in range(n_cells):
        optimized_cells.append({
            'id': cells[i]['id'],
            'center': (
                cells[i]['center'][0],  #  keep x
                cells[i]['center'][1],  # keep y
                gbest_pos[i]           #  z
            ),
            'radius': gbest_pos[n_cells + i],
            'nucleus_radius': cells[i]['nucleus_radius']
        })
    if visual_freq > 0:
        plt.figure()
        plt.plot(history)
        plt.title("fitness convergence")
        plt.xlabel("iteration")
        plt.ylabel("fitness score")
        plt.savefig(f"{imgid}_pso_convergence.png")
        plt.close()
    
    return optimized_cells

def create_sphere_mesh(center, radius):
    """creat sphere"""
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)  #  
    mesh.apply_translation(center)
    return mesh

def save_spheres_as_ply(spheres, filename="spheres.ply"):
    """save ply"""
    if not spheres:
        return
    
    meshes = []
    for sphere in spheres:
        center = sphere['center']
        radius = sphere['radius']
        mesh = create_sphere_mesh(center, radius)
        meshes.append(mesh)
    
    combined_mesh = trimesh.util.concatenate(meshes)
    combined_mesh.export(filename)

def process_single_sample(args):
    sample_data, sample_idx, batch_idx, opt = args
    inputs, masks, segname, imgname, labels, pos, radius = sample_data
    
    imgid = os.path.basename(imgname[0]).split('.')[0] if isinstance(imgname, list) else os.path.basename(imgname).split('.')[0]
    imgid = f"{imgid}_b{batch_idx}_s{sample_idx}"
    
    try:
        nuclei_info = []
        for j in range(len(pos)):
            [x, y] = pos[j]
            x, y = x.item(), y.item()
            radius_one = radius[j].item()
            nuclei_info.append({'center': (int(x), int(y)), 'radius': int(radius_one)})
        
        # initialization cells
            cells = initialize_cells(nuclei_info)
        
        # PSO optimization
        optimized_cells = pso_optimize_cells(cells, imgid, max_iter=200, n_swarm=30)  # 减少迭代次数和粒子数
        
        #save 
        save_path = os.path.join(opt.checkpoints_dir, "results", f"{imgid}_optimized_cells.ply")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_spheres_as_ply(optimized_cells, filename=save_path)
        
        return optimized_cells, imgid
        
    except Exception as e:
        print(f"Error processing sample {imgid}: {e}")
        return None, imgid

def train_parallel(dataloader, device, opt):
    """parallel training"""
    all_results = []
    
    # prepare parameters
    args_list = []
    for batch_idx, batch_data in enumerate(dataloader):
        inputs, masks, segname, imgname, labels, pos, radius = batch_data
        
        batch_size = len(inputs) if hasattr(inputs, '__len__') else 1
        
        for sample_idx in range(batch_size):
            sample_data = (
                inputs[sample_idx] if batch_size > 1 else inputs,
                masks[sample_idx] if batch_size > 1 else masks,
                segname[sample_idx] if batch_size > 1 else segname,
                imgname[sample_idx] if batch_size > 1 else imgname,
                labels[sample_idx] if batch_size > 1 else labels,
                pos[sample_idx] if batch_size > 1 else pos[0],
                radius[sample_idx] if batch_size > 1 else radius[0]
            )
            
            args_list.append((sample_data, sample_idx, batch_idx, opt))
    
    # 
    print(f"Starting parallel processing with {opt.num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=opt.num_workers) as executor:
        futures = [executor.submit(process_single_sample, args) for args in args_list]
        
        # tqmd 
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
            try:
                result, imgid = future.result()
                if result is not None:
                    all_results.append((result, imgid))
                    if len(all_results) % opt.save_freq == 0:
                        print(f"Processed {len(all_results)} samples")
            except Exception as e:
                print(f"Error in future: {e}")
    
    return all_results
def train_sequential(dataloader, device, opt):
 
    all_results = []
    for batch_idx, (inputs, masks, segname, imgname, labels, pos, radius) in enumerate(dataloader):
        print(f"Processing batch {batch_idx}/{len(dataloader)}")
        batch_size = len(inputs) if hasattr(inputs, '__len__') else 1
        for sample_idx in range(batch_size):
            #  
            current_pos = pos[sample_idx] if batch_size > 1 else pos[0]
            current_radius = radius[sample_idx] if batch_size > 1 else radius[0]
            current_imgname = imgname[sample_idx] if batch_size > 1 else imgname[0]
            imgid = os.path.basename(current_imgname).split('.')[0]
            imgid = f"{imgid}_b{batch_idx}_s{sample_idx}"
            nuclei_info = []
            for j in range(len(current_pos)):
                [x, y] = current_pos[j]
                x, y = x.item(), y.item()
                radius_one = current_radius[j].item()
                nuclei_info.append({'center': (int(x), int(y)), 'radius': int(radius_one)})
            cells = initialize_cells(nuclei_info)
            optimized_cells = pso_optimize_cells(cells, imgid, max_iter=40, n_swarm=30)
            cell_boundaries = get_cell_boundaries(optimized_cells, imgid)
            save_path_ply = os.path.join(opt.checkpoints_dir, "results", f"{imgid}_optimized_cells.ply")
            os.makedirs(os.path.dirname(save_path_ply), exist_ok=True)
            save_spheres_as_ply(optimized_cells, filename=save_path_ply)
            save_path_json = os.path.join(opt.checkpoints_dir, "results", f"{imgid}_cell_boundaries.json")
            with open(save_path_json, 'w') as f:
                json.dump(cell_boundaries, f)
            
            #  
            all_results.append({
                'imgid': imgid,
                'optimized_cells': optimized_cells,
                'cell_boundaries': cell_boundaries
            })
            
            #  
            print(f"Processed sample {imgid} - {len(optimized_cells)} cells")
            
    return all_results

def main():
    # load data
    traindata = pickle.load(open(opt.data, 'rb'))
    custom_data_loader_train = DatasetLoader(traindata, size=80)
    train_loader = torch.utils.data.DataLoader(
        dataset=custom_data_loader_train, 
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    #  
    os.makedirs(os.path.join(opt.checkpoints_dir, "results"), exist_ok=True)
    
    results = train_sequential(train_loader, device, opt)
    print(f"Training completed. Processed {len(results)} samples.")
        

if __name__ == "__main__":
    # 
    if hasattr(mp, 'get_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    main()
