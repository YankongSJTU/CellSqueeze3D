import json
import numpy as np
import sys
import pandas as pd
import math
from collections import Counter
import os
from scipy.spatial import ConvexHull
import glob

def calculate_area_from_boundary(boundary):
    """calculate """
    if len(boundary) < 3:
        return 0

    try:
        n = len(boundary)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += boundary[i][0] * boundary[j][1]
            area -= boundary[j][0] * boundary[i][1]
        return abs(area) / 2.0
    except:
        return 0

def calculate_entropy(values, bins='auto'):
    """calculate entropy"""
    if len(values) <= 1:
        return 0

    values = np.array(values)
    values = values[np.isfinite(values)]

    if len(values) <= 1:
        return 0

    try:
        hist, _ = np.histogram(values, bins=bins, density=False)

        hist = hist[hist > 0]

        if len(hist) <= 1:
            return 0

        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    except:
        return 0

def process_json_file(json_file_path):
    """dule with single json file """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_file_path}: {str(e)}")
        return None

    nuclear_areas = []
    cell_volumes = []
    nuclear_radii = []
    cell_radii = []

    for cell in data:
        nuclear_area = 0
        if 'boundary' in cell and cell['boundary'] and len(cell['boundary']) >= 3:
            nuclear_area = calculate_area_from_boundary(cell['boundary'])
        elif 'nuc_radius' in cell and cell['nuc_radius'] > 0:
            nuclear_area = math.pi * (cell['nuc_radius'] ** 2)

        cell_volume = 0
        if 'radius' in cell and cell['radius'] > 0:
            cell_volume = (4/3) * math.pi * (cell['radius'] ** 3)

        if nuclear_area > 0 and cell_volume > 0:
            nuclear_areas.append(nuclear_area)
            cell_volumes.append(cell_volume)
            if 'nuc_radius' in cell:
                nuclear_radii.append(cell['nuc_radius'])
            if 'radius' in cell:
                cell_radii.append(cell['radius'])

    results = {
        'sample_name': os.path.basename(json_file_path).replace('.json', ''),
        'nuclear_area_entropy': calculate_entropy(nuclear_areas),
        'cell_volume_entropy': calculate_entropy(cell_volumes),
        'nuclear_radius_entropy': calculate_entropy(nuclear_radii),
        'cell_radius_entropy': calculate_entropy(cell_radii),
        'num_cells': len(nuclear_areas),
        'mean_nuclear_area': np.mean(nuclear_areas) if nuclear_areas else 0,
        'mean_cell_volume': np.mean(cell_volumes) if cell_volumes else 0,
        'mean_nuclear_radius': np.mean(nuclear_radii) if nuclear_radii else 0,
        'mean_cell_radius': np.mean(cell_radii) if cell_radii else 0,
        'std_nuclear_area': np.std(nuclear_areas) if nuclear_areas else 0,
        'std_cell_volume': np.std(cell_volumes) if cell_volumes else 0
    }

    return results

def process_all_json_files(json_folder):
    """ Load all json files"""
    results = []
    json_files = glob.glob(os.path.join(json_folder, '*.json'))

    print(f"Found {len(json_files)} JSON files to process")

    for json_file in json_files:
        try:
            result = process_json_file(json_file)
            if result:
                results.append(result)
                print(f"Processed: {result['sample_name']} - {result['num_cells']} cells")
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")

    if not results:
        print("No valid results to save")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    columns_order = [
        'sample_name', 'num_cells',
        'nuclear_area_entropy', 'cell_volume_entropy',
        'nuclear_radius_entropy', 'cell_radius_entropy',
        'mean_nuclear_area', 'mean_cell_volume',
        'mean_nuclear_radius', 'mean_cell_radius',
        'std_nuclear_area', 'std_cell_volume'
    ]
    df = df[columns_order]

    output_csv = os.path.join(json_folder, 'sample_entropy_analysis.csv')
    df.to_csv(output_csv, index=False, float_format='%.6f')
    print(f"Results saved to: {output_csv}")

    return df

if __name__ == "__main__":
    json_folder = sys.argv[1]

    results_df = process_all_json_files(json_folder)

    if not results_df.empty:
        print("\n Results:")
        print(results_df.head())

        print("\n basic statistics:")
        print(results_df.describe())

        stats_csv = os.path.join(json_folder, 'entropy_statistics.csv')
        results_df.describe().to_csv(stats_csv)
        print(f"Statistics saved to: {stats_csv}")
    else:
        print("No data was processed")
