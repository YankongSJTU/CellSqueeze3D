import argparse
import os
import pickle
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing
from utils.utils import *
from multiprocessing.pool import Pool
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data', help="datasets")
    parser.add_argument('--image_dir', type=str, default='image')
    parser.add_argument('--tumortype', type=str, default='dataBRCA1')
    parser.add_argument('--nuc_seg_dir', type=str, default='segment')
    parser.add_argument('--splitlist', type=str, default='splitlist.csv') # Samples split into train and test
    parser.add_argument('--piecenumber', type=int, default='90') # Samples split into train and test
    parser.add_argument('--patchsize', type=int, default='1000') # Samples split into train and test
    parser.add_argument('--allpatch', type=str, default='select') # Samples split into train and test
    parser.add_argument('--basenamelen', type=int, default='12',help='length of basename in spllist.csv') # Samples split into train and test
    opt = parser.parse_known_args()[0]
    return opt

opt = parse_args()
img_fnames = os.listdir(os.path.join(opt.datadir, opt.image_dir))
pat2img = {}

for img_fname in img_fnames:
    pat = img_fname[:int(opt.basenamelen)]
    if pat not in pat2img:
        pat2img[pat] = []
    pat2img[pat].append(img_fname)


CVf_split = pd.read_csv(opt.datadir+'/'+opt.splitlist,header=None)
CV_train = CVf_split[CVf_split.iloc[:,1].isin(['train', 'test'])]

missing_keys = []
for pat_name in CV_train.iloc[:, 0]:
    if pat_name not in pat2img:
        missing_keys.append(pat_name)

if missing_keys:
    print(f"Waring: Key not exist: {missing_keys}")
    print("Available keys:", list(pat2img.keys()))

#CV_test=CVf_split[CVf_split.iloc[:,1] == 'test']
def generate_data(opt, pat2img, split_df, trainortest):
    pool = Pool(processes=10)
    results = pool.starmap_async(getCellData, [(opt, pat_name, pat2img) for pat_name in split_df.iloc[:, 0]])
    pool.close()
    pool.join()
    results.wait()
    outputs = results.get()
    sample_names, segment_names,img_names, nuc_patches, nuc_patches_pos, nuc_patches_no,nuc_patches_radius =[], [], [], [], [], [],[]
    for (
        sample_name_pool,
        segment_name_pool,
        img_name_pool,
        nuc_patch_pool,
        nuc_patch_pos_pool,
        nuc_patch_no_pool,
        nuc_patch_radius_pool,
    ) in outputs:
        sample_names.append(sample_name_pool)
        segment_names.append(segment_name_pool)
        img_names.append(img_name_pool)
        nuc_patches.append(nuc_patch_pool)
        nuc_patches_pos.append(nuc_patch_pos_pool)
        nuc_patches_no.append(nuc_patch_no_pool)
        nuc_patches_radius.append(nuc_patch_radius_pool)

    data = {
        "x_samplename": sample_names,
        "x_segmentname": segment_names,
        "x_imgname": img_names,
        "x_nucpatch": nuc_patches,
        "x_nucpatch_pos": nuc_patches_pos,
        "x_nucpatch_no": nuc_patches_no,
        "x_nuc_radius": nuc_patches_radius
    }

    output_file = opt.datadir+".traindata.pkl"
    #output_file = opt.datadir+"/"+opt.datadir+".traindata.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Generated data saved to: {output_file}")
    return output_file

# merge data
def merge_data(input_file, output_file,tumortype):
    merged_data = {"x_segmentname": [], "x_tumor": [], "x_nucpatch": [], "x_imgname": [], "x_nucpatch_pos": [],"x_nuc_radius":[]}


    with open(input_file, "rb") as f:
        print("Processing: " + input_file + "\n")
        data = pickle.load(f)

        for i in range(len(data["x_samplename"])):
            j=0
            for k in range(len(data["x_imgname"][i])):
                cellrankl = np.sum(data["x_nucpatch_no"][i][0][:j])
                cellrankh = np.sum(data["x_nucpatch_no"][i][0][:(j + 1)])
                if cellrankh > cellrankl:
                    merged_data["x_imgname"].append(data["x_imgname"][i][j])
                    merged_data["x_segmentname"].append(data["x_segmentname"][i][j])
                    merged_data["x_nucpatch"].append(data["x_nucpatch"][i][0][cellrankl:cellrankh])
                    merged_data["x_nucpatch_pos"].append(data["x_nucpatch_pos"][i][0][cellrankl:cellrankh])
                    merged_data["x_nuc_radius"].append(data["x_nuc_radius"][i][0][cellrankl:cellrankh])
                    merged_data["x_tumor"].append(tumortype)
                    j=j+1

    # save data
    with open(output_file, "wb") as f:
        pickle.dump(merged_data, f)
    print(f"Merged data saved to: {output_file}")

tumortype=opt.tumortype
train_input_file = generate_data(opt, pat2img, CV_train, "train")
train_output_file = opt.datadir+".traindata.pkl"
#train_output_file = opt.datadir+"/"+opt.datadir+".traindata.pkl"
merge_data(train_input_file, train_output_file,tumortype)
