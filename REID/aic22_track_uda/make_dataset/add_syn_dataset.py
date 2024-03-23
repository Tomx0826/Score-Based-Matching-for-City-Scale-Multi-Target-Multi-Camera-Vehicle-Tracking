import os
import cv2
import multiprocessing as mp
import glob
import random
from multiprocessing import Pool

random.seed(44)

data_root = "/home/data/AIC22_ZongYe/ReID/"
train_des_dir = "train_reID2"
val_des_dir = "validation_reID2"

def parse_file(gt_path):
    label_dict = {}
    with open(gt_path, "r") as f:
        gts = f.readlines()
        for line in gts:
            line = line.strip().split(",")[:-4]
            frame = "{:04d}".format(int(line[0]))
            car_id = str(line[1])          
            left = int(line[2])
            top = int(line[3])
            right = left + int(line[4])
            bot = top + int(line[5])
            
            if frame not in label_dict:
                label_dict[frame] = []
            label_dict[frame].append([left, top, right, bot, car_id])
   
    return label_dict
    
    
def crop_car_image(gt_path):
#    pid = mp.current_process().name.split('-')[-1]
#    print(pid)

    info = gt_path.split('/')
    scenario = info[-4]
    camera = info[-3]
    frame_dir = os.path.join(data_root, info[-5]+'_frame', scenario, camera)
    
    label_dict = parse_file(gt_path)
    for frame, cars in label_dict.items():
        filename = scenario + '_' + camera + '_' + frame + '.jpg'
        frame_path = os.path.join(frame_dir, filename)
        img = cv2.imread(frame_path)
        
        for car in cars:
            left, top, right, bot, car_id = car
            cropped_img = img[top:bot, left:right]
            
            result_dir = os.path.join(data_root, info[-5]+'_reID', car_id)
            if not os.path.isdir(result_dir):
                os.makedirs(result_dir, exist_ok=True)
            
            result_name = filename[:-4] + '_' + car_id + '.jpg'
            result_path = os.path.join(result_dir, result_name)    
            cv2.imwrite(result_path, cropped_img) 


def main():
    now_train_dir = os.path.join(data_root, train_des_dir)
    now_val_dir = os.path.join(data_root, val_des_dir)
    now_id_count = len(os.listdir(now_train_dir)) + len(os.listdir(now_val_dir)) + 1 #667
    print(now_id_count)
    scenario = "S07"
    syn_dataset_dir = "/home/data/AIC22_ZongYe/ReID/syn/"
    syn_imgs = sorted(glob.glob(syn_dataset_dir+"*.jpg"))
    
    dict_id_to_train_or_val = {}
    
    for img_path in syn_imgs:
        
        tid, _, _, _, _, _ = os.path.basename(img_path).split("_")
        
        if tid not in dict_id_to_train_or_val:
            dict_id_to_train_or_val[tid] = []
        dict_id_to_train_or_val[tid].append(img_path)
    
    for tid in dict_id_to_train_or_val:
        img_set = dict_id_to_train_or_val[tid]
        
        
        des = random.randint(1,10)
        if des == 5:
            result_dir = os.path.join(now_val_dir,str(now_id_count))
            #result_dir = os.path.join("/home/data/AIC22_ZongYe/ReID/validation_syn",str(now_id_count))
        else:
            result_dir = os.path.join(now_train_dir,str(now_id_count))
            #result_dir = os.path.join("/home/data/AIC22_ZongYe/ReID/train_syn",str(now_id_count))
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir, exist_ok=True)
         
        fid = 1
        for img_path in img_set:
            img = cv2.imread(img_path)
            _, cam, _, _, _, _  = os.path.basename(img_path).split("_")
            result_name = f'{scenario}_{cam}_{fid:04d}_{now_id_count}.jpg'
            cv2.imwrite(os.path.join(result_dir,result_name), img)
            fid += 1
            
        now_id_count += 1
if __name__ == "__main__":
    main()
    