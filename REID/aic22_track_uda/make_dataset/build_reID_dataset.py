import os
import cv2
import multiprocessing as mp

from multiprocessing import Pool


data_root = "/Disk_New/AI_City_Challenge/AIC21-MTMC/datasets/AIC21_Track3_MTMC_Tracking"

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
            
            
            car_id = int(car_id)-1
            assert car_id >= 0
            car_id = str(car_id)
            
            result_dir = os.path.join(data_root, info[-5]+'_reID', car_id)
            if not os.path.isdir(result_dir):
                os.makedirs(result_dir, exist_ok=True)
            

            result_name = filename[:-4] + '_' + car_id + '.jpg'
            result_path = os.path.join(result_dir, result_name)    
            cv2.imwrite(result_path, cropped_img) 


def main():
    for sub_result in ["train_reID", "validation_reID"]:
        result_dir = os.path.join(data_root, sub_result)
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir, exist_ok=True)
    
    gt_paths = []
    for sub_data in ["train", "validation"]:
        data_dir = os.path.join(data_root, sub_data) 
        for scenario in os.listdir(data_dir):
            scenario_dir = os.path.join(data_dir, scenario)
            for camera in os.listdir(scenario_dir):
                gt_path = os.path.join(scenario_dir, camera, "gt/gt.txt")
                gt_paths.append(gt_path)
    
    pool = Pool(5)            
    pool.map(crop_car_image, gt_paths)
        
        
if __name__ == "__main__":
    main()
    