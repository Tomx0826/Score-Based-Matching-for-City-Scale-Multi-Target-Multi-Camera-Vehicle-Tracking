import os


data_root = "/runtmp/Data/AIC22_Track1_MTMC_Tracking/"

for sub_data in ["train", "validation"]:
    data_dir = os.path.join(data_root, sub_data) 
    for scenario in os.listdir(data_dir):
        scenario_dir = os.path.join(data_dir, scenario)
        for camera in os.listdir(scenario_dir):
            video_path = os.path.join(scenario_dir, camera, "vdo.avi")
            
            result_dir = os.path.join(data_dir+'_frame', scenario, camera)
            if not os.path.isdir(result_dir):
                os.makedirs(result_dir, exist_ok=True)
            filename = scenario + "_" + camera + "_" + "%4d.jpg"
            result_path = os.path.join(result_dir, filename)
            
            FPS = "10"
            if scenario=='S03' and camera=='c015':
                FPS = "8"
            
            CMD = "ffmpeg -i " + video_path + " -vf fps=" + FPS + \
                  " " + result_path
            print(CMD)
            #os.system(CMD)
