# Score-Based-Matching-for-City-Scale-Multi-Target-Multi-Camera-Vehicle-Tracking
These codes are modified from https://github.com/Yejin0111/AICITY2022-Track1-MTMC for main flow, https://github.com/LCFractal/AIC21-MTMC and https://github.com/michuanhaohao/AICITY2021_Track2_DMT for Re-ID/UDA training.  

## 1. DET
 ### Follow the same steps in https://github.com/Yejin0111/AICITY2022-Track1-MTMC:
   1. Data preparation:
    a. Put the data (1) DET/data/, (2) DET/pretrained/ and (3) DET/pth/ in the folder AICITY2022-MTMC/DET/Swin-Transformer-Object-Detection/  
    b. Put the dataset TOOLS/data/AIC/AIC21_Track3_MTMC_Tracking in the folder AICITY2022-MTMC/TOOLS/
   2. Run "python vid2img_recursive_std.py" in the folder AICITY2022-MTMC/TOOLS/
   3. Run "bash test.sh" in the folder AICITY2022-MTMC/DET/Swin-Transformer-Object-Detection/ to generate detection results test-det-2666.bbox.json  

## 2. REID
   ### Re-ID training in the folder aic22_track_uda:  
   - Stage 1: ReID training  
	**Data Generation**  
&nbsp; &nbsp; step 1: Execute build_reID_dataset.py to generate train_reID/ and validation_reID/  
&nbsp; &nbsp; step 2: Execute add_syn_dataset.py to generate train_reID2/ and validation_reID2/  
	**Then, run trainV6.py.**  

     
``` 
python trainV6.py --config_file=./configs_syn/stage1/resnext101a.yml    
```
   - Stage 2: UDA training  
	**Data Preparation**  
	&nbsp; &nbsp; Put data "UDA/Test/" under folder make_dataset for UDA training  
	**Revise the Code:**  
	&nbsp; &nbsp; Revise **./datasets/make_dataloaderV2.py** at line 19 and line 20  
		>line 19 | 'aic': AIC       --> #'aic': AIC  
		line 20 | #'aic': AIC_UDA  --> 'aic': AIC_UDA

    
		**Then, run train_stage2V3_v1.py.**  
```
python train_stage2V3_v1.py --config_file=./configs_syn/stage2/resnext101a.yml
``` 
### Re-ID feature extraction in the folder reid:  
&nbsp; &nbsp; &nbsp; &nbsp; Step 1: Put the testing data "crop_test_det_2666_89" in the folder reid/datasets/  
&nbsp; &nbsp; &nbsp; &nbsp; Step 2: Put the trained weights "resnext101_ibn_a_2.pth" in the folder reid/reid_model/focal/rxt_uda/  
&nbsp; &nbsp; &nbsp; &nbsp; Step 3: Run python extract_image_feat_resnext101.py "aic22_resnext_101_uda_focal.yml" to generate ReID features (dets_feat.pkl)  
&nbsp; &nbsp; &nbsp; &nbsp; Step 4: Run read_dict.py to generate aic22_uda.pkl  
