# Score-Based-Matching-for-City-Scale-Multi-Target-Multi-Camera-Vehicle-Tracking
These codes are modified from https://github.com/Yejin0111/AICITY2022-Track1-MTMC for main flow, https://github.com/LCFractal/AIC21-MTMC and https://github.com/michuanhaohao/AICITY2021_Track2_DMT for Re-ID and UDA training.  
## DATASET
The CityFlowV2 dataset needs to be applied for from [AiCityChallenge](<https://www.aicitychallenge.org/2022-challenge-tracks/>).


## DET
 ### Follow the same steps of DET [here](<https://github.com/Yejin0111/AICITY2022-Track1-MTMC>):
   1. Data preparation:  
    a. Put the data (1) DET/data/, (2) DET/pretrained/ and (3) DET/pth/ in the folder AICITY2022-MTMC/DET/Swin-Transformer-Object-Detection/  
    b. Put the dataset TOOLS/data/AIC/AIC21_Track3_MTMC_Tracking in the folder AICITY2022-MTMC/TOOLS/
   2. Run "python vid2img_recursive_std.py" in the folder AICITY2022-MTMC/TOOLS/
   3. Run "bash test.sh" in the folder AICITY2022-MTMC/DET/Swin-Transformer-Object-Detection/ to generate detection results test-det-2666.bbox.json  

## REID
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
	&nbsp; &nbsp; (To follow the flow of [UDA training](<https://github.com/michuanhaohao/AICITY2021_Track2_DMT>), we adopt [ByteTrack](<https://github.com/ifzhang/ByteTrack>) with CityFlowV2 to generate UDA/Test/)  
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
&nbsp; &nbsp; &nbsp; &nbsp; Step 2: Put the trained weights or our trained weights, "resnext101_ibn_a_2.pth," in the folder reid/reid_model/focal/rxt_uda/  
&nbsp; &nbsp; &nbsp; &nbsp; Step 3: Run python extract_image_feat_resnext101.py "aic22_resnext_101_uda_focal.yml" to generate ReID features (dets_feat.pkl)  
&nbsp; &nbsp; &nbsp; &nbsp; Step 4: Run read_dict.py to generate aic22_uda.pkl  

## SCMT_2
&nbsp; &nbsp; **Data preparation:**  
&nbsp; &nbsp; &nbsp; &nbsp; Put the data "CityFlowV2" in the folder SCMT_2/dataset/  
&nbsp; &nbsp; **Generate PCA features:**  
&nbsp; &nbsp; &nbsp; &nbsp; 1. Put aic22_uda.pkl in the folder ./dataspace/AICITY_test_pca/  
&nbsp; &nbsp; &nbsp; &nbsp; 2. under dir "AICITY_test_pca" --> run python gen_detection_feat.py  
&nbsp; &nbsp; &nbsp; &nbsp; 3. under dir "AICITY_test_pca" --> run python pca.py  
&nbsp; &nbsp; **Run SCMT**  
&nbsp; &nbsp; &nbsp; &nbsp; 1. Run "python run_aicity_pca.py AICITY test_pca --dir_save scmt" to get the SCMT results.  
&nbsp; &nbsp; &nbsp; &nbsp; 2. Run "python stat_occlusion_scmt2.py scmt" to generate the dictionary for occlusion objects. 

## ICA_2  
&nbsp; &nbsp; &nbsp; &nbsp; 1. Put the results of SCMT (the folder scmt) in the folder /ICA_2/data/  
&nbsp; &nbsp; &nbsp; &nbsp; 2. Run "bash run.sh scmt"  
&nbsp; &nbsp; &nbsp; &nbsp; 3. The final MTMC tracking results are in the file "submit_result_expand_1.3_truncation_filter_roi_result.txt" 

Our Result can be download here: [track1.txt](<https://drive.google.com/file/d/1tpGhTlV8YqP_4oihVO1H3Fm9PnN70qiv/view?usp=sharing>)
