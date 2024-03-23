# Score-Based-Matching-for-City-Scale-Multi-Target-Multi-Camera-Vehicle-Tracking

## DET
 **Follow the same steps in https://github.com/Yejin0111/AICITY2022-Track1-MTMC:**  
   1. Data preparation:  
    a. Put the data (1) DET/data/, (2) DET/pretrained/ and (3) DET/pth/ in the folder AICITY2022-MTMC/DET/Swin-Transformer-Object-Detection/
    b. Put the dataset TOOLS/data/AIC/AIC21_Track3_MTMC_Tracking in the folder AICITY2022-MTMC/TOOLS/
   2. Run "python vid2img_recursive_std.py" in the folder AICITY2022-MTMC/TOOLS/
   3. Run "CUDA_VISIBLE_DEVICES=0 bash test.sh" in the folder AICITY2022-MTMC/DET/Swin-Transformer-Object-Detection/ to generate detection results test-det-2666.bbox.json  

## REID
   **Re-ID training in the folder aic22_track_uda:**   
   - Stage 1: CityFlow dataset + VehicleX dataset  
      Data Generation:  
        >Step 1: Execute build_reID_dataset.py to generate train_reID/ and validation_reID/  
        >Step 2: Execute add_syn_dataset.py to generate train_reID2/ and validation_reID2/  

      Then, run trainV6.py.  
	``` 
	python trainV6.py --config_file=./configs_syn/stage1/resnext101a.yml    
	```
   - Stage 2: UDA training   
	Revise the Code: ./datasets/make_dataloaderV2.py, at line 19 and line 20
		>line 19 | 'aic': AIC       --> #'aic': AIC    
   		>line 20 | #'aic': AIC_UDA  --> 'aic': AIC_UDA
   
