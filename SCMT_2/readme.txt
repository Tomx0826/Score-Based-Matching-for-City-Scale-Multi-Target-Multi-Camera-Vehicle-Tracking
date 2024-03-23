
前置作業：
  1. aic22_uda.pkl放進./dataspace/AICITY_test_pca/
  2. under dir "AICITY_test_pca" --> run python gen_detection_feat.py
  3. under dir "AICITY_test_pca" --> run python pca.py

under dir "SCMT_2"  
python run_aicity_pca.py AICITY test_pca --dir_save scmt 
python stat_occlusion_scmt2.py scmt
