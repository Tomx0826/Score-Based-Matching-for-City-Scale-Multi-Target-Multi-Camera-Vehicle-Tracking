# encoding: utf-8
import os
import os.path as osp
import re
import glob
import xml.dom.minidom as XD

from .bases import BaseImageDataset


class AIC(BaseImageDataset):
    dataset_dir = 'REID/aic22_track_uda/make_dataset'

    def __init__(self, root='/Disk_New/AI_City_Challenge/TEST', verbose=True, crop_test=False, **kwargs):
        super(AIC, self).__init__()
        self.crop_test = crop_test
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train_reID2')
        self.vali_dir = osp.join(self.dataset_dir, 'validation_reID2')
        #self.query_dir = osp.join(self.dataset_dir, 'image_query')
        #self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=False)
        vali = self._process_dir(self.vali_dir, relabel=False)
        
        
        
        #query = self._process_dir_test(self.query_dir, relabel=False)
        #gallery = self._process_dir_test(self.gallery_dir, relabel=False, query=False)

        if verbose:
            print("=> AIC loaded")
            #self.print_dataset_statistics(train, query, gallery)            
            self.print_dataset_statistics(train, vali)            

            
        self.train = train
        self.vali = vali
        #self.query = query
        #self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.list_train_pids = self.get_imagedata_info(self.train)
        self.num_vali_pids, self.num_vali_imgs, self.num_vali_cams, self.list_vali_pids = self.get_imagedata_info(self.vali)
        #self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        #self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.vali_dir):
            raise RuntimeError("'{}' is not available".format(self.vali_dir))
        #if not osp.exists(self.query_dir):
        #    raise RuntimeError("'{}' is not available".format(self.query_dir))
        #if not osp.exists(self.gallery_dir):
        #    raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False, if_track=False):
        pid_container = sorted(os.listdir(dir_path), key=lambda f: int(f.split('.')[0]))
        ## pid2label={pID_x: new_pID_1, pID_y: new_pID_2, ...}
        pid2label = {int(pid_dir): label for label, pid_dir in enumerate(pid_container)}

        dataset = []
        for pid_dir in pid_container:
            pid = int(pid_dir)
            if relabel: pid = pid2label[int(pid_dir)]
            for img_path in glob.glob(osp.join(dir_path, pid_dir, '*.jpg')):
                camid = int(img_path.split('/')[-1].split('_')[1][1:])
                dataset.append((img_path, pid-1, camid, -1)) 

        return dataset

    def _process_dir_test(self, dir_path, relabel=False, query=True):        
        dataset = []
        for pid_dir in os.listdir(dir_path):
            for img_path in glob.glob(osp.join(dir_path, pid_dir, '*.jpg')):
                camid = int(img_path.split('/')[-1].split('_')[1][1:])   
                dataset.append((img_path, -1, camid, -1))

        return dataset

    def _process_track(self,path): #### Revised
        ## file format: track 1: img_i, img_j...
        ##              track 2: img_x, img_y...
        ##              ...
        file = open(path)
        ## tracklet={track 1:[img_i, img_j...], track 2:[img_i, img_j...], ...}
        tracklet = dict()
        ## frame2trackID={img_i:track 1, img_j:track 1, img_x:track 2, ...}
        frame2trackID = dict()
        nums = []
        for track_id, line in enumerate(file.readlines()):
            curLine = line.strip().split(" ")
            nums.append(len(curLine))
            #  curLine = list(map(eval, curLine))
            tracklet[track_id] = curLine
            for frame in curLine:
                frame2trackID[frame] = track_id
        return tracklet, nums, frame2trackID


class AIC_UDA(BaseImageDataset):
    dataset_dir = 'REID/aic22_track_uda/make_dataset/UDA' 

    def __init__(self, root='/Disk_New/AI_City_Challenge/TEST', verbose=True, crop_test=False, **kwargs):
        super(AIC_UDA, self).__init__()
        self.crop_test = crop_test
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()

        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> AIC_UDA loaded")
            #self.print_dataset_statistics(train, query, gallery)            
            self.print_uda_statistics(gallery)            

            
        self.gallery = gallery

        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.list_gallery_pids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        
        dataset = []
        for camera in os.listdir(dir_path):
            #limit = 0
            camid = int(camera.split('c')[1])
            for img_path in glob.glob(osp.join(dir_path, camera, '*.png')):
                img_file = img_path.split('/')[-1]      
                _, track_id, img_id = img_file[:-4].split('_')
                if (int(img_id) + 2) % 3 != 0: 
                    continue
                
                track_id = int(track_id)
                dataset.append((img_path, -1, camid, track_id))                
                
                #if limit == 119:
                #    break
                #limit += 1
                
        return dataset
         
        
if __name__ == '__main__':
    #aic = AIC(root='/home/data/AIC22_ZongYe')
    aic = AIC(root='/home/data')
