# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
import copy

import cv2
import numpy as np
import pickle

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from opts import opt


def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)

    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir), #c041
        "image_filenames": image_filenames, # ./dataset/CityFlowV2/AICITY/test/c041/img1/0001.jpg~2001.jpg
        "detections": detections, # [fid, -1, x, y, w, h, conf, -1, -1, -1] + feat
        "groundtruth": groundtruth,
        "image_size": image_size, # (960, 1280), 45 46:(720, 1280)
        "min_frame_idx": min_frame_idx, # 1
        "max_frame_idx": max_frame_idx, # 2001
        "feature_dim": feature_dim, # 2048
        "update_ms": update_ms #100
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0, frame_img=None):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        color_hist = None
        if frame_img is not None:
            color_hist = []
            H, W, _ = frame_img.shape
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            w = int(bbox[2])
            h = int(bbox[3])
            x2 = x1 + w
            y2 = y1 + h
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W - 1, x2)
            y2 = min(H - 1, y2)
            for i in range(3):
                color_hist += cv2.calcHist([frame_img[y1 : y2, x1 : x2]], [i], None, [8], [0.0,255.0]).T.tolist()[0]
            color_hist = np.array(color_hist)
            norm = np.linalg.norm(color_hist)
            color_hist /= norm
            # feature = np.hstack((feature, color_hist))
            # print('feat', feature.shape, feature.dtype)
        detection_list.append(Detection(bbox, confidence, feature, frame_idx, color_hist=color_hist))
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    sequence = sequence_dir.split("/")[-1]

    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine', # self._metric = _nn_cosine_distance
        max_cosine_distance, # =0.45, Gating threshold for cosine distance "metric (object appearance)."
        nn_budget # =1, Maximum size of the appearance descriptors
    )
    
    roi_mask = cv2.imread('./track_roi/roi_{}.png'.format(sequence))
    tracker = Tracker(metric, cam_name=sequence, mask=roi_mask, image_filenames=seq_info['image_filenames'])
    results = []
    

    def frame_callback(vis, frame_idx):
        # Load image and generate detections.
        img = cv2.imread(seq_info["image_filenames"][frame_idx])
        detections = create_detections( 
            seq_info["detections"], frame_idx, min_detection_height, frame_img=img)
        # create_detections
        # return detection_list.append(Detection(bbox, confidence, feature, frame_idx, color_hist=color_hist))
        
        ## c042不mask ----> Q1
        if sequence != 'c042':
            detections = [d for d in detections 
                if d.confidence >= min_confidence and 
                (roi_mask[int(d.tlwh[1] + d.tlwh[3] / 2), int(d.tlwh[0] + d.tlwh[2] / 2), 0] / 255 == 1)]
        else:
            detections = [d for d in detections 
                if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores) # nms_max_overlap = 1.0 (完全重疊)
        detections = [detections[i] for i in indices]
        # Update tracker.
        tracker.predict() 
        # 第一步 predict:
        # kf 預測 --> self.mean, self.covariance // 更新mean, cov
        #             self.age +1 // 
        #             self.time_since_update --> track update後歸零
        # 第二步 matching
        tracker.update(detections)

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

    def batch_callback():
        tracker.postprocess()
        # pass

    # Run tracker.
    if display:
        # visualizer = visualization.Visualization(seq_info, update_ms=5)
        visualizer = visualization.NoVisualization(seq_info)
    else:
        visualizer = visualization.NoVisualization(seq_info)

    if sequence in ['c042','c044']:
        visualizer.run_reverse(frame_callback, batch_callback)
    else:
        visualizer.run(frame_callback, batch_callback)
        # frame_callback --> 抓某張frame_idx下的所有偵測框
        # visualizer.run --> 從1~>2001 Loop執行frame_callback
        
        

    if sequence in ['c042']:
        res1 = copy.deepcopy(tracker.tracks_all)
        tracker = Tracker(metric, cam_name=sequence, mask=roi_mask, image_filenames=seq_info['image_filenames'])
        visualizer.run(frame_callback, batch_callback)
        for track1 in tracker.tracks_all:
            if track1.storage[0].tlwh[0] > 600 and track1.storage[-1].tlwh[0] < 650 and track1.storage[0].tlwh[0] > track1.storage[-1].tlwh[0]:
                track1_frames = [d.frame_idx for d in track1.storage]
                for track2 in res1:
                    if track2.storage[0].tlwh[0] > 600 and track2.storage[-1].tlwh[0] < 650 and track2.storage[0].tlwh[0] > track1.storage[-1].tlwh[0]:
                        track2_frames = [d.frame_idx for d in track2.storage]
                        same_frame = set(track1_frames) & set(track2_frames)
                        if len(same_frame) > 7:
                            same_count = 0
                            for fid in same_frame:
                                det1 = [d for d in track1.storage if d.frame_idx == fid]
                                det2 = [d for d in track2.storage if d.frame_idx == fid]
                                if tracker._det_iou(det1[0], det2[0]) > 0.95:
                                    same_count += 1
                            if same_count > 5:
                                for det2 in track2.storage:
                                    if det2.frame_idx < track1.storage[0].frame_idx:
                                        track1.storage.append(det2) 
    if sequence in ['c044']:
        res1 = copy.deepcopy(tracker.tracks_all)
        tracker = Tracker(metric, cam_name=sequence, mask=roi_mask, image_filenames=seq_info['image_filenames'])
        visualizer.run(frame_callback, batch_callback)
        for track1 in tracker.tracks_all:
            if track1.storage[0].tlwh[0] > 570 and track1.storage[-1].tlwh[0] < 570:
                track1_frames = [d.frame_idx for d in track1.storage]
                for track2 in res1:
                    if track2.storage[0].tlwh[0] > 570 and track2.storage[-1].tlwh[0] < 570:
                        track2_frames = [d.frame_idx for d in track2.storage]
                        same_frame = set(track1_frames) & set(track2_frames)
                        if len(same_frame) > 6:
                            same_count = 0
                            for fid in same_frame:
                                det1 = [d for d in track1.storage if d.frame_idx == fid]
                                det2 = [d for d in track2.storage if d.frame_idx == fid]
                                if tracker._det_iou(det1[0], det2[0]) > 0.95:
                                    same_count += 1
                            if same_count > 5:
                                for det2 in track2.storage:
                                    if det2.frame_idx < track1.storage[0].frame_idx:
                                        track1.storage.append(det2)
                                track1.storage.sort(key=lambda d : d.frame_idx)

    dict_cam={}
    # Store results.
    for track in tracker.tracks_all:
        for det in track.storage:
            bbox = det.tlwh
            if sequence == 'c042' and (roi_mask[int(bbox[1] + bbox[3] / 2), int(bbox[0] + bbox[2] / 2), 0] / 255 != 1):
                continue
            frame_idx = det.frame_idx
            key = '{}_{}_{}'.format(sequence, frame_idx, track.track_id)
            if key in dict_cam:
                continue
            dict_cam[key] = np.hstack((det.feature, det.color_hist))
            results.append([
                    frame_idx, track.track_id, round(bbox[0]), round(bbox[1]), round(bbox[2]), round(bbox[3])])
                 
                    
             
    with open(output_file, 'w') as f:
        for row in sorted(results):
            print('%d,%d,%d,%d,%d,%d,1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    with open(output_file.replace('.txt', '.pkl'), 'wb') as fid:
        pickle.dump(dict_cam, fid, protocol=2)

def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)
