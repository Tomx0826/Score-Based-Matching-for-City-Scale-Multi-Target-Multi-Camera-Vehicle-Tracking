# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
#from scipy.optimize import linear_sum_assignment as linear_assignment 
from . import kalman_filter
from opts import opt

INFTY_COST = 1e+5


def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """

    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.
    
    # Cosine distance with maha threshold
    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
        
    # Cosine distance with cosine threshold
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    #print(np.max(cost_matrix))
    cost_matrix_ = cost_matrix.copy()

    indices = linear_assignment(cost_matrix_)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            #print(cost_matrix_[row,col])
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    # 1. confirmed_tracks, high_score_det_indices
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    assert opt.woC == True
    if opt.woC:  ## opt.woC == True
        track_indices_l = [
            k for k in track_indices
            # if tracks[k].time_since_update == 1 + level
        ]
        # 做完cosine distance + mask with maha dist. + mask with cosine dist. + linear assignment
        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        matches += matches_l
    else:
        for level in range(cascade_depth):
            if len(unmatched_detections) == 0:  # No detections left
                break

            track_indices_l = [
                k for k in track_indices
                if tracks[k].time_since_update == 1 + level 
            ]
            if len(track_indices_l) == 0:  # Nothing to match at this level
                continue

            matches_l, _, unmatched_detections = \
                min_cost_matching(
                    distance_metric, max_distance, tracks, detections,
                    track_indices_l, unmatched_detections)
            matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections

def gate_cost_matrix(
        cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False, gating_threshold=50.0, add_identity=True):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    assert not only_position
    # gating_threshold = kalman_filter.chi2inv95[4]
    gating_threshold = gating_threshold
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        # 馬氏距離平方(預測值 VS detection box)
        gating_distance = track.kf.gating_distance(track.mean, track.covariance, measurements, only_position, add_identity)

        # 43 range 1000~20000
        # 41 range 600~2000
        #print("gd",gating_distance)

        # 第row個track predtiction box VS detection boxes
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
        
        # 原cost_matrix只有appearance距離
        #print(cost_matrix[row])
        #print(gating_distance)
        #print('\n')
        if opt.MC:
            cost_matrix[row] = opt.MC_lambda * cost_matrix[row] + (1 - opt.MC_lambda) *  gating_distance

        #bbox = tracks[track_idx].to_tlwh()
        #candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        #max_w = 1.3*bbox[2]
        #max_h = 1.3*bbox[3]
        #min_w = 0.7*bbox[2]
        #min_h = 0.7*bbox[3]
        #for i in range(len(detection_indices)):
        #    if candidates[i,2] > max_w or candidates[i,3] > max_h:
        #        cost_matrix[row,i]=gated_cost
        #    if candidates[i,2] < min_w or candidates[i,3] < min_h:
        #        cost_matrix[row,i]=gated_cost


    return cost_matrix