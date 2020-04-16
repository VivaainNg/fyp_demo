from skimage.measure import label
from skimage.morphology import dilation


import os
import matlab
import json
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import random
from PIL import Image
from scipy.io import loadmat
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.signal import medfilt


import pdb

################ Config ##########################


def load_config_file(config_file):
    '''
    -- Doc for parameters in the json file --

    feature_oversample:   Whether data augmentation is used (five crop and filp).
    sample_rate:          How many frames between adjacent feature snippet.

    with_bg:              Whether hard negative mining is used.
    diversity_reg:        Whether diversity loss and norm regularization are used.
    diversity_weight:     The weight of both diversity loss and norm regularization.

    train_run_num:        How many times the experiment is repeated.
    training_max_len:     Crop the feature sequence when training if it exceeds this length.

    learning_rate_decay:  Whether to reduce the learning rate at half of training steps.
    max_step_num:         Number of training steps.
    check_points:         Check points to test and save models.
    log_freq:             How many training steps the log is added to tensorboard.

    model_params:
    cls_branch_num:       Branch number in the multibranch network.
    base_layer_params:    Filter number and size in each layer of the embedding module.
    cls_layer_params:     Filter number and size in each layer of the classification module.
    att_layer_params:     Filter number and size in each layer of the attention module.

    detect_params:        Parameters for action localization on the CAS. 
                          See detect.py for details.

    base_sample_rate:     'sample_rate' when feature extraction.
    base_snippet_size:    The size of each feature snippet.

    bg_mask_dir:          The folder of masks of static clips.

    < Others are easy to guess >

    '''

    all_params = json.load(open(config_file))

    dataset_name = all_params['dataset_name']
    feature_type = all_params['feature_type']

    all_params['file_paths'] = all_params['file_paths'][dataset_name]
    all_params['action_class_num'] = all_params['action_class_num'][
        dataset_name]
    all_params['base_sample_rate'] = all_params['base_sample_rate'][
        dataset_name][feature_type]
    all_params['base_snippet_size'] = all_params['base_snippet_size'][
        feature_type]

    assert (all_params['sample_rate'] % all_params['base_sample_rate'] == 0)

    all_params['model_class_num'] = all_params['action_class_num']
    if all_params['with_bg']:
        all_params['model_class_num'] += 1

    all_params['model_params']['class_num'] = all_params['model_class_num']

    # Convert second to frames
    all_params['detect_params']['proc_value'] = int(
        all_params['detect_params']['proc_value'] * all_params['sample_rate'])

    print(all_params)
    return all_params


################ Class Name Mapping #####################

ucf_crime_old_cls_names = {
    1: 'Abuse',
    2: 'Arrest',
    3: 'Arson',
    4: 'Assault',
    5: 'Burglary',
    6: 'Explosion',
    7: 'Fighting',
    8: 'RoadAccidents',
    9: 'Robbery',
    10: 'Shooting',
    11: 'Shoplifting',
    12: 'Stealing',
    13: 'Vandalism'
}

ucf_crime_old_cls_indices = {v: k for k, v in ucf_crime_old_cls_names.items()}

ucf_crime_new_cls_names = {
    0: 'Abuse',
    1: 'Arrest',
    2: 'Arson',
    3: 'Assault',
    4: 'Burglary',
    5: 'Explosion',
    6: 'Fighting',
    7: 'RoadAccidents',
    8: 'Robbery',
    9: 'Shooting',
    10: 'Shoplifting',
    11: 'Stealing',
    12: 'Vandalism',
    13: 'Background'
}

ucf_crime_new_cls_indices = {v: k for k, v in ucf_crime_new_cls_names.items()}

old_cls_names = {
    'ucf_crime': ucf_crime_old_cls_names,
}

old_cls_indices = {
    'ucf_crime': ucf_crime_old_cls_indices,
}

new_cls_names = {
    'ucf_crime': ucf_crime_new_cls_names,
}

new_cls_indices = {
    'ucf_crime': ucf_crime_new_cls_indices,
}

################ Load dataset #####################

def load_annotation_file(anno_file):
    '''Load action instaces from a single file (Only for ucf_crime).'''
    anno_data = pd.read_csv(anno_file, header=None, delimiter=' ')
    anno_data = np.array(anno_data)
    return anno_data


def __get_ucf_crime_anno(anno_dir):
    dataset_dict = {}
    anno_file = os.listdir(anno_dir)[0]
    anno_file_pth = os.path.join(anno_dir, anno_file)
    anno_data = load_annotation_file(anno_file_pth)
    
    for entry in anno_data:
        video_name = entry[0].split('.')[0]
        #TODO:Run 2nd variation experiment
        if video_name.startswith('Normal'):
            continue

        #action_label points to Abuse, Arrest, Arson....
        action_label = entry[2] 
        #points to indices of Abuse, Arrest,...
        action_label = new_cls_indices['ucf_crime'][action_label]  
        
        if video_name not in dataset_dict.keys():
            dataset_dict[video_name] = {
                    #'duration': duration,
                    'frame_rate': 30,
                    'labels': [],
                    'annotations': {},
            }
            
        if action_label not in dataset_dict[video_name]['labels']:
            dataset_dict[video_name]['labels'].append(action_label)
            dataset_dict[video_name]['annotations'][action_label] = []
        # Track temporal annotations (only for test set)    
        if anno_file.split('_')[1].startswith('test'):
            # Frame number to seconds
            start1 = np.round(entry[4]/30, 1)
            end1 = np.round(entry[6]/30, 1)
            start2 = np.round(entry[8]/30, 1)
            end2 = np.round(entry[10]/30, 1)
        
            if start1 == -.0:
                continue
                #dataset_dict[video_name]['annotations'][action_label].append([])
            elif start2 == -.0 and start1 != -.0:
                dataset_dict[video_name]['annotations'][action_label].append([start1, end1])
            else:
                dataset_dict[video_name]['annotations'][action_label].append([start1, end1]) 
                dataset_dict[video_name]['annotations'][action_label].append([start2, end2])
    return dataset_dict      
    

def __load_features(
        dataset_dict,  # dataset_dict will be modified
        dataset_name,
        feature_type,
        sample_rate,
        base_sample_rate,
        temporal_aug,
        rgb_feature_dir,
        flow_feature_dir):

    assert (feature_type in ['i3d']) #i3d, untri

    assert (sample_rate % base_sample_rate == 0)
    f_sample_rate = int(sample_rate / base_sample_rate)

    # sample_rate of feature sequences, not original video

    ###############
    def __process_feature_file(filename):
        ''' Load features from a single file. '''

        feature_data = np.load(filename)

        frame_cnt = feature_data['frame_cnt'].item()

        feature = feature_data['feature']

        # Feature: (B, T, F)
        # Example: (1, 249, 1024) or (10, 249, 1024) (Oversample)

        if temporal_aug:  # Data augmentation with temporal offsets
            feature = [
                feature[:, offset::f_sample_rate, :]
                for offset in range(f_sample_rate)
            ]
            # Cut to same length, OK when training
            min_len = int(min([i.shape[1] for i in feature]))
            feature = [i[:, :min_len, :] for i in feature]

            assert (len(set([i.shape[1] for i in feature])) == 1)
            feature = np.concatenate(feature, axis=0)

        else:
            feature = feature[:, ::f_sample_rate, :]

        return feature, frame_cnt

        # Feature: (B x f_sample_rate, T, F)

    ###############

    # Load all features
    for k in dataset_dict.keys():
        #TODO:Run 2nd variation experiment
        if k.startswith('Normal'):
            continue

        print('Loading: {}'.format(k))

        # Init empty
        dataset_dict[k]['frame_cnt'] = -1
        dataset_dict[k]['rgb_feature'] = -1
        dataset_dict[k]['flow_feature'] = -1

        if rgb_feature_dir:

            if dataset_name == 'ucf_crime':
                rgb_feature_file = os.path.join(rgb_feature_dir, k + '-rgb.npz')
            else:
                rgb_feature_file = os.path.join(rgb_feature_dir,
                                                'v_' + k + '-rgb.npz')

            rgb_feature, rgb_frame_cnt = __process_feature_file(
                rgb_feature_file)

            dataset_dict[k]['frame_cnt'] = rgb_frame_cnt
            dataset_dict[k]['rgb_feature'] = rgb_feature

        if flow_feature_dir:

            if dataset_name == 'ucf_crime':
                flow_feature_file = os.path.join(flow_feature_dir,
                                                 k + '-flow.npz')
            else:
                flow_feature_file = os.path.join(flow_feature_dir,
                                                 'v_' + k + '-flow.npz')

            flow_feature, flow_frame_cnt = __process_feature_file(
                flow_feature_file)

            dataset_dict[k]['frame_cnt'] = flow_frame_cnt
            dataset_dict[k]['flow_feature'] = flow_feature

        if rgb_feature_dir and flow_feature_dir:
            assert (rgb_frame_cnt == flow_frame_cnt)
            assert (dataset_dict[k]['rgb_feature'].shape[1] == dataset_dict[k]
                    ['flow_feature'].shape[1])
            assert (dataset_dict[k]['rgb_feature'].mean() !=
                    dataset_dict[k]['flow_feature'].mean())

    return dataset_dict


def __load_background(
        dataset_dict,  # dataset_dict will be modified
        dataset_name,
        bg_mask_dir,
        sample_rate,
        action_class_num):

    bg_mask_files = os.listdir(bg_mask_dir)
    bg_mask_files.sort()

    # Select only normal files
    normal_files = [i for i in bg_mask_files if i.startswith('Normal')]
    # Random sample (rate in terms of %) of normal files as hard negative
    norm_sample_rate = .5
    random_sampled_norm = random.sample(normal_files, int(len(normal_files) * norm_sample_rate))
    # Select only abnormal files
    abnormal_files = [i for i in bg_mask_files if not i.startswith('Normal')]
    bg_mask_files = abnormal_files + random_sampled_norm
    
    for bg_mask_file in bg_mask_files:

        video_name = bg_mask_file[:-4]

        new_key = video_name + '_bg'

        if video_name not in dataset_dict.keys():
            continue

        bg_mask = np.load(os.path.join(bg_mask_dir, bg_mask_file))
        bg_mask = bg_mask['mask']

        assert (dataset_dict[video_name]['frame_cnt'] == bg_mask.shape[0])

        # Remove if static clips are too long or too short
        bg_ratio = bg_mask.sum() / bg_mask.shape[0]
        if bg_ratio < 0.05 or bg_ratio > 0.30:
            print('Bad bg {}: {}'.format(bg_ratio, video_name))
            continue

        bg_mask = bg_mask[::sample_rate]  # sample rate of original videos

        dataset_dict[new_key] = {}

        if type(dataset_dict[video_name]['rgb_feature']) != int:

            rgb = np.array(dataset_dict[video_name]['rgb_feature'])
            bg_mask = bg_mask[:rgb.shape[1]]  # same length
            bg_rgb = rgb[:, bg_mask.astype(bool), :]
            dataset_dict[new_key]['rgb_feature'] = bg_rgb

            frame_cnt = bg_rgb.shape[
                1]  # Pseudo frame count of a virtual bg video

        if type(dataset_dict[video_name]['flow_feature']) != int:

            flow = np.array(dataset_dict[video_name]['flow_feature'])
            bg_mask = bg_mask[:flow.shape[1]]
            bg_flow = flow[:, bg_mask.astype(bool), :]
            dataset_dict[new_key]['flow_feature'] = bg_flow

            frame_cnt = bg_flow.shape[
                1]  # Pseudo frame count of a virtual bg video

        dataset_dict[new_key]['annotations'] = {action_class_num: []}
        dataset_dict[new_key]['labels'] = [action_class_num]  # background class

        fps = dataset_dict[video_name]['frame_rate']
        dataset_dict[new_key]['frame_rate'] = fps
        dataset_dict[new_key]['frame_cnt'] = frame_cnt  # Pseudo
        dataset_dict[new_key]['duration'] = frame_cnt / fps  # Pseudo

    return dataset_dict


def get_dataset(dataset_name,
                subset,
                file_paths,
                sample_rate,
                base_sample_rate,
                action_class_num,
                modality='both',
                feature_type=None,
                feature_oversample=True,
                temporal_aug=False,
                load_background=False):

    assert (dataset_name in ['ucf_crime']) #thumos14, ActivityNet...

    if load_background:
        assert (subset in ['val'])
    else:
        assert (subset in ['val', 'test'])

    assert (modality in ['both', 'rgb', 'flow', None])
    assert (feature_type in ['i3d']) #i3d, untri

    dataset_dict = __get_ucf_crime_anno(anno_dir=file_paths[subset]['anno_dir'])

    _temp_f_type = (feature_type +
                    '-oversample' if feature_oversample else feature_type +
                    '-resize')

    if modality == 'both':
        rgb_dir = file_paths[subset]['feature_dir'][_temp_f_type]['rgb']
        flow_dir = file_paths[subset]['feature_dir'][_temp_f_type]['flow']
    elif modality == 'rgb':
        rgb_dir = file_paths[subset]['feature_dir'][_temp_f_type]['rgb']
        flow_dir = None
    elif modality == 'flow':
        rgb_dir = None
        flow_dir = file_paths[subset]['feature_dir'][_temp_f_type]['flow']
    else:
        rgb_dir = None
        flow_dir = None

    dataset_dict = __load_features(dataset_dict, dataset_name, feature_type,
                                   sample_rate, base_sample_rate, temporal_aug,
                                   rgb_dir, flow_dir)

    if load_background:
        dataset_dict = __load_background(dataset_dict, dataset_name,
                                         file_paths[subset]['bg_mask_dir'],
                                         sample_rate, action_class_num)

    return dataset_dict


def get_single_label_dict(dataset_dict):
    '''
    If a video has multiple action classes, we treat it as multiple videos with
    single class. And the weight of each of them is reduced.
    '''
    new_dict = {}  # Create a new dict

    for k, v in dataset_dict.items():
        for label in v['labels']:
            
            new_key = '{}-{}'.format(k, label)

            new_dict[new_key] = dict(v)
            #label_single will contain the class of activities: Abuse, Arrest....
            new_dict[new_key]['label_single'] = label
            new_dict[new_key]['annotations'] = v['annotations'][label]
            new_dict[new_key]['weight'] = (1 / len(v['labels']))

            new_dict[new_key]['old_key'] = k

    return new_dict  # This dict should be read only


def get_videos_each_class(dataset_dict):

    videos_each_class = defaultdict(list)

    for k, v in dataset_dict.items():

        if 'label_single' in v.keys():
            label = v['label_single']
            videos_each_class[label].append(k)

        else:
            for label in v['labels']:
                videos_each_class[label].append(k)

    return videos_each_class


################ Post-Processing #####################


def normalize(x):
    x -= x.min()
    x /= x.max()
    return x


def smooth(x):  # Two Dim nparray, On 1st dim
    temp = np.array(x)

    temp[1:, :] = temp[1:, :] + x[:-1, :]
    temp[:-1, :] = temp[:-1, :] + x[1:, :]

    temp[1:-1, :] /= 3
    temp[0, :] /= 2
    temp[-1, :] /= 2

    return temp


def __get_frame_ticks(feature_type, frame_cnt, sample_rate, snippet_size=None):
    '''Get the frames of each feature snippet location.'''

    assert (feature_type in ['i3d']) #i3d, untri
    assert (snippet_size is not None)

    clipped_length = frame_cnt - snippet_size
    clipped_length = (clipped_length // sample_rate) * sample_rate
    # the start of the last chunk

    frame_ticks = np.arange(0, clipped_length + 1, sample_rate)
    # From 0, the start of chunks, clipped_length included

    return frame_ticks


def interpolate(x,
                feature_type,
                frame_cnt,
                sample_rate,
                snippet_size=None,
                kind='linear'):
    '''Upsample the sequence the original video fps.'''

    frame_ticks = __get_frame_ticks(feature_type, frame_cnt, sample_rate,
                                    snippet_size)

    full_ticks = np.arange(frame_ticks[0], frame_ticks[-1] + 1)
    # frame_ticks[-1] included

    interp_func = interp1d(frame_ticks, x, kind=kind)
    out = interp_func(full_ticks)

    return out

################ Action Localization #####################


def detections_to_mask(length, detections):

    mask = np.zeros((length, 1))
    for entry in detections:
        mask[entry[0]:entry[1]] = 1

    return mask


def mask_to_detections(mask, metric, weight_inner, weight_outter):

    out_detections = []
    detection_map = label(mask, background=0)
    detection_num = detection_map.max()

    for detection_id in range(1, detection_num + 1):

        start = np.where(detection_map == detection_id)[0].min()
        end = np.where(detection_map == detection_id)[0].max() + 1

        length = end - start

        inner_area = metric[detection_map == detection_id]

        left_start = min(int(start - length * 0.25),
                         start - 1)  # Context size 0.25
        right_end = max(int(end + length * 0.25), end + 1)

        outter_area_left = metric[left_start:start, :]
        outter_area_right = metric[end:right_end, :]

        outter_area = np.concatenate((outter_area_left, outter_area_right),
                                     axis=0)

        if outter_area.shape[0] == 0:
            detection_score = inner_area.mean() * weight_inner
        else:
            detection_score = (inner_area.mean() * weight_inner +
                               outter_area.mean() * weight_outter)

        out_detections.append([start, end, None, detection_score])

    return out_detections


def detect_with_thresholding(metric,
                             thrh_type,
                             thrh_value,
                             proc_type,
                             proc_value,
                             debug_file=None):

    assert (thrh_type in ['max', 'mean'])
    assert (proc_type in ['dilation', 'median'])

    out_detections = []

    if thrh_type == 'max':
        mask = metric > thrh_value

    elif thrh_type == 'mean':
        mask = metric > (thrh_value * metric.mean())

    if proc_type == 'dilation':
        mask = dilation(mask, np.array([[1] for _ in range(proc_value)]))
    elif proc_type == 'median':
        mask = medfilt(mask[:, 0], kernel_size=proc_value)
        # kernel_size should be odd
        mask = np.expand_dims(mask, axis=1)

    return mask


################ Output Detection To Files ################


def output_detections_ucf_crime(out_detections, out_file_name):

    for entry in out_detections:
        class_id = entry[3]
        class_name = new_cls_names['ucf_crime'][class_id]
        old_class_id = int(old_cls_indices['ucf_crime'][class_name])
        entry[3] = old_class_id

    out_file = open(out_file_name, 'w')

    for entry in out_detections:
        out_file.write('{} {:.2f} {:.2f} {} {:.4f}\n'.format(
            entry[0], entry[1], entry[2], int(entry[3]), entry[4]))

    out_file.close()

################ Visualization #####################

def prepare_gt(gtpth):
    gt_list = []
    fps = 30
    f = open(gtpth, 'r')
    for line in f.readlines():
        line2 = []
        line = line.replace('.mp4', '')
        line = line.split('  ')
        # Skip Normal videos
        if line[0].startswith('Normal'):
            continue
        gt_list.append(line)
    
    df = pd.DataFrame(gt_list)
    df.columns = ['videoname', 'cls', 'start1', 'end1', 'start2', 'end2', '_']
    df = df.drop(columns=['_'])
    
    # Every row of ground-truth annotations consist of only [videoname, cls, start, end]
    gtdf = df.loc[df['start2'] != '-1']
    gtdf_col = ['videoname', 'cls', 'start2', 'end2', 'start2', 'end2']
    gtdf = gtdf.reindex(columns=gtdf_col)
    gtdf.columns = df.columns
    gtdf = df.append(gtdf)
    gtdf = gtdf.sort_values(by=['videoname', 'start1'])
    gtdf = gtdf.rename(columns = {'start1': 'start', 'end1': 'end'})
    gtdf = gtdf.drop(columns=['start2', 'end2'])
    gtdf['start'] = gtdf['start'].apply(lambda x: np.round(int(x)/fps, decimals=2))
    gtdf['end'] = gtdf['end'].apply(lambda x: np.round(int(x)/fps, decimals=2))
    gtdf['cls'] = gtdf['cls'].apply(lambda x: ucf_crime_old_cls_indices[x])
    f.close()
    return gtdf


def prepare_detections(detlist): 
    columns = ['videoname', 'start', 'end', 'cls', 'conf'] 
    if detlist == []: #Empty detections
        df = pd.DataFrame(columns=columns) 
    else:
        df = pd.DataFrame(detlist)
        df.columns = columns   
    return df


def segment_iou(pred_segment, gt_segments):
    """
    Compute the temporal intersection over union between a
    predicted segment and all the test segments.
    Parameters
    ----------
    pred_segment : 1d array
        Temporal predicted segment containing [starting, ending] times.
    gt_segments : 2d array
        Temporal ground_truth segments containing N x [starting, ending] times.
    Outputs
    -------
    tIoU : 1d array
        Temporal intersection over union score of the N's ground-truth segments.
    """
    pred_segment = pred_segment.astype(float)
    tIoU = None

    for i in gt_segments:
        tt1 = np.maximum(pred_segment[0], gt_segments[:, 0])
        tt2 = np.minimum(pred_segment[1], gt_segments[:, 1])

        # Intersection including Non-negative overlap score.
        segments_intersection = (tt2 - tt1).clip(0)

        # Segment union.
        segments_union = (gt_segments[:, 1] - gt_segments[:, 0]) \
        + (pred_segment[1] - pred_segment[0]) - segments_intersection

        # Compute overlap as the ratio of the intersection over union of two segments.
        tIoU = segments_intersection.astype(float) / segments_union

        if gt_segments.shape[0] < 2: # only single gt segment per vid
            tIoU = np.array([tIoU])

    return tIoU


def softmax(x, dim):
    x = F.softmax(torch.from_numpy(x), dim=dim)
    return x.numpy()


def metric_scores(pth, **all_params):
    '''
    Arguments:
    -------------
    pth: 
        Path to a video's output from test.py module based
        on rgb/flow/both/late-fusion modality. 

    all_params: 
        Value of all parameters based on config file

    Returns:
    --------------
    metric:
        The average scores of all branches
    
    branch_scores_dict: 
        Dictionary to store scores of every individual branches
    
    out_detections:
        Entries containing localized detections
        in the form of [videoname, start, end, class, confidence_score]

    '''
    detect_params = all_params['detect_params']
    cas_data = np.load(pth)
    video_name = pth.split('/')[-1][:-4]
    fps = 30
    out_detections = []
    metric = None

    avg_score = cas_data['avg_score']
    global_score = cas_data['global_score']
    branch_scores = cas_data['branch_scores']

    global_score = softmax(global_score, dim=0)
    frame_cnt = avg_score.shape[0]*all_params['base_sample_rate']*all_params['sample_rate']
    duration = frame_cnt/fps
    ######################## Get Average score for all branches ########################
    for class_id in range(all_params['action_class_num']):
        if global_score[class_id] <= detect_params['global_score_thrh']:
            continue
        metric = softmax(avg_score, dim=1)[:, class_id:class_id + 1]
        metric = normalize(metric)

        metric = interpolate(
                metric[:, 0],
                all_params['feature_type'],
                frame_cnt,
                all_params['base_sample_rate'] * all_params['sample_rate'],
                all_params['base_snippet_size'],
                detect_params['interpolate_type']
            )
        metric = np.expand_dims(metric, axis=1)

        ######################## Generate Action Localization ########################
        mask = detect_with_thresholding(
            metric, 
            detect_params['thrh_type'], 
            detect_params['thrh_value'],
            detect_params['proc_type'], 
            detect_params['proc_value'])

        temp_out = mask_to_detections(
            mask, 
            metric, 
            detect_params['weight_inner'],
            detect_params['weight_outter'])

        for entry in temp_out:

            entry[2] = class_id
            entry[3] += global_score[class_id] * detect_params['weight_global']

            entry[0] = (entry[0] + detect_params['sample_offset']) / fps
            entry[1] = (entry[1] + detect_params['sample_offset']) / fps

            entry[0] = max(0, entry[0])
            entry[1] = max(0, entry[1])
            entry[0] = min(duration, entry[0])
            entry[1] = min(duration, entry[1])

        for entry_id in range(len(temp_out)):
            temp_out[entry_id] = [video_name] + temp_out[entry_id]

        out_detections += temp_out

    for entry in out_detections:
        class_id = entry[3]
        class_name = new_cls_names['ucf_crime'][class_id]
        old_class_id = int(old_cls_indices['ucf_crime'][class_name])
        entry[3] = old_class_id

        
    ######################## Get every individual branch scores ########################
    branch_scores_dict = {}
    
    for branch_num in range(branch_scores.shape[0]):
        for class_id in range(all_params['action_class_num']):
            if global_score[class_id] <= detect_params['global_score_thrh']:
                continue
            
            b_metric = softmax(branch_scores[branch_num, 0], dim=1)[:, class_id:class_id + 1]
            b_metric = normalize(b_metric)
        
            b_metric = interpolate(
                    b_metric[:, 0],
                    all_params['feature_type'],
                    frame_cnt,
                    all_params['base_sample_rate'] * all_params['sample_rate'],
                    all_params['base_snippet_size'],
                    detect_params['interpolate_type']
                )
            b_metric = np.expand_dims(b_metric, axis=1)
            branch_scores_dict[branch_num] = b_metric

    if metric is not None:
        frame_cnt = metric.shape[0]
    else: #Action instances (global_score) < threshold .1 
        metric = np.zeros((frame_cnt, 1))
        for b in range(branch_scores.shape[0]):
            branch_scores_dict[b] = np.zeros((frame_cnt, 1))

    return metric, frame_cnt, branch_scores_dict, out_detections

