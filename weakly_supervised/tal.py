import matplotlib
matplotlib.use('Agg')

import numpy as np 
import pandas as pd 
import os
import io
import base64
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg
from utils import smooth, ucf_crime_old_cls_names, ucf_crime_old_cls_indices
from utils import prepare_gt, prepare_detections, segment_iou


class GenerateTALAllOutput(object):
    def __init__(self, videoname, gt_file_pth, detected_list, frame_cnt):
        self.videoname = videoname
        self.gt_file_pth = gt_file_pth
        self.detected_list = detected_list
        self.frame_cnt = frame_cnt

    def return_all_tal_data(self):
        ground_truth = os.listdir(self.gt_file_pth)[0]
        ground_truth = os.path.join(self.gt_file_pth, ground_truth)

        gtdf = prepare_gt(ground_truth)
        detdf = prepare_detections(self.detected_list)
        num_class = len(ucf_crime_old_cls_names.keys())

        gt_by_cls, det_by_cls = [], []
        for clss in range(1, num_class+1):
            gt_by_cls.append(gtdf[gtdf['cls'] == clss].reset_index(drop=True).drop('cls', 1))

            # For predicted sections, get all segments regardless of class
            det_by_cls.append(detdf[detdf['videoname'].str.contains(ucf_crime_old_cls_names[clss])])

        for clss in range(num_class):
            class_name = ucf_crime_old_cls_names[clss+1]
            if self.videoname.startswith(class_name):
                predicted, gt = get_predict_gt_start_end(
                    gt_by_cls[clss][gt_by_cls[clss]['videoname'] == self.videoname], 
                    det_by_cls[clss][det_by_cls[clss]['videoname'] == self.videoname]
                )
                break

        return predicted, class_name, gt

       

class GenerateTALPredictions(GenerateTALAllOutput):
    def __init__(self, videoname, gt_file_pth, detected_list, frame_cnt):
        super().__init__(videoname, gt_file_pth, detected_list, frame_cnt) 

    def get_only_predicted_results(self):
        predicted, class_name, _ = super().return_all_tal_data()
        ################ Get predicted barcodes (for plotting barchart) ################
        # If no correct class predicted
        misclassify_flag = True

        if predicted.empty:
            predicted_class = 'None'
        else: #Contains at least 1 prediction/detection
            predicted_class = ''
            if predicted[predicted['cls'].str.contains(class_name)].shape[0] == 0:
                predicted = predicted
                predicted_class = np.unique(predicted['cls'])[0]
            # Contains correct classification
            else:
                predicted = predicted[predicted['cls'].str.contains(class_name)]
                misclassify_flag = False
                predicted_class = np.unique(predicted['cls'])[0]

        start_end_conf = predicted[['start', 'end', 'conf']]
        predicted = predicted[['start', 'end']].values.astype(float) * 30 #fps

        predicted_barcode = np.zeros((1, self.frame_cnt))
        for pred_rows in range(len(predicted)):
            start_frame = int(predicted[pred_rows, 0])
            end_frame = int(predicted[pred_rows, 1])
            predicted_barcode[:, start_frame: end_frame] = 1

        assert(predicted_barcode.shape[1] == self.frame_cnt)
        return predicted_barcode, misclassify_flag, predicted_class, start_end_conf


class GenerateTALGroundTruth(GenerateTALAllOutput):
    def __init__(self, videoname, gt_file_pth, detected_list, frame_cnt):
        super().__init__(videoname, gt_file_pth, detected_list, frame_cnt)

    def get_only_gt_results(self): 
        _, _, gt = super().return_all_tal_data()
        ################ Get ground truth barcodes (for plotting barchart) ################
        gt = gt[['start', 'end']].values * 30 #fps

        gt_barcode = np.zeros((1, self.frame_cnt))
        for gt_rows in range(len(gt)):
            start_frame = int(gt[gt_rows, 0])
            end_frame = int(gt[gt_rows, 1])
            gt_barcode[:, start_frame: end_frame] = 1

        assert(gt_barcode.shape[1] == self.frame_cnt)
        return gt_barcode, gt



def compute_localization_prediction(ground_truth, prediction):
    """
    Compute and return detected localized actions whose IOU score 
    above .0 
    Parameters
    ----------
    ground_truth : 
        Data frame containing the ground truth instances.
        Required fields: ['videoname', 'start', 'end']
    prediction : 
        Data frame containing the prediction instances.
        Required fields: ['videoname', 'start', 'end', 'cls', 'conf']

    Outputs
    -------
    prediction: 
        Dataframe containing following fields: 
        ['videoname', 'start', 'end', 'cls', 'iou_score']

    """
    prediction = prediction.reset_index(drop=True)

    # Sort predictions by decreasing (confidence) score order.
    sort_idx = prediction['conf'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)


    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('videoname')

    # Assigning true positive to truly grount truth instances.
    iou_score_list = []
    for idx, this_pred in prediction.iterrows():
        try:
            # Check if there is at least one ground truth in the video associated with predicted video.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['videoname'])
        except Exception as e:
            fp[:, idx] = 1
            continue


        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['start', 'end']].values, this_gt[['start', 'end']].values)

        iou_score_list.append(np.max(tiou_arr))

    # Clean-up dataframe
    cls_name_list = []
    #prediction = prediction.drop(columns=['conf'])
    prediction['iou_score'] = iou_score_list
    cls_num = [i for i in prediction['cls'].values]
    for j in cls_num:
        cls_name_list.append(ucf_crime_old_cls_names[int(j)])
    prediction['cls'] = cls_name_list
    flag = prediction['iou_score'].values != .0 # Remove prediction that has zero IoU score
    prediction = prediction[flag]

    return prediction


def get_predict_gt_start_end(gt, prediction):
    predicted = compute_localization_prediction(gt, prediction)
    return predicted, gt


def plot_tal_charts(
    videoname, 
    gt_file_pth, 
    detected_list, 
    frame_cnt,
    modality):

    '''
    Arguments:
    ----------
    videoname: 
        Name of the selected video
    gt_file_pth: 
        Path to ground truth annotations
    detected_list:
        Possible detected proposals (multiple)
    frame_cnt:
        Total frame numbers for the selected video
    modality:
        Modality of CAS based on selected videoname (rgb, flow, ...)


    Returns:
    ----------
    talImageB64String:
        Base-64 string representation of charts plot for TAL (per modality)
    predicted_class:
        Class label based on detections (per modality)
    start_end_conf:
        Localized predictions in form of [start, end, confidence score] dataframe
    '''

    tal_predict_output = GenerateTALPredictions(videoname, gt_file_pth, detected_list, frame_cnt)
    predicted_barcode, misclassify_flag, predicted_class, start_end_conf = tal_predict_output.get_only_predicted_results()

    fig = plt.figure(figsize=(10, 1.5))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    fig.add_subplot(gs[0, 0])
    plt.yticks([])
    axes = plt.gca()

    #Plot Full prediction
    if misclassify_flag is False:
        barprops = dict(aspect='auto',
                        cmap=plt.cm.Greens,
                        interpolation='nearest')
    else:
        barprops = dict(aspect='auto',
                    cmap=plt.cm.Greys,
                    interpolation='nearest')
    axes.imshow(predicted_barcode, **barprops)

    #Remove fig margin
    fig.tight_layout()

    # Convert fig to PNG image
    fig_image = io.BytesIO()
    FigureCanvasAgg(fig).print_png(fig_image)
    # Encode PNG image to base64 string
    talImageB64String = "data:image/png;base64,"
    talImageB64String += base64.b64encode(fig_image.getvalue()).decode('utf8')

    return talImageB64String, predicted_class, start_end_conf


def plot_gt_charts(
    videoname, 
    gt_file_pth, 
    detected_list, 
    frame_cnt,
    modality):
    '''
    Arguments:
    ----------
    videoname: 
        Name of the selected video
    gt_file_pth: 
        Path to ground truth annotations
    detected_list:
        Possible detected proposals (multiple)
    frame_cnt:
        Total frame numbers for the selected video
    modality:
        Modality of CAS based on selected videoname (rgb, flow, ...)


    Returns:
    ----------
    talImageB64String:
        Base-64 string representation of ground truth charts for TAL
    gt_start_end:
        Localized ground truth annotations [[start, end]] in numpy array form
    '''

    tal_gt_output = GenerateTALGroundTruth(videoname, gt_file_pth, detected_list, frame_cnt)
    gt_barcode, gt_start_end = tal_gt_output.get_only_gt_results()

    fig = plt.figure(figsize=(10, 1.5))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    fig.add_subplot(gs[0, 0])
    plt.yticks([])
    axes = plt.gca()

    #Plot Ground truth
    barprops = dict(aspect='auto',
                    cmap=plt.cm.Oranges,
                    interpolation='nearest')

    axes.imshow(gt_barcode, **barprops)

    #Remove fig margin
    fig.tight_layout()

    # Convert fig to PNG image
    fig_image = io.BytesIO()
    FigureCanvasAgg(fig).print_png(fig_image)
    # Encode PNG image to base64 string
    gt_talImageB64String = "data:image/png;base64,"
    gt_talImageB64String += base64.b64encode(fig_image.getvalue()).decode('utf8')

    return gt_talImageB64String, gt_start_end



