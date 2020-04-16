import matplotlib
matplotlib.use('Agg')

import os
import io
import base64
import numpy as np
import numpy.matlib
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.io import loadmat
from utils import prepare_gt


def visualize_mil(
    videoname, 
    frame_cnt, 
    path_to_MIL_test_output_dir,
    gt_file_pth):
    '''
    Parameters:
    ---------
        videoname: 
            Videoname of a particular choosen test set
        frame_cnt: 
            Frame length for corresponding test set
        path_to_MIL_test_output_dir:
            Path to directory containing all test set's prediction (MIL)
        gt_file_pth:
            Path to file containing ground-truth at temporal-level

    Returns:
    ---------
        x: 
            Interval for frames count on the x-axis
        y:
            Scores at the y-axis for corresponding frame intervals.
        gt_bar: 
            Numpy array containing the values for plotting ground-truth bar chart.

    '''
    fps = 30
    # Obtain temporal annotations
    ground_truth = os.listdir(gt_file_pth)[0]
    ground_truth = os.path.join(gt_file_pth, ground_truth)
    gtdf = prepare_gt(ground_truth)
    gtdf = gtdf[gtdf['videoname'] == videoname]
    gtdf = gtdf[['start', 'end']].values * fps

    gt_bar = np.zeros((1, frame_cnt))
    for gt_rows in range(len(gtdf)):
        start_frame = int(gtdf[gt_rows, 0])
        end_frame = int(gtdf[gt_rows, 1])
        gt_bar[:, start_frame: end_frame] = 1

    assert(gt_bar.shape[1] == frame_cnt)

    
    videoname_MIL = videoname + '_C'
    video_MIL_output = os.path.join(
        path_to_MIL_test_output_dir,
        videoname_MIL
    )

    predictions_mil = loadmat(video_MIL_output)
    predictions_mil = predictions_mil['predictions']

    #---- Prediction of scores for each 32 segments in MIL ------
    num_seg = 32
    total_segments = np.linspace(1, frame_cnt, num=num_seg+1)
    total_segments = total_segments.round()

    Frames_Score = []
    count = -1
    for iv in range(0, num_seg):
        F_Score = np.matlib.repmat(
            predictions_mil[iv],
            1,
            (int(total_segments[iv+1])-int(total_segments[iv]))
        )
        count = count + 1
        if count == 0:
            Frames_Score = F_Score
        if count > 0:
            Frames_Score = np.hstack((Frames_Score, F_Score))

    x = np.linspace(1, frame_cnt, frame_cnt)
    scores = Frames_Score
    scores1 = scores.reshape((scores.shape[1],))
    y = scipy.signal.savgol_filter(scores1, 101, 3)
    x = x.tolist()
    y = y.tolist()

    return x, y, gt_bar



def plot_mil_charts(x, y, gt_bar):
    '''
    Parameters:
    ---------
        x: 
            Frames of selected video along the x-axis in chart.
        y: 
            Scores of every frame along the y-axis in chart.
        gt_bar:
            Values of ground-truth (temporal-level) for plotting bar chart.


    Returns:
    ---------
        milImageB64String:
            Base-64 string representation of charts plot for MIL
    '''

    x, y = x[1:], y
    height = gt_bar.flatten().tolist()[1:]
    assert(len(x) == len(height))
    assert(len(x) == len(y))

    fig = plt.figure()
    frame_cnt = len(x)
    xmin = 0
    xmax = frame_cnt
    ymin = 0
    ymax = 1
    plt.axis([xmin, xmax, ymin, ymax])
    plt.bar(x, height=height, width=1.0, color='#893101')
    plt.plot(x, y, color='green', linewidth=3)
    plt.yticks([])
    fig.tight_layout()
    
    # Convert fig to PNG image
    fig_image = io.BytesIO()
    FigureCanvasAgg(fig).print_png(fig_image)

    # Encode PNG image to base64 string
    milImageB64String = "data:image/png;base64,"
    milImageB64String += base64.b64encode(fig_image.getvalue()).decode('utf8')
    return milImageB64String