import os
import pandas as pd
import time
import cv2

# from current app package(dir) import app module
from app import app 
from flask import render_template, request, Response
from weakly_supervised import mil, tal
from utils import metric_scores, load_config_file, ucf_crime_old_cls_names

# access settings 
from settings import path_to_MIL_test_output_dir, cas_dir
from settings import tal_config_file, ucf_video_dir, gt_file_pth, fps



@app.route("/")
def submit():    
    return render_template("submit.html")


@app.route("/get_result", methods=['POST'])
def get_result():
    test_subset_name = 'test'
    all_params = load_config_file(tal_config_file)
    locals().update(all_params)

    # cas_file allows user to select and upload a single CAS file (i.e. Explosion008_x264.npz file)
    # Doesn't matter from which CAS modalities(dirs), as all the modalities detection will be displayed
    cas_file = request.files['cas_file']
    videoname = cas_file.filename[:-4]

    
    class_name = ''
    num_class = len(ucf_crime_old_cls_names.keys())
    for clss in range(num_class):
        class_name = ucf_crime_old_cls_names[clss+1]
        if videoname.startswith(class_name):
            class_name = class_name
            break

    global video_path
    video_path = os.path.join(ucf_video_dir, '{}/{}.mp4'.format(class_name, videoname))
    video_path = video_path.replace('\\', '/')
    
    list_modality = os.listdir(cas_dir)
    talImageB64String, predicted_class_dict  = {}, {}
    merge_localized_actions = []
    #Retrieve data from TAL
    for mod in list_modality:
        modality = mod.split('-')[-1] #Only get the modality name
        pth_to_modality = os.path.join(cas_dir, mod)
        pth_to_modality = os.path.join(pth_to_modality, '{}.npz'.format(videoname))
        pth_to_modality = pth_to_modality.replace('\\', '/')
    
        _, frame_cnt, _, detected_list = metric_scores(pth_to_modality, **all_params)

        talImageB64String[modality], predicted_class, start_end_conf = tal.plot_tal_charts(
            videoname, gt_file_pth, detected_list, 
            frame_cnt, modality
        )

        predicted_class_dict[modality] = predicted_class
        start_end_conf['modality'] = modality
        start_end_conf.start = start_end_conf.start * fps
        start_end_conf.end = start_end_conf.end * fps
        merge_localized_actions.append(start_end_conf)

    # predict_df_display contains [start,end,confidence_score,modality] of every localized actions
    predict_df_display = pd.concat(merge_localized_actions)
    # For each modalities, sort [start & end] values
    predict_df_display = predict_df_display.groupby(['modality']).apply(
        lambda x: x.sort_values(['start', 'end'])
    )


    gt_talImageB64String, gt_start_end = tal.plot_gt_charts(
        videoname, gt_file_pth, detected_list, 
        frame_cnt, _
    )
    # gt_df_display contains [start,end]
    gt_df_display = pd.DataFrame(gt_start_end, columns=['start', 'end'])
    gt_df_display = gt_df_display.sort_values(by=['start', 'end'])

    #Retrieve data from MIL
    x_axis_frame, y_axis_score, gt_bar = mil.visualize_mil(
        videoname, 
        frame_cnt, 
        path_to_MIL_test_output_dir, 
        gt_file_pth
    )
    milImageB64String = mil.plot_mil_charts(
        x_axis_frame, y_axis_score, gt_bar)


    api_response = {
        'modality': modality,
        'video_path': video_path,
        'fig_mil': milImageB64String,
        'tal_gt': gt_talImageB64String,
        'fig_tal_both': talImageB64String['both'],
        'class_both': predicted_class_dict['both'],
        'fig_tal_rgb': talImageB64String['rgb'],
        'class_rgb': predicted_class_dict['rgb'],
        'fig_tal_flow': talImageB64String['flow'],
        'class_flow': predicted_class_dict['flow'],
        'fig_tal_late_fusion': talImageB64String['fusion'],
        'class_late_fusion': predicted_class_dict['fusion'],
        'frame_cnt': frame_cnt,
        'predict_df_display': [predict_df_display.to_html(index=False)],
        'gt_df_display': [gt_df_display.to_html(index=False)]
    }

    return render_template(
        "result.html", 
        **api_response
    )


def generate_frame(starting_frame):
    if starting_frame is None:
        starting_frame = 0
    cap = cv2.VideoCapture(video_path)
    while True:
        _, frame = cap.read()
        
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.rectangle(
            frame,
            (100, 0),
            (190, 25),
            (0, 0, 0),
            cv2.FILLED
        )
        
        cv2.putText(
            frame, 
            str(cap.get(cv2.CAP_PROP_POS_FRAMES)), 
            (100, 17), 
            font, 
            0.7, 
            (255, 255, 255), 
            1)

        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= float(starting_frame):
            (_, encodedImage) = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + 
                b'\r\n'
            )
            time.sleep(.6 / fps)

    cap.release()



@app.route('/get_frame_num', methods=['GET'])
def get_frame_num():
    selected_frame_num = request.args.get('frame_num')
    if selected_frame_num == "" or selected_frame_num is None:
        selected_frame_num = 0
    print('Selected frame num: {}'.format(selected_frame_num))

    return Response(
        generate_frame(selected_frame_num),
        mimetype='multipart/x-mixed-replace; boundary=frame')