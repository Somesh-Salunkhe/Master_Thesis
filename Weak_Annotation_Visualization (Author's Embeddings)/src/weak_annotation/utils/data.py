# Data operation utilities
import numpy as np
import pandas as pd

def sliding_window_samples(data, win_len, overlap_ratio=None):
    windows = []
    indices = []
    curr = 0
    overlapping_elements = 0
    float_prec = False

    if overlap_ratio is not None:
        if not ((overlap_ratio / 100) * win_len).is_integer():
            float_prec = True
        overlapping_elements = int((overlap_ratio / 100) * win_len)
        if overlapping_elements >= win_len:
            print('Number of overlapping elements exceeds window size.')
            return
    changing_bool = True
    while curr < len(data) - win_len:
        windows.append(data[curr:curr + win_len])
        indices.append([curr, curr + win_len])
        if (float_prec == True) and (changing_bool == True):
            curr = curr + win_len - overlapping_elements - 1
            changing_bool = False
        else:
            curr = curr + win_len - overlapping_elements
            changing_bool = True
    return np.array(windows), np.array(indices)

def apply_sliding_window(data, window_size, window_overlap, weights=None):
    output_x = None
    output_y = None
    output_w = None
    output_sbj = []
    for i, subject in enumerate(np.unique(data[:, 0])):
        subject_data = data[data[:, 0] == subject]
        if weights is not None:
            subject_w = weights[data[:, 0] == subject]
            tmp_w, _ = sliding_window_samples(subject_w, window_size, window_overlap)
        subject_x, subject_y = subject_data[:, :-1], subject_data[:, -1]
        tmp_x, _ = sliding_window_samples(subject_x, window_size, window_overlap)
        tmp_y, _ = sliding_window_samples(subject_y, window_size, window_overlap)
        if weights is not None:
            if output_w is None:
                output_w = tmp_w
            else:
                output_w = np.concatenate((output_w, tmp_w), axis=0)
        if output_x is None:
            output_x = tmp_x
            output_y = tmp_y
            output_sbj = np.full(len(tmp_y), subject)
        else:
            output_x = np.concatenate((output_x, tmp_x), axis=0)
            output_y = np.concatenate((output_y, tmp_y), axis=0)
            output_sbj = np.concatenate((output_sbj, np.full(len(tmp_y), subject)), axis=0)
    output_y = [[i[-1]] for i in output_y]
    if weights is not None:
        return output_sbj, output_x[:, :, 1:], np.array(output_y).flatten().astype(int), np.array(output_w).flatten()
    else:
        return output_sbj, output_x[:, :, 1:], np.array(output_y).flatten().astype(int)

def unwindow_inertial_data(orig, ids, preds, win_size, win_overlap):
    unseg_preds = []
    if not ((win_overlap / 100) * win_size).is_integer():
        float_prec = True
    else:
        float_prec = False

    for sbj in np.unique(orig[:, 0]):
        sbj_data = orig[orig[:, 0] == sbj]
        sbj_preds = preds[ids==sbj]
        sbj_unseg_preds = []
        changing_bool = True
        for i, pred in enumerate(sbj_preds):
            if (float_prec == True) and (changing_bool == True):
                sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * (int(win_size * (1 - win_overlap * 0.01)) + 1)))
                if i + 1 == len(preds):
                    sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * (int(win_size * (win_overlap * 0.01)) + 1)))
                changing_bool = False
            else:
                sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * (int(win_size * (1 - win_overlap * 0.01)))))
                if i + 1 == len(preds):
                    sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * int(win_size * (win_overlap * 0.01))))
                changing_bool = True
        sbj_unseg_preds = np.concatenate((sbj_unseg_preds, np.full(len(sbj_data) - len(sbj_unseg_preds), sbj_preds[-1])))
        unseg_preds = np.concatenate((unseg_preds, sbj_unseg_preds))
    assert len(unseg_preds) == len(orig)    
    return unseg_preds, orig[:, -1]

def convert_samples_to_segments(ids, labels, sampling_rate):
    f_video_ids, f_labels, f_t_start, f_t_end, f_score = [], np.array([]), np.array([]), np.array([]), np.array([])
    for id in np.unique(ids):
        sbj_labels = labels[(ids == id)]
        curr_start_i = 0
        curr_end_i = 0
        curr_label = sbj_labels[0]
        for i, l in enumerate(sbj_labels):
            if curr_label != l:
                act_start = curr_start_i / sampling_rate
                act_end = curr_end_i / sampling_rate
                act_label = curr_label - 1
                if curr_label != 0:
                    f_video_ids.append('sbj_' + str(int(id)))
                    f_labels = np.append(f_labels, act_label)
                    f_t_start = np.append(f_t_start, act_start)
                    f_t_end = np.append(f_t_end, act_end)
                    f_score = np.append(f_score, 1)
                curr_label = l
                curr_start_i = i + 1
                curr_end_i = i + 1    
            else:
                curr_end_i += 1        
    return {
        'video-id': f_video_ids,
        'label': f_labels,
        't-start': f_t_start,
        't-end': f_t_end,
        'score': f_score
    }
