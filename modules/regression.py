predicted_df_dict = {}

import numpy as np
from tqdm import tqdm

def top_wavlm_predict_ege(e):
    rmse_top_list = []
    rmse_all_list = []
    rmse_top_all = []
    rmse_top_squared_all = []
    weighted_difference_rmse_all = []
    harmonic_mean_metric_all = []
    change_index_adjusted_performance_all = []
    performance_weighted_change_score_all = []

    wavlm_embeddings, wavlm_labels, wavlm_speakers = wavlm_embeddings_all[e], labels_all[e], speakers_all[e]
    ege_embeddings, ege_labels, ege_speakers = egemaps_embeddings_all[e], labels_all[e], speakers_all[e]

    top_features = wavlm_fi_dict[e][:num_wavlm_features[e]]
    all_features = wavlm_fi_dict[e][:]

    for i in tqdm(range(88)):
        rmse_top = prediction_pipeline(wavlm_embeddings[:, top_features], ege_embeddings[:,i], regression_model='ridge')
        rmse_all = prediction_pipeline(wavlm_embeddings[:, all_features], ege_embeddings[:,i], regression_model='ridge')
        rmse_top_list.append(np.round(rmse_top, 3))
        rmse_all_list.append(np.round(rmse_all, 3))
        rmse_top_all.append(np.round(rmse_top/rmse_all, 3))
        rmse_top_squared_all.append(np.round(rmse_top**2/rmse_all, 3))

        # Calculate additional metrics
        weighted_difference_rmse_all.append(np.round(weighted_difference_rmse(rmse_all, rmse_top), 3))
        harmonic_mean_metric_all.append(np.round(harmonic_mean_metric(rmse_all, rmse_top), 3))
        change_index_adjusted_performance_all.append(np.round(change_index_adjusted_performance(rmse_all, rmse_top), 3))
        performance_weighted_change_score_all.append(np.round(performance_weighted_change_score(rmse_all, rmse_top), 3))

    prediction_metrics_dict = {
        'Feature': egemaps_feature_names,
        'rmse_top_all': rmse_top_all,
        'rmse_top_squared_all': rmse_top_squared_all,
        'rmse_top': rmse_top_list,
        'rmse_all': rmse_all_list,
        'weighted_difference_rmse_all': weighted_difference_rmse_all,
        'harmonic_mean_metric_all': harmonic_mean_metric_all,
        'change_index_adjusted_performance_all': change_index_adjusted_performance_all,
        'performance_weighted_change_score_all': performance_weighted_change_score_all
    }
    return prediction_metrics_dict
