# Defining a function for all steps

def prediction_pipeline(labeled_array, labels_list, regression_model, scoring='neg_root_mean_squared_error'):

  '''
  Pipeline for predicting eGeMAPS features from some or all dimensions of Hybrid BYOL-S embeddings

  Parameters
  ------------
  labeled_array: numpy array
      The embeddings
  labels_list: list
      The list of labels
  regression_model: string
      The regression model
  scoring: string
      The metric for measuring the performance

  Returns
  ------------
  r: int
      The R-squared value of prediction
  p: int
      The pearson correlation of prediction
  '''

  X_train, X_test, y_train, y_test = train_test_split(labeled_array, labels_list, test_size=0.20)

  if regression_model == 'ridge':
    regressor = Ridge()
    parameters = {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    r, p = get_hyperparams(X_train, X_test, y_train, y_test, regressor, parameters, scoring=scoring)

  return r, p


# Defining a function for hyperparameter tuning and getting the accuracy on the test set
def get_hyperparams(X_train, X_test, y_train, y_test, regressor, parameters, scoring='neg_mean_squared_error'): # check scoring
  '''
  Splits into training and testing set with different speakers

  Parameters
  ------------
  X_train: numpy array
    The normalised embeddings that will be used for training
  X_test: numpy array
    The embeddings that will be used for testing
  y_train: list
    The labels that will be used for training
  y_test: list
    The labels that will be used for testing
  regressor: object
    The instance of the regression model
  parameters: dictionary
    The dictionary of parameters for GridSearchCV

  Returns
  ------------
  r_squared: int
      The R-squared value of prediction
  pearson_correlation: int
      The pearson correlation of prediction
  '''

  # Define the pipeline
  pipe = Pipeline([
          ('scaler', StandardScaler()),
          ('model', regressor)
      ])

  grid = GridSearchCV(pipe, param_grid=parameters, cv=5, n_jobs=-1, scoring=scoring)
  grid.fit(X_train, y_train)

  y_pred = grid.predict(X_test)
  r_squared = r2_score(y_test, y_pred)
  pearson_correlation = np.corrcoef(y_test, y_pred)[0, 1]

  return r_squared, pearson_correlation


def top_byols_predict_ege(e):
  
  '''
  predict egemaps features one at a time using top or all dimensions of Hybrid BYOL-S

  Parameters
  ------------
  e: string
      The emotion on which the analysis needs to be done

  Returns
  ------------
  prediction_metrics_dict: dictionary
      dictionary with the egemaps features, pearson top, pearson all and percentage

  '''

  en_errors_r_squared_top = []
  en_errors_r_squared_all = []
  pearson_corr_top = []
  pearson_corr_all = []
  percentage = []

  byols_embeddings, byols_labels, byols_speakers = emotion_all_merger(emotion=e, other_emotions=[em for em in emotions if em!=e], embeddings_dict=byols_embeddings_dict, speakers_dict=byols_speakers_list_dict, num=45)
  ege_embeddings, ege_labels, ege_speakers = emotion_all_merger(emotion=e, other_emotions=[em for em in emotions if em!=e], embeddings_dict=eGeMAPS_embeddings_dict, speakers_dict=eGeMAPS_speakers_list_dict, num=45)

  top_features = byols_fi_dict[e][:num_byols_features[e]]
  all_features = byols_fi_dict[e][:]


  for i in tqdm(range(88)):

    en_r_squared_top, p_corr_top = prediction_pipeline(byols_embeddings[:, top_features], ege_embeddings[:,i], regression_model='ridge')
    en_errors_r_squared_top.append(np.round(en_r_squared_top, 3))
    pearson_corr_top.append(np.round(p_corr_top, 3))

    en_r_squared_all, p_corr_all = prediction_pipeline(byols_embeddings[:, all_features], ege_embeddings[:,i], regression_model='ridge')
    en_errors_r_squared_all.append(np.round(en_r_squared_all, 3))
    pearson_corr_all.append(np.round(p_corr_all, 3))

    percentage.append(np.round(p_corr_top*100/p_corr_all, 3))

  prediction_metrics_dict = {'Feature':egemaps_feature_names, 'Pearson top':pearson_corr_top, 'Pearson all': pearson_corr_all, 'Percentage':percentage}

  return prediction_metrics_dict



def summary_table(e):

    '''
    Makes a summary table with the prediction metrics for the most important egemaps features

    Parameters
    ------------
    e: string
        The emotion on which the analysis needs to be done

    Returns
    ------------
    sorted_df: pandas dataframe
        pandas dataframe with the egemaps features, importance, pearson top, pearson all and percentage (sorted in decreasing order of importance)

    '''

    df1 = egemaps_top_features_df[e]
    df2 = predicted_df_dict[e]
    merged_df = pd.merge(df1, df2, on='Feature', how='inner')
    return merged_df
