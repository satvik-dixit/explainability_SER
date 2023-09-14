# function for dividing the dataset into dictionaries with emotions as keys and embeddings/labels/speakers as values

def label_division(embeddings_array, labels, required_labels, speakers):

    '''
    Divides the dataset into separate emotions

    Parameters
    ------------
    embeddings_array: numpy aray
        The embeddings of the samples
    labels: list
        The list of emotions of all samples
    required_labels: list
        The list of emotion categories which need to be included
    speaker: list
        The list of speakers of all samples

    Returns
    ------------
    final_embeddings_dict: dictionary
        A dictionary of which contains the embeddings for every emotion separately
    final_labels_dict: dictionary
        A dictionary of which contains the labels for every emotion separately
    final_speakers_list_dict: dictionary
        A dictionary of which contains the speakers for every emotion separately
    '''

    final_embeddings_dict = {}
    final_labels_dict = {}
    final_speakers_list_dict = {}

    for label in required_labels:
        label_indices = np.where(np.array(labels) == label)[0]
        label_embeddings = embeddings_array[label_indices, :]
        final_speakers_list_dict[label] = list(np.array(speakers)[label_indices])
        final_embeddings_dict[label] = label_embeddings
        final_labels_dict[label] = [label] * len(label_indices)

    return final_embeddings_dict, final_labels_dict, final_speakers_list_dict


# function for combining one emotion (50%) and the rest of the 5 emotions (10% each)

def emotion_all_merger(emotion, other_emotions, embeddings_dict, speakers_dict, num=45):

  '''
  Merges emotions to form a new dataset combining one emotion (50%) and the rest of the emotion (equally in the remaining 50%)

  Parameters
  ------------
  emotion: string
      The emotion category which is 50%
  other_emotions: string
      The list of emotions of all the other emotions which combine to form the other 50%
  embeddings_dict: dictionary
      The dictionary of embeddings
  speakers_dict: dictionary
      The dictionary of speakers
  num: number of samples of the emotion

  Returns
  ------------
  embeddings: numpy array
      The embeddings of the new dataset
  labels: list
      The labels of the new dataset
  speakers: list
      The speakers of the new dataset
  '''

  np.random.seed(42)

  emo_indices = np.random.choice(len(speakers_dict[emotion]), size=num, replace=False)

  embeddings = embeddings_dict[emotion][emo_indices]
  speakers = [speakers_dict[emotion][i] for i in emo_indices]
  labels = [1]*num

  for em in other_emotions:
    em_indices = np.random.choice(len(speakers_dict[em]), size=int(num/len(other_emotions)), replace=False)
    em_speakers = [speakers_dict[em][i] for i in em_indices]
    embeddings_e = embeddings_dict[em][em_indices]

    embeddings = np.concatenate((embeddings, embeddings_e), axis=0)
    speakers = speakers + em_speakers
    labels = labels + [0]*int(num/len(other_emotions))

  return embeddings, labels, speakers


# Function for ranking all features in decreasing order of feature importance using coefficients

def feature_importance_method(feature_names, trained_model):

  '''
  Returns the feature importance of every feature sorted in decreasing order

  Parameters
  ------------
  feature_names: list
      The list of names of the features
  trained_model: model 
      The model trained on classification task

  Returns
  ------------
  sorted_importances: list
     The list of features and their feature importance values sorted in decreasing order

  '''

  coef_abs = abs(trained_model.coef_[0])
  feature_indices = np.argsort(coef_abs)[::-1]
  sorted_importances = [(feature_names[i], coef_abs[i]) for i in feature_indices]
  return sorted_importances


def classification_pipeline(embeddings, speakers, labels_list, feature_names):

    '''
    Loads and resamples audio files

    Parameters
    ------------
    embeddings: numpy aray
        The embeddings of the dataset
    speakers: list
        The list of speakers of all samples
    labels_list: list
        The list of emotions (labels) of all samples
  feature_names: list
      The list of names of the features

    Returns
    ------------
    model: model
      The trained model
    mean_f1: int
      The mean F1 score of classification
    '''

    np.random.seed(42)

    X = embeddings
    y = labels_list
    groups = speakers

    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

    # Outer cross-validation loop
    outer_cv = GroupShuffleSplit(n_splits=5, train_size=0.8, random_state=42)
    outer_scores = []
    feats_array = []
    for train_index, test_index in outer_cv.split(X, y, groups):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

        # Inner cross-validation loop for hyperparameter tuning
        inner_cv = GroupShuffleSplit(n_splits=5, train_size=0.8, random_state=42)
        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=inner_cv)
        grid_search.fit(X_train, y_train, groups=[groups[i] for i in train_index])

        # Get the best hyperparameters found in the inner loop
        best_params = grid_search.best_params_

        # Fit the model with the best hyperparameters on the training data
        model = LogisticRegression( **best_params)


        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average='binary') # change to binary
        outer_scores.append(score)

    # Compute and print the mean F1 score across all outer folds
    mean_f1 = np.round(100*np.mean(outer_scores), 1)
    return model, mean_f1


def plot_top_performance(e, importance_dict):

  '''
  Loads and resamples audio files

  Parameters
  ------------
  e: string
    The emotion for which the analysis is being done
  importance_dict: dictionary
    The dictionary containing the top dimensions of Hybrid BYOL-S for every emotion

  Returns
  ------------
  x_val: int
    The size of the minimal set of deimensions of Hybrid BYOL-S for best performance
  max_performance: int
    The max performance of the top dimensions of Hybrid BYOL-S 

  '''

  byols_embeddings, byols_labels, byols_speakers = emotion_all_merger(emotion=e, other_emotions=[em for em in emotions if em!=e], embeddings_dict=byols_embeddings_dict, speakers_dict=byols_speakers_list_dict, num=45)
  top_scores = []
  bottom_scores = []
  rest_scores = []

  for n in range(10, 1024, 10):

    top_features = importance_dict[e][:n]
    best_model, top_f1 = classification_pipeline(embeddings=byols_embeddings[:, top_features], speakers=byols_speakers, labels_list=byols_labels, feature_names=byols_feature_names)
    top_scores.append(top_f1)

    rest_features = importance_dict[e][n+1:]
    best_model, rest_f1 = classification_pipeline(embeddings=byols_embeddings[:, rest_features], speakers=byols_speakers, labels_list=byols_labels, feature_names=byols_feature_names)
    rest_scores.append(rest_f1)

    bottom_features = importance_dict[e][2048-n-1:]
    best_model, bottom_f1 = classification_pipeline(embeddings=byols_embeddings[:, bottom_features], speakers=byols_speakers, labels_list=byols_labels, feature_names=byols_feature_names)
    bottom_scores.append(bottom_f1)

  n_features = range(10, 1024, 10)
  fig, ax = plt.subplots()

  ax.plot(n_features, top_scores, label='Top dimensions')
  ax.plot(n_features, rest_scores, label='Rest dimensions')
  ax.plot(n_features, bottom_scores, label='Bottom dimensions')

  ax.set_xlabel('Number of dimensions')
  ax.set_ylabel('F1 score')
  ax.set_title(e)
  ax.legend()

  max_performance = max(top_scores)

  ax.set_ylim(0, 100)
  x_val = n_features[np.argmax(top_scores)]
  ax.scatter(x_val, max_performance, marker='x', color='red')

  return x_val, max_performance


