# Explaining Deep Learning Embeddings for Speech Emotion Recognition
This GitHub repository contains code and resources for the paper titled **"Explaining Deep Learning Embeddings for Speech Emotion Recognition by Predicting Interpretable Acoustic Variables"**. In this project, we explore the use of deep learning embeddings for speech emotion recognition (SER) and propose a method to explain the underlying acoustic features captured by these embeddings.

## Paper's abstract
Speech emotion recognition (SER) is used for many applications including mental health assessments. Pre-trained self-supervised deep learning embeddings have shown superior performance over traditional handcrafted features. Explaining these audio deep learning representations is key to improve trust and advance the science of what acoustic information characterizes each emotion in speech such as high or low fundamental frequency or loudness. We first demonstrate that a deep learning embedding (Hybrid BYOL-S) outperforms a standard handcrafted feature set (eGeMAPS) using Ridge regression on the EmoDB dataset. To understand what acoustic information is used by the deep learning embedding model we probe it by predicting interpretable handcrafted feature values from the most important deep learning embedding dimensions for classifying a given emotion. This method allows us to (a) verify whether the features known to be relevant in classifying an emotion are represented within the top embedding dimensions, and (b) challenge data-driven findings that use handcrafted features that state that certain features seem to be required for detecting an emotion by showing a better performing model does not depend on that information. We provide a tutorial so this approach can be leveraged on other datasets or embeddings. 

## Repository Structure
This repository is organized as follows: 
- ```SER_explainability_tutorial.ipynb```: A Jupyter notebook providing a step-by-step tutorial for replicating the study's results and conducting the interpretability analysis.
- ```requirements.txt```: A file containing the required Python dependencies for running the code in the notebook.
- ```LICENSE```: The license file governing the use and distribution of the code and resources in this repository.
- ```README.md```: The readme file you are currently reading.
- ```import_functions.py```: A file with the functions to import and preprocess datasets.
- ```classification_functions.py```: A file with the functions for SER classification and calculating feature importance.
- ```prediction_functions.py```:  A file with the functions for predicting eGeMAPS features from some or all dimensions of Hybrid BYOL-S embeddings


## Getting Started
To get started with the code and replicate our experiments, follow these steps:


1. Clone this repository to your local machine: ```git clone https://github.com/satvik-dixit/explainability_ICASSP_2024.git ```

2. ```conda create -n ser_env python=3.9 ```

3. ```pip install -r requirements.txt ```

4. Explore the Jupyter notebook ```SER_explainability_tutorial.ipynb``` for a detailed tutorial on the experiments described in the paper. 

## Citation
Hopefully, a paper will be published soon.

# Supplementary Results

Scores: 
- feature importance (absolute standardized coefficient from logistic regression)
- Pearson's r between true labels and predicted labels using all Hybrid BYOL-S dimensions
- Pearson's r between true labels and predicted labels using top Hybrid BYOL-S dimensions

Anger:
| Feature                       | Scores               |
|-------------------------------|----------------------|
| slopeUV0-500\_sma3nz\_amean   | [0.093, 0.878, 0.901] |
| loudness\_sma3\_meanRisingSlope | [0.084, 0.573, 0.79]  |
| hammarbergIndexV\_sma3nz\_amean | [0.083, 0.938, 0.975] |
| alphaRatioV\_sma3nz\_amean     | [0.077, 0.948, 0.956] |
| mfcc2V\_sma3nz\_amean          | [0.07, 0.933, 0.939]  |

Fear:
| Feature                                   | Scores               |
|-------------------------------------------|----------------------|
| shimmerLocaldB\_sma3nz\_stddevNorm         | [0.11, 0.694, 0.554] |
| F0semitoneFrom27.5Hz\_sma3nz\_stddevNorm   | [0.099, 0.549, 0.467] |
| loudness\_sma3\_percentile50.0             | [0.087, 0.738, 0.909] |
| loudness\_sma3\_meanFallingSlope          | [0.082, 0.573, 0.549] |
| logRelF0-H1-H2\_sma3nz\_amean             | [0.082, 0.593, 0.911] |

Neutrality:
| Feature                      | Scores               |
|------------------------------|----------------------|
| jitterLocal\_sma3nz\_amean   | [0.301, 0.536, 0.677] |
| StddevUnvoicedSegmentLength   | [0.29, 0.862, 0.893]  |
| mfcc1\_sma3\_amean            | [0.285, 0.7, 0.898]   |
| loudnessPeaksPerSec           | [0.279, 0.562, 0.703] |
| F2frequency\_sma3nz\_amean    | [0.245, 0.322, 0.848] |

Joy:
| Feature                                    | Scores               |
|--------------------------------------------|----------------------|
| loudness\_sma3\_meanFallingSlope           | [1.903, 0.633, 0.795] |
| F0semitoneFrom27.5Hz\_sma3nz\_pctlrange0-2 | [1.699, 0.284, 0.697] |
| HNRdBACF\_sma3nz\_amean                    | [1.658, 0.904, 0.872] |
| equivalentSoundLevel\_dBp                 | [1.397, 0.655, 0.726] |
| StddevVoicedSegmentLengthSec               | [1.178, 0.507, 0.184] |

Sadness:
| Feature                        | Scores                |
|--------------------------------|-----------------------|
| MeanUnvoicedSegmentLength       | [0.389, 0.564, 0.918] |
| slopeV500-1500\_sma3nz\_amean   | [0.358, -0.029, 0.711]|
| StddevUnvoicedSegmentLength     | [0.356, 0.654, 0.783] |
| hammarbergIndexV\_sma3nz\_amean | [0.352, 0.741, 0.953] |
| F3frequency\_sma3nz\_amean      | [0.346, 0.712, 0.685] |

Disgust:
| Feature                       | Scores                |
|-------------------------------|-----------------------|
| F1frequency\_sma3nz\_stddevNorm | [1.923, 0.674, 0.369] |
| loudness\_sma3\_meanFallingSlope | [1.731, 0.719, 0.724] |
| mfcc4\_sma3\_amean             | [1.422, 0.825, 0.953] |
| HNRdBACF\_sma3nz\_stddevNorm   | [1.414, 0.695, 0.324] |
| loudnessPeaksPerSec            | [1.199, 0.592, 0.628] |


# Questions and Issues
If you have any questions or encounter any issues while using this repository, please feel free [to open an issue](https://github.com/satvik-dixit/explainability_ICASSP_2024/issues). We are here to assist you.

Thank you for your interest in our research, and we hope this repository proves valuable in your exploration of deep learning embeddings for speech emotion recognition and interpretability analysis.
