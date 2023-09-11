# Explaining Deep Learning Embeddings for Speech Emotion Recognition
This GitHub repository contains code and resources for the paper titled **"Explaining Deep Learning Embeddings for Speech Emotion Recognition by Predicting Interpretable Acoustic Variables"**, presented at _ICASSP 2024_. In this project, we explore the use of deep learning embeddings for speech emotion recognition (SER) and propose a method to explain the underlying acoustic features captured by these embeddings.

## Paper's abstract
Speech emotion recognition (SER) is used for many applications including mental health assessments. Pre-trained self-supervised deep learning embeddings have shown superior performance over traditional handcrafted features. Explaining these audio deep learning representations is key to improve trust and advance the science of what acoustic information characterizes each emotion in speech such as high or low fundamental frequency or loudness. We first demonstrate that a deep learning embedding (Hybrid BYOL-S) outperforms a standard handcrafted feature set (eGeMAPS) using Ridge regression on the EmoDB dataset. To understand what acoustic information is used by the deep learning embedding model we probe it by predicting interpretable handcrafted feature values from the most important deep learning embedding dimensions for classifying a given emotion. This method allows us to (a) verify whether the features known to be relevant in classifying an emotion are represented within the top embedding dimensions, and (b) challenge data-driven findings that use handcrafted features that state that certain features seem to be required for detecting an emotion by showing a better performing model does not depend on that information. We provide a tutorial so this approach can be leveraged on other datasets or embeddings. 

## Repository Structure
This repository is organized as follows: 
- ``` SER_explainability_tutorial.ipynb```: A Jupyter notebook providing a step-by-step tutorial for replicating the study's results and conducting the interpretability analysis.
- ``` requirements.txt```: A file containing the required Python dependencies for running the code in the notebook.
- ``` LICENSE```: The license file governing the use and distribution of the code and resources in this repository.
- ``` README.md```: The readme file you are currently reading.

## Getting Started
To get started with the code and replicate our experiments, follow these steps:


1. Clone this repository to your local machine: ``` git clone https://github.com/satvik-dixit/explainability_ICASSP_2024.git ```

2. ``` conda create -n ser_env python=3.9 ```

3. ``` pip install -r requirements.txt ``` <TODO @satvik: please edit the _requirements.txt_ file>

4. Explore the Jupyter notebook ```SER_explainability_tutorial.ipynb``` for a detailed tutorial on the experiments described in the paper. 

## Citation
Hopefully, a paper will be published soon :) 

# Questions and Issues
If you have any questions or encounter any issues while using this repository, please feel free [to open an issue](https://github.com/satvik-dixit/explainability_ICASSP_2024/issues). We are here to assist you.

Thank you for your interest in our research, and we hope this repository proves valuable in your exploration of deep learning embeddings for speech emotion recognition and interpretability analysis.
