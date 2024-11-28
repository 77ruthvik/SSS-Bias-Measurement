# Shadow-Stereotype-Superposition (SSS) Gender Bias Measurement Metric

## Requirements
The Requirements to test our metric can be found in the `requirements.txt` file

## Obtaining the Reported Results for SSS (Double-slit) Metric
- **Wasserstein Distance Computation**: The code for obtaining the Wasserstein Distance of our SSS (Double-slit) can be accessed through `SSS_WassersteinDistance.py` file.
- **Plots**: The code for obtaining the plots for our SSS (Double-slit) can be accessed through `SSS_WassersteinDistance_plots.py` file.

## Obtaining the Reported Results for SSS (Single-slit) Metric
- **Accuracy Computation**: The code for obtaining the Accuracy Results of our SSS (Single-slit) can be accessed through `SSS_Accuracy.py` file.
- **Plots**: The `SSS_Accuracy.py` file also returns the plots for our SSS (Single-slit) Metric.

## Parameters:
- All these files take in 2 parameters: `model_name` and `WinoBias Datasets path`. Modify the `model_name` parameter to obtain the results for your preferred model.

## Generating Biased Models
We also provide the code for generating biased models. It can be accessed in the path: `Biased_Models\`.

## Training Biased Models Using RedditBias
- The code for generating the biased models using RedditBias datasets is in the path: `Biased_Models\RedditBias\Training.py`.
- This file uses gender bias data from RedditBias to finetune the models and create their biased versions.

## Training Biased Models Using ChatGPT generations
- The code for using ChatGPT generations to create a dimensionally stereotyped dataset can be accessed in the path: `Biased_Models\Dimensionally Stereotyped\`.
- First, execute the code in `Generate_dataset.py` to obtain the ChatGPT generations. This file contains a method named `generate_dataset`, and it takes in your OpenAI API_KEY to obtain the generations from ChatGPT
- Then, execute the `BiasedModels.py` file to train biased models using this dataset

## Acknowledgements
This repository utilizes data from the following GitHub Repository
- [RedditBias: A Real-World Resource for Bias Evaluation and Debiasing of Conversational Language Models](https://github.com/umanlp/RedditBias)

