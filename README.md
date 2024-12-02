# Shadow-Stereotype-Superposition (SSS) Gender Bias Measurement Metric

## Requirements
The Requirements to test our metric can be found in the `requirements.txt` file

## Obtaining the Reported Results for ShadowBias<sub>SocialGroups</sub> Metric
- **Wasserstein Distance Computation**: The code for obtaining the Wasserstein Distance of our ShadowBias<sub>SocialGroups</sub> can be accessed through `SSS_WassersteinDistance.py` file.

## Obtaining the Reported Results for ShadowBias<sub>Stereotype</sub> Metric
- **Accuracy Computation**: The code for obtaining the Accuracy Results of our ShadowBias<sub>Stereotype</sub> can be accessed through `SSS_Accuracy.py` file.

Both files contains the code to test our metric across WinoBias, Crows-Pairs, and RedditBias datasets.

## Parameters:
- All these files take in 3 parameters: `model_name`, `Dataset path`, `Dataset Name`.
- Modify the `model_name` parameter to obtain the results for your preferred model.
- Modify the `Dataset Name` and `Dataset path` parameters to obtain the results for your preferred dataset.

## Generating Biased Models
We also provide the code for generating biased models. It can be accessed in the path: `Biased_Models\`.

## Training Biased Models Using ChatGPT generations
- The code for using ChatGPT generations to create a dimensionally stereotyped dataset can be accessed in the path: `Biased_Models\Dimensionally Stereotyped\`.
- First, execute the code in `Generate_dataset.py` to obtain the ChatGPT generations. This file contains a method named `generate_dataset`, and it takes in your OpenAI API_KEY to obtain the generations from ChatGPT.
- It will store the generated in 6 datasets, based on Category and Dimension.
- Then, execute the `BiasedModels.py` file to train biased models using these datasets

## Acknowledgements
This repository utilizes data from the following Repositories
- [Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods](https://github.com/uclanlp/corefBias)
- [CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models](https://github.com/nyu-mll/crows-pairs)
- [RedditBias: A Real-World Resource for Bias Evaluation and Debiasing of Conversational Language Models](https://github.com/umanlp/RedditBias)

The complete list of terms for Sociability, Morality, and Status dimensions have been obtained using code from the following Repository
- [ Comprehensive stereotype content dictionaries using a semi-automated method](https://github.com/gandalfnicolas/SADCAT)


