## GPT and Interpolation-based Data Augmentation for Multiclass Intrusion Detection in IIoT


<div align="center">

  [Reviewed Article (soon to be published in IEEE Access)](https://ieeeaccess.ieee.org) | [Dataset](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot) 
 
</div>

### Description

The Industrial Internet of Things (IIoT) leverages interconnected devices for data collection, monitoring, and analysis in industrial processes. Despite its benefits, IIoT introduces cybersecurity vulnerabilities due to inadequate security protocols. This paper focuses on intrusion detection in IIoT networks, addressing challenges of limited and imbalanced datasets.

Prior works have proposed Machine Learning (ML) for intrusion detection in IIoT, with ML models reliant on diverse and representative training data. Limited datasets and class imbalance hinder model generalization, emphasizing the need for data augmentation.

<div align="center">
  <img src="assets/workflow_v4.png" width="300px" alt="Workflow">
  <p><em>Figure 1: Workflow with alternative scenarios for IIoT traffic data augmentation and classification evaluation.</em></p>
</div>

Data augmentation involves creating artificial data to address imbalances. In image domains, transformations like flipping and scaling are common. In tabular data, methods like SMOTE generate synthetic samples. Recent works, such as REalTabFormer and GReaT, explore GPT-based models for generating realistic tabular data.

## TL;DR

The evaluation employs IIoT traffic data, comparing performance across multiple scenarios.

Results reveal varied impacts on different algorithms. GPT-based methods generate data with class-specific feature value issues, leading to performance degradation. XGBoost remains indifferent to data augmentation.

Intrusion detection solutions show differing responses to data augmentation. Random Forest benefits, Tabnet exhibits uncertain behavior, and XGBoost remains largely unaffected. GPT-based methods generate invalid data, impacting classification performance.

<div align="center">
    <img src="assets/results_table.png" width="300px" alt="results">
    <p><em>Figure 2: Comparative Results of Multiclass Classification Performance Using Macro Average (%).</em></p>
</div>

This work highlights the nuanced impact of data augmentation on intrusion detection in IIoT. GPT-based methods may introduce challenges, emphasizing the importance of systematic evaluation. XGBoost, a top-performing algorithm, shows limited improvement with data augmentation. 

### Repository structure

```
dataAugmentationTests/                  
├── notebooks/             # Jupyter notebooks
│   ├── 1_data_analysis_<augmentation_method>.ipynb     # Data analysis
│   ├── 2_<augmentation_method>_augmentation.ipynb      # Data augmentation
│   ├── 3_<augmentation_method>_evaluation.ipynb        # Models evaluation
│   └── ...                       
├── src/                   # Source code
│   ├── utils.py           # Utility functions
│   └── ...                
├── results/               # Output files
│   ├── metrics/           # Evaluation metrics (*.csv)
│   ├── conf_matrix/       # Confusion matrices (*.csv)
│   └── ...                
├── data/                  # Placeholder for input data
├── old_repo/              # Previous repository backup
├── assets/                # Images and other assets
│
├── .gitignore             # To be ignored by git
├── README.md              # Project README file
└── requirements.txt       # Dependencies file

```


### Acknowledgements

This work is partially funded by FCT - Fundação para a Ciência e a Tecnologia, I.P., through projects UIDB/04524/2020, and under the Scientific Employment Stimulus - Institutional Call - CEECINST/00051/2018, and by ANI - Agência Nacional de Inovação, S.A., through project POCI-01-0247-FEDER-046083.


<hr style="height:1px; background-color:grey; border:none;">

<p align="center">
<img src="assets/CIIC_logo.png" width="700px"/>
</p>

<hr style="height:1px; background-color:grey; border:none;">

<p align="center">
<img src="assets/fundo_financiamento.png" width="700px"/>
</p>

<hr style="height:1px; background-color:grey; border:none;">