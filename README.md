## Assessing GPT-based Data Augmentation for Intrusion Detection in IIoT

---

<p align="center">
<img src="Assets/CIIC_logo.png" width="1000px"/>
</p>

---

### Description
This project focuses on data augmentation techniques for Industrial Internet of Things (IIoT) networks. IIoT involves interconnected devices for data collection, monitoring, and analysis in industrial or agricultural production. However, IIoT systems are vulnerable to cybersecurity attacks due to lacking security protocols. This project aims to explore effective data augmentation strategies for IIoT cybersecurity and assess their impact on classification accuracy, generalization, and different algorithms. 

Our investigation aims to explore specific aspects: (I) identify the most effective data augmentation strategies for IIoT cybersecurity contexts. Secondly, (II) analyze the contribution of data augmentation to assist in achieving more accurate classification on a towards achieving more accurate data classification. Moreover, we wish to (III) assess whether data augmentation leads to a more generalized model or if it results in overfitting specific instances. And finally, we aim to (IV) evaluate which algorithm produces the best classification results and determine the extent of its impact on different algorithms.

### Dataset Used

**EDGE-IIoTSET**
- [Paper][paper]
- [Full dataset][edge_full]
- [Sample dataset][edge_sample] (the one we used!)

[edge_full]: <https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications>
[edge_sample]: <https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot>
[paper]: <https://ieeexplore.ieee.org/document/9751703>

### Organization

Every test is in a different folder indicated by the name of the used algorithm.

- **decisionTree** - All tests using the Decision Tree algorithm
    - testNormal - with no augmentation
    - testRealTabFormer - using RealTabFormer generated data
    - testSmote - using Smote generated data
    - testSmoteNC - using SmoteNC generated data
- **dnn**
    - testNormal - with no augmentation
    - testRealTabFormer - using RealTabFormer generated data
    - testSmote - using Smote generated data
    - testSmoteNC - using SmoteNC generated data
- **randomForest**
    - testNormal - with no augmentation
    - testRealTabFormer - using RealTabFormer generated data
    - testSmote - using Smote generated data
    - testSmoteNC - using SmoteNC generated data
- **tabnet**
    - testNormal - with no augmentation
    - testRealTabFormer - using RealTabFormer generated data
    - testSmote - using Smote generated data
    - testSmoteNC - using SmoteNC generated data
- **xgBoost**
    - testNormal - with no augmentation
    - testRealTabFormer - using RealTabFormer generated data
    - testSmote - using Smote generated data
    - testSmoteNC - using SmoteNC generated data

Within each algorithm folder, there exist additional subfolders containing various test cases.
The algorithm was executed using the command "nohup python prepare.py" which ran the `prepare.py` file and redirected the output to the `nohup.out` file. This command generates some files such as:
Here is an improved list of files:
- `nohup.out`: Contains all the results from the model.
- `confusion_matrix.png`: Presents the confusion matrix.
- `feature_importances.jpg`: Illustrates the most important features and their respective importance.
- Additionally, there is typically a saved model file, which may have a different name for each run.



### Environment

The tests were performed on a MacBook Pro 16'' (2018) equipped with an Intel Core i7 2.2 GHz 6-Core processor and 16GB of 2400 MHz DDR4 RAM.
We utilized Anaconda to create a Python environment, specifically employing **Python version 3.9.6**. There are two available files for the environment: `environment.yml`, which does not contain build information, and `environment_with_builds.yml`, which includes build information.

To create the environment using Anaconda, you can execute the following command:

```sh
conda env create -f environment.yml python=3.9.6
```


### More information
- Name: Francisco Melícias
- Personal email: melicias1999@gmail.com
- Student email: francisco.s.melicias@ipleiria.pt
- [Github][github]
- [Linkedin][linkedin]
- [Paper][paper]


[github]: <https://github.com/Melicias>
[linkedin]: <https://www.linkedin.com/in/francisco-melicias/>
[paper]: <>