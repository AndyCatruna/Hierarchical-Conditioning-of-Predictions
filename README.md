# Hierarchical Conditioning of Predictions in Multi-Task Settings for Image-based Vehicle Analysis

## Prerequisites
Install all dependencies with the following command:
```pip install -r requirements.txt```

## Data

For Stanford Cars [1] and Autovit [2] datasets, contact the authors for the data.

For CarModelDataset, send an e-mail to andy_eduard.catruna@upb.ro.

[1] Krause, Jonathan, et al. "3d object representations for fine-grained categorization." Proceedings of the IEEE international conference on computer vision workshops. 2013.

[2] Dutulescu, Andreea, et al. "What is the Price of Your Used Car? Automated Predictions using XGBoost and Neural Networks." 2023 24th International Conference on Control Systems and Computer Science (CSCS). IEEE, 2023.

Each dataset should have a separate folder with ```train.csv``` and ```test.csv``` files and an ```images/``` folder.

The .csv files should contain the following columns: ```img_path, make_class, model_class, year_class, full_class``` where the classes are indices. 

## Training and Evaluation

This repository contains training and evaluation scripts for the *Swin* and *ViT* backbones with the experimental *Unified, Separate, and Conditional* prediction heads across the 3 datasets.

To run the experiments in the paper, utilize the following scripts:
- **CarModelDataset**
    - Run ```scripts/car_model_dataset_swin.sh``` for *Swin* models
    - Run ```scripts/car_model_dataset_vit.sh``` for *ViT* models

- **Stanford Cars**
    - Run ```scripts/stanford_swin.sh``` for *Swin* models
    - Run ```scripts/stanford_vit.sh``` for *ViT* models

- **Autovit**
    - Run ```scripts/autovit_swin.sh``` for *Swin* models
    - Run ```scripts/autovit_vit.sh``` for *ViT* models
    - Run ```scripts/autovit_tasks.sh``` for training on multiple tasks including *orientation*, *mileage*, and *price estimation*.

The weights of the pretrained top-performing models can be found [here](https://drive.google.com/drive/folders/1ekb0Yz_ZOuKDY_n8993M9mwjrIXSgdwo?usp=sharing).

## Citation

If you find our work useful and utilize the models or the dataset, please cite our paper:

```
TBD
```

## License & Acknowledgement
This work is released under the Apache 2.0 License (see LICENSE).

This work was funded by the “Automated car damage detection and cost prediction – InsureAI” project, Contract Number 30/221_ap3/22.07.2022, MySMIS code: 142909.