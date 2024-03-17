# PMFL: Partial Meta-Federated Learning for heterogeneous tasks and its applications on real-world medical records ([BigData2022](https://arxiv.org/html/2112.05321v2))

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/html/2112.05321v2)

## Abstract
Federated machine learning is a versatile and flexible tool to utilize distributed data from different sources, especially when communication technology develops rapidly and an unprecedented amount of data could be collected on mobile devices nowadays. Federated learning method exploits not only the data but the computational power of all devices in the network to achieve more efficient model training. Nevertheless, while most traditional federated learning methods work well for homogeneous data and tasks, adapting the method to heterogeneous data and task distribution is challenging. This limitation has constrained the applications of federated learning in real-world contexts, especially in healthcare settings. Inspired by the fundamental idea of meta-learning, in this study we propose a new algorithm, which is an integration of federated learning and meta-learning, to tackle this issue. In addition, owing to the advantage of transfer learning for model generalization, we further improve our algorithm by introducing partial parameter sharing to balance global and local learning. We name this method partial meta-federated learning (PMFL). Finally, we apply the algorithms to two medical datasets. We show that our algorithm could obtain the fastest training speed and achieve the best performance when dealing with heterogeneous medical datasets.

![image](https://github.com/destiny301/PMFL/blob/main/flowchart.png)

## Usage
1. randomly select 5 diseases from 8 diseases(all of them are about lung) as training tasks of MAML, and use the trained model to train another disease
2. algorithm difference:
- w/o FL: train normally, without any pretraining process
- w/ FL: with traditional Federated Learning algorithm pretraining
- MetaFL: with metaFL algorithm(the combination of meta-learning and federated learning) pretraining
- PMFL: freeze part(like, 50%) parameters of the model when training metaFL
4. run 5 times, and show the mean and std in the images

## Results

### Evaluate Pneumonia

<img src="https://github.com/destiny301/PMFL/blob/main/result/01Pneumonia_rocauc.png" width="400">

## Citation
If you use PMFL in your research or wish to refer to the results published here, please use the following BibTeX entry. Sincerely appreciate it!
```shell
@inproceedings{zhang2022pmfl,
  title={PMFL: Partial Meta-Federated Learning for heterogeneous tasks and its applications on real-world medical records},
  author={Zhang, Tianyi and Zhang, Shirui and Chen, Ziwei and Bengio, Yoshua and Liu, Dianbo},
  booktitle={2022 IEEE International Conference on Big Data (Big Data)},
  pages={4453--4462},
  year={2022},
  organization={IEEE}
}
```
