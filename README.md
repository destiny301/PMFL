# Partial Meta-Federated Learning

## Results
1. randomly select 5 diseases from 8 diseases(all of them are about lung) as training tasks of MAML, and use the trained model to train another disease
2. algorithm difference:
- w/o FL: train normally, without any pretraining process
- w/ FL: with traditional Federated Learning algorithm pretraining
- MetaFL: with metaFL algorithm(the combination of meta-learning and federated learning) pretraining
- PMFL: freeze part(like, 50%) parameters of the model when training metaFL
4. run 5 times, and show the mean and std in the images
----

![image](https://github.com/destiny301/PMFL/blob/main/result/01Atelectasis_rocauc.png)
![image](https://github.com/destiny301/PMFL/blob/main/result/01Consolidation_rocauc.png)
![image](https://github.com/destiny301/PMFL/blob/main/result/01LungLesion_rocauc.png)
![image](https://github.com/destiny301/PMFL/blob/main/result/01LungOpacity_rocauc.png)
![image](https://github.com/destiny301/PMFL/blob/main/result/01PleuralEffusion_rocauc.png)
![image](https://github.com/destiny301/PMFL/blob/main/result/01PleuralOther_rocauc.png)
![iamge](https://github.com/destiny301/PMFL/blob/main/result/01Pneumonia_rocauc.png)
![image](https://github.com/destiny301/PMFL/blob/main/result/01Pneumothorax_rocauc.png)
