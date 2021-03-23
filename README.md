# MAML-Federated-Learning

## Results
1. randomly select 5 diseases from 8 diseases(all of them are about lung) as training tasks of MAML, and use the trained model to train another disease
2. algorithm difference:
- w/o maml: train normally, without any MAML
- w/ maml: with normal MAML algorithm
- part-freeze maml: freeze part(like, 50%) parameters of the model when training MAML
4. run 5 times, and show the mean and std in the images
----

![image](https://github.com/destiny301/PMFL/blob/main/result/RoundNotIncludeMAML/01Atelectasis.png)
![image](https://github.com/destiny301/PMFL/blob/main/result/RoundNotIncludeMAML/01Consolidation.png)
![image](https://github.com/destiny301/PMFL/blob/main/result/RoundNotIncludeMAML/01LungLesion.png)
![image](https://github.com/destiny301/PMFL/blob/main/result/RoundNotIncludeMAML/01LungOpacity.png)
![image](https://github.com/destiny301/PMFL/blob/main/result/RoundNotIncludeMAML/01PleuralEffusion.png)
![image](https://github.com/destiny301/PMFL/blob/main/result/RoundNotIncludeMAML/01PleuralOther.png)
![iamge](https://github.com/destiny301/PMFL/blob/main/result/RoundNotIncludeMAML/01Pneumonia.png)
![image](https://github.com/destiny301/PMFL/blob/main/result/RoundNotIncludeMAML/01Pneumothorax.png)
