# kaggle-Toxic-Comment-Classification-Challenge

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

### Steps to reproduce (RNN + CNN + bert)

- Download the dataset from Kaggle and put it in data/
- Tensorflow 1.13.2, Keras 2.2.4
- run toxic-data-preprocessing.py (data preprocessing)
- run version5_*.ipynb (MultiBiGRU models)
- run version6_*.ipynb (CNN models)
- run model_blending.ipynb (model blending)
- run version7_bert.py (bert models)
- run model_blending_2.ipynb (model blending)
- public leaderboard 0.98770 (29/4544), private leaderboard 0.98747 (15/4544)

### Steps to reproduce (RNN + CNN)

- Download the dataset from Kaggle and put it in data/
- Tensorflow 1.13.2, Keras 2.2.4
- run toxic-data-preprocessing.py (data preprocessing)
- run version5_*.ipynb (MultiBiGRU models)
- run version6_*.ipynb (CNN models)
- run model_blending.ipynb (model blending)
- public leaderboard 0.98709 (325/4544), private leaderboard 0.98677 (174/4544)

### Experiments

- [TFIDF](https://nbviewer.jupyter.org/github/qinhanmin2014/kaggle-Toxic-Comment-Classification-Challenge/blob/master/version1_TFIDF.ipynb)

| method | cv score | public leaderboard | private leaderboard |
| ------ | -------- | ------------------ | ------------------- |
| TFIDF+LR | 0.97820 | 0.97409 | 0.97423 |

- [Word2Vec](https://nbviewer.jupyter.org/github/qinhanmin2014/kaggle-Toxic-Comment-Classification-Challenge/blob/master/version2_Word2Vec.ipynb)

| method | cv score | public leaderboard | private leaderboard |
| ------ | -------- | ------------------ | ------------------- |
| Word2Vec+google | 0.96361 | 0.95277 | 0.95227 |
| Word2Vec+glove | 0.96747 | 0.95629 | 0.95782 |
| Word2Vec+fasttext | 0.96907 | 0.95780 | 0.95839 |

- [RNN](https://nbviewer.jupyter.org/github/qinhanmin2014/kaggle-Toxic-Comment-Classification-Challenge/blob/master/version3_RNN.ipynb)

| method | validation set | public leaderboard | private leaderboard |
| ------ | -------------- | ------------------ | ------------------- |
| GRU | 0.98412 | 0.97692 | 0.97691 |
| GRU+google | 0.98588 | 0.97638 | 0.97743 |
| GRU+glove | 0.99002 | 0.98271 | 0.98210 |
| GRU+fasttext | 0.99000 | 0.98352 | 0.98287 |
| LSTM | 0.97919 | 0.96848 | 0.97106 |
| LSTM+google | 0.98475 | 0.97530 | 0.97585 |
| LSTM+glove | 0.98884 | 0.98191 | 0.98157 |
| LSTM+fasttext | 0.98908 | 0.98222 | 0.98128 |

- [BiGRU&BiLSTM](https://nbviewer.jupyter.org/github/qinhanmin2014/kaggle-Toxic-Comment-Classification-Challenge/blob/master/version4_BiGRU_BiLSTM.ipynb)
(* means repeat the program 5 times and take the average)

| method | validation set | public leaderboard | private leaderboard |
| ------ | -------------- | ------------------ | ------------------- |
| BiGRU&BiLSTM+glove+preprocess(\*) | | 0.98597 | 0.98536 |
| BiGRU&BiLSTM+fasttext+preprocess(\*) | | 0.98600 | 0.98536 |

- MultiBiGRU

| method | validation set | public leaderboard | private leaderboard |
| ------ | -------------- | ------------------ | ------------------- |
| MultiBiGRU+glove+preprocess+KFold | | 0.98604 | 0.98553 |
| MultiBiGRU+fasttext+preprocess+KFold | | 0.98634 | 0.98597 |
| MultiBiGRU+char+preprocess+KFold | | 0.98171 | 0.98119 |

- CNN

| method | validation set | public leaderboard | private leaderboard |
| ------ | -------------- | ------------------ | ------------------- |
| CNN+glove+preprocess+KFold | | 0.98550 | 0.98449 |
| CNN+fasttext+preprocess+KFold | | 0.98569 | 0.98517 |
| CNN+char+preprocess+KFold | | 0.97419 | 0.97256 | 

- Combine different models

| models | public leaderboard | private leaderboard |
| ------ | ------------------ | ------------------- |
| MultiBiGRU+CNN | 0.98652 | 0.98606 |
| MultiBiGRU+MultiBiGRU(char)+CNN | 0.98687 | 0.98655 |
| MultiBiGRU+MultiBiGRU(char)+CNN (weighted ensemble) | 0.98709 | 0.98677 |

### Solution based on bert

| method | validation set | public leaderboard | private leaderboard |
| ------ | -------------- | ------------------ | ------------------- |
| bert-based-uncased+preprocess | | 0.98653 | 0.98660 |

- Combine bert (sub2) with the best solution above (sub1)

| models | public leaderboard | private leaderboard |
| ------ | ------------------ | ------------------- |
| 0.5 * sub1 + 0.5 * sub2 | 0.98686 | 0.98690 |
| 0.9 * sub1 + 0.1 * sub2 | 0.98770 | 0.98747 |

### References

- https://github.com/zake7749/DeepToxic
