# kaggle-Toxic-Comment-Classification-Challenge

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

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

- [MultiBiGRU](https://nbviewer.jupyter.org/github/qinhanmin2014/kaggle-Toxic-Comment-Classification-Challenge/blob/master/version5_MultiBiRNN_GRU.ipynb),
[MultiBiLSTM](https://nbviewer.jupyter.org/github/qinhanmin2014/kaggle-Toxic-Comment-Classification-Challenge/blob/master/version5_MultiBiRNN_LSTM.ipynb),
[MultiBiLSTM_GRU](https://nbviewer.jupyter.org/github/qinhanmin2014/kaggle-Toxic-Comment-Classification-Challenge/blob/master/version5_MultiBiRNN_GRU_LSTM.ipynb)
(* means repeat the program 5 times and take the average)

| method | validation set | public leaderboard | private leaderboard |
| ------ | -------------- | ------------------ | ------------------- |
| MultiBiGRU+glove+preprocess(\*) | | 0.98610 | 0.98534 |
| MultiBiGRU+fasttext+preprocess(\*) | | 0.98607 | 0.98575 |
| MultiBiLSTM+glove+preprocess(\*) | | 0.98600 | 0.98518 |
| MultiBiLSTM+fasttext+preprocess(\*) | | 0.98608 | 0.98540 |
| MultiBiLSTM_GRU+glove+preprocess(\*) | | 0.98503 | 0.98508 |
| MultiBiLSTM_GRU+fasttext+preprocess(\*) | | 0.98624 | 0.98561 |

- Combine different models

| models | public leaderboard | private leaderboard |
| ------ | ------------------ | ------------------- |
| BiGRU&BiLSTM(\*) | 0.98606 | 0.98545 |
| MultiBiRNN(\*) | 0.98644 | 0.98598 |
| BiGRU&BiLSTM(\*)+MultiBiRNN(\*) | 0.98649 | 0.98595 |
