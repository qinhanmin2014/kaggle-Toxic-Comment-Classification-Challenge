# kaggle-Toxic-Comment-Classification-Challenge

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

- [TFIDF](https://nbviewer.jupyter.org/github/qinhanmin2014/kaggle-San-Francisco-Crime-Classification/blob/master/version1_TFIDF.ipynb)

| method | cv score | public leaderboard | private leaderboard |
| ------ | -------- | ------------------ | ------------------- |
| TFIDF+LR | 0.97820 | 0.97409 | 0.97423 |

- [Word2Vec](https://nbviewer.jupyter.org/github/qinhanmin2014/kaggle-San-Francisco-Crime-Classification/blob/master/version2_Word2Vec.ipynb)

| method | cv score | public leaderboard | private leaderboard |
| ------ | -------- | ------------------ | ------------------- |
| Word2Vec+google | 0.96361 | 0.95277 | 0.95227 |
| Word2Vec+glove | 0.96747 | 0.95629 | 0.95782 |
| Word2Vec+fasttext | 0.96907 | 0.95780 | 0.95839 |

- [RNN](https://nbviewer.jupyter.org/github/qinhanmin2014/kaggle-San-Francisco-Crime-Classification/blob/master/version3_RNN.ipynb)

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

- [BiGRU&BiLSTM](https://nbviewer.jupyter.org/github/qinhanmin2014/kaggle-San-Francisco-Crime-Classification/blob/master/version4_BiGRU&BiLSTM.ipynb)

| method | validation set | public leaderboard | private leaderboard |
| ------ | -------------- | ------------------ | ------------------- |
| BiGRU&BiLSTM+glove | 0.98977 | 0.98333 | 0.98308 |
| BiGRU&BiLSTM+fasttext | 0.98982 | 0.98360 | 0.98309 |

- [MultiBiRNN](https://nbviewer.jupyter.org/github/qinhanmin2014/kaggle-San-Francisco-Crime-Classification/blob/master/version5_MultiBiRNN.ipynb)

| method | validation set | public leaderboard | private leaderboard |
| ------ | -------------- | ------------------ | ------------------- |
| MultiBiGRU+glove | 0.98949 | 0.98369 | 0.98280 |
| MultiBiGRU+fasttext | 0.98986 | 0.98327 | 0.98301 |
| MultiBiLSTM+glove | 0.98955 | 0.98222 | 0.98246 |
| MultiBiLSTM+fasttext | 0.99019 | 0.98426 | 0.98326 |
