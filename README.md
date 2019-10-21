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

- [BiGRU&BiLSTM](https://nbviewer.jupyter.org/github/qinhanmin2014/kaggle-Toxic-Comment-Classification-Challenge/blob/master/version4_BiGRU%26BiLSTM.ipynb)
(* means repeat the program 5 times and take the average)

| method | validation set | public leaderboard | private leaderboard |
| ------ | -------------- | ------------------ | ------------------- |
| BiGRU&BiLSTM+glove | 0.98977 | 0.98333 | 0.98308 |
| BiGRU&BiLSTM+glove+preprocess | 0.99018 | 0.98397 | 0.98385 |
| BiGRU&BiLSTM+glove+preprocess(\*) | | 0.98538 | 0.98497 |
| BiGRU&BiLSTM+fasttext | 0.98982 | 0.98360 | 0.98309 |
| BiGRU&BiLSTM+fasttext+preprocess | 0.99042 | 0.98496 | 0.98468 |
| BiGRU&BiLSTM+fasttext+preprocess(\*) | | 0.98551 | 0.98522 |


- [MultiBiRNN](https://nbviewer.jupyter.org/github/qinhanmin2014/kaggle-Toxic-Comment-Classification-Challenge/blob/master/version5_MultiBiRNN.ipynb)
(* means repeat the program 5 times and take the average)

| method | validation set | public leaderboard | private leaderboard |
| ------ | -------------- | ------------------ | ------------------- |
| MultiBiGRU+glove | 0.98949 | 0.98369 | 0.98280 |
| MultiBiGRU+glove+preprocess | 0.99059 | 0.98464 | 0.98388 |
| MultiBiGRU+glove+preprocess(\*) | | 0.98555 | 0.98498 |
| MultiBiGRU+fasttext | 0.98986 | 0.98327 | 0.98301 |
| MultiBiGRU+fasttext+preprocess | 0.98930 | 0.98397 | 0.98362 |
| MultiBiGRU+fasttext+preprocess(\*) | | 0.98565 | 0.98528 |
| MultiBiLSTM+glove | 0.98955 | 0.98222 | 0.98246 |
| MultiBiLSTM+glove+preprocess | 0.99040 | 0.98445 | 0.98356 |
| MultiBiLSTM+glove+preprocess(\*) | | 0.98557 | 0.98507 |
| MultiBiLSTM+fasttext | 0.99019 | 0.98426 | 0.98326 |
| MultiBiLSTM+fasttext+preprocess | 0.99040 | 0.98522 | 0.98493 |
| MultiBiLSTM+fasttext+preprocess(\*) | | 0.98581 | 0.98539 |


- Combine different models

| models | public leaderboard | private leaderboard |
| ------ | ------------------ | ------------------- |
| BiGRU&BiLSTM + MultiBiRNN | 0.98574 | 0.98554 |
| BiGRU&BiLSTM(\*) + MultiBiRNN(\*) | 0.98601 | 0.98566 |
