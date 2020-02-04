import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import torch
print(torch.__version__)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pickle
import time
import warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import BertModel, BertPreTrainedModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader

MAX_LEN = 256
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
#LEARNING_RATE_MODEL = 1e-5
#LEARNING_RATE_CLASSIFIER = 1e-3
# MAX_GRAD_NORM = 1.0
EARLY_STOPPING_ROUNDS = 2
NUM_MODELS = 3
MODEL_PATH = "models/bert_{}".format(time.strftime('%Y%m%d%H%M'))
os.mkdir(MODEL_PATH)

train = pd.read_csv('data/train_preprocessed.csv')
test = pd.read_csv('data/test_preprocessed.csv')
train['comment_text'].fillna("", inplace=True)
test['comment_text'].fillna("", inplace=True)
classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
X_train_valid_raw, y_train_valid = train['comment_text'].str.lower(), train[classes].values
X_test_raw = test['comment_text'].str.lower()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_train_valid = np.array(list(map(lambda x: tokenizer.encode(x, max_length=MAX_LEN, pad_to_max_length=True), X_train_valid_raw)))
X_test = np.array(list(map(lambda x: tokenizer.encode(x, max_length=MAX_LEN, pad_to_max_length=True), X_test_raw)))

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


pred = np.zeros((X_test.shape[0], 6))
for i in range(NUM_MODELS):
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.1, random_state=i, shuffle=True)
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.long)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.long)
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_data = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    test_data = TensorDataset(X_test)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)
    model = torch.nn.DataParallel(model)
    model.to(device)

    #optimizer_grouped_parameters = [
    #        {"params": model.module.bert.parameters(), "lr": LEARNING_RATE_MODEL},
    #        {"params": model.module.classifier.parameters(), "lr": LEARNING_RATE_CLASSIFIER}
    #    ]
    #optimizer = AdamW(optimizer_grouped_parameters)
    optimizer_grouped_parameters = [
            {"params": model.module.parameters(), "lr": LEARNING_RATE}
        ]
    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=int(len(train_loader) * 0.5),
                    num_training_steps=len(train_loader) * NUM_EPOCHS)
    total_step = len(train_loader)
    best_score = -np.inf
    best_epoch = None
    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, (cur_X_train, cur_y_train) in enumerate(train_loader):
            cur_X_train = cur_X_train.to(device)
            cur_y_train = cur_y_train.to(device)
            outputs = model(cur_X_train)
            loss = nn.BCEWithLogitsLoss()(outputs[0], cur_y_train)
            model.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            if (i + 1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))
        model.eval()
        predictions = []
        with torch.no_grad():
            correct = 0
            total = 0
            for (cur_X_valid, _) in tqdm(valid_loader):
                cur_X_valid = cur_X_valid.to(device)
                outputs = model(cur_X_valid)
                predictions.append(outputs[0].cpu())
        predictions = np.vstack(predictions)
        valid_score = roc_auc_score(np.array(y_valid), predictions)
        print("epoch {} valid score {}".format(epoch, valid_score))
        save_file_name = "epoch_{}_valid_{:.4f}.ckpt".format(epoch, valid_score)
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, save_file_name))
        if valid_score > best_score:
            best_score = valid_score
            best_epoch = epoch
        elif epoch - best_epoch >= EARLY_STOPPING_ROUNDS:
            break

    save_file_name = "epoch_{}_valid_{:.4f}.ckpt".format(best_epoch, best_score)
    print("best model:", save_file_name)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, save_file_name)))
    model.eval()
    predictions = []
    with torch.no_grad():
        correct = 0
        total = 0
        for (cur_X_test,) in tqdm(test_loader):
            cur_X_test = cur_X_test.to(device)
            outputs = model(cur_X_test)
            predictions.append(outputs[0].cpu())
    pred += np.vstack(predictions)

pred /= NUM_MODELS
submission = pd.read_csv("data/sample_submission.csv")
submission[classes] = pred
submission.to_csv("submission/v7_bert.csv.gz", compression="gzip", index=False)