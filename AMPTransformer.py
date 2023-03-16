# dependencies
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
    get_cosine_schedule_with_warmup,
    BertModel,
    BertTokenizer,
    EsmModel,
    EsmTokenizer,
    AutoModel,
)
import gc
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import tokenizers
import time
import math
import string
import random
import joblib
from joblib import dump, load
import warnings
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
import Bio
from Bio import SeqIO
import propy
from propy import PyPro
import peptides
import catboost
from catboost import CatBoostClassifier
import lightgbm
from lightgbm import LGBMClassifier
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import FeatureMetadata
from autogluon.tabular import TabularDataset, TabularPredictor
import xgboost
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Mean pooling the output of the finetuned protein transformer models
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # adds a new dimension after the last index of the attention_mask, then expands it to the size of the last hidden state.
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        # clamps all elements in the sum_mask such that the minimum value is 1e-9, making any zeroes a small number instead.
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        # returns the mean of the embeddings
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


# Pytorch Model for Protein Sequence Classification
class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        # configuration for the model, loads the model specified in the config section at the top of the notebook.
        self.config = torch.load(config_path)
        self
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # returns the mean of the embeddings, as specified in the MeanPooling() function.
        self.pool = MeanPooling()
        # linear layer, reduces the dimenzionality of the hidden_size (in this case 1024) specified in the config file,
        # and reduces it to the pca_dim specified in the config file (In this case 64)
        self.fc1 = nn.Linear(self.config.hidden_size, self.cfg.pca_dim)
        # second linear layer, reduces the dimensonality from the pca_dim*6 (6, for 6 features) to 1 (dTm) nn.Linear changed to nn.Softmax for classification
        self.fc2 = nn.Linear(self.cfg.pca_dim, 1)
        # initializes weights of the linear layers
        self._init_weights(self.fc1)
        self._init_weights(self.fc2)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, batch):
        outputs = self.model(batch["input_ids"], batch["attention_mask"])
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, batch["attention_mask"])
        return feature

    def forward(self, batch):
        feature = self.fc1(self.feature(batch))
        output = self.fc2(feature)
        return output


# prepare the sequence input for prediction by the pretrained protbert model
def prepare_input(cfg, text):
    if "esm" in cfg.model:
        tokenizer = AutoTokenizer.from_pretrained("esm_models/tokenizer/")
    else:
        tokenizer = AutoTokenizer.from_pretrained("protbert_models/tokenizer/")
    inputs = tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=cfg.max_len,
        pad_to_max_length=True,
        truncation=True,
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


# test dataset
class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.text = df["sequence"].values
        if "esm" in cfg.model:
            tokenizer = AutoTokenizer.from_pretrained("esm_models/tokenizer/")
        else:
            tokenizer = AutoTokenizer.from_pretrained("protbert_models/tokenizer/")

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        # prepares the text inputs for use in the bert model (tokenize, pad, truncate, encode)
        inputs1 = prepare_input(self.cfg, self.text[item])
        # gets the labels for each item
        # returns a dict of all inputs
        return {
            "input_ids": inputs1["input_ids"],
            "attention_mask": inputs1["attention_mask"],
        }


# separate amino acids in the protein sequence by spaces.
def add_spaces(x):
    return " ".join(list(x))


# load the fasta sequence file as a dataframe
def load_test(file_path):
    with open(file_path) as fasta_file:
        identifiers = []
        lengths = []
        seq = []
        for seq_record in SeqIO.parse(fasta_file, "fasta"):  # (generator)
            seq.append(str(seq_record.seq))
            identifiers.append(seq_record.id)
            lengths.append(len(seq_record.seq))
    test = pd.DataFrame()
    test["sequence"] = seq
    test["len"] = lengths
    test["identifiers"] = identifiers
    test.drop_duplicates(inplace=True)
    test.reset_index(drop=True, inplace=True)
    test["sequence"] = test.sequence.map(add_spaces)
    return test


# predict antimicrobial vs non-antimicrobial on each sequence batch and append to preds
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for batch in tk0:
        for k, v in batch.items():
            batch[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(batch)
        preds.append(y_preds.to("cpu").numpy())
    predictions = np.concatenate(preds)
    return predictions


# classify as antimicrobial or non-antimicrobial using fine-tuned transformer models


def protbert_prediction(file_path):
    # define the path and output directory
    Output_Directory = f"protbert_models/"
    if not os.path.exists(Output_Directory):
        os.makedirs(Output_Directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # configuration for the protbert model
    class CFG:
        num_workers = 1
        model = "Rostlab/prot_bert"
        batch_size = 32
        max_len = 100
        folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        pca_dim = 256

    # Tokenizer for protbert
    tokenizer = AutoTokenizer.from_pretrained("protbert_models/tokenizer/")
    CFG.tokenizer = AutoTokenizer.from_pretrained("protbert_models/tokenizer/")
    # path to config file
    CFG.path = Output_Directory
    CFG.config_path = CFG.path + "config.pth"
    # load the fasta file
    test = load_test(file_path)
    # protbert predict the fasta sequences
    def pred(test):
        test_dataset = TestDataset(CFG, test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        predictions_ = []
        for fold in CFG.folds:
            model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
            state = torch.load(
                CFG.path + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                map_location=torch.device("cpu"),
            )
            model.load_state_dict(state["model"])
            prediction = inference_fn(test_loader, model, device)
            predictions_.append(prediction)
            del model, state, prediction
            gc.collect()
            torch.cuda.empty_cache()
        predictions = np.mean(predictions_, axis=0)
        test["predictions"] = predictions
        return predictions

    test["predictions"] = pred(test)
    test["protbert"] = torch.sigmoid(torch.tensor(test["predictions"]))
    # save a csv for the protbert predictions
    test.to_csv("protbert_prediction")
    return test["protbert"]


def esm_prediction(file_path):
    # define the path and output directory
    Output_Directory = f"esm_models/"
    if not os.path.exists(Output_Directory):
        os.makedirs(Output_Directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # configuration for the esm model
    class CFG:
        num_workers = 1
        model = "facebook/esm2_t33_650M_UR50D"
        gradient_checkpointing = False
        batch_size = 32
        max_len = 100
        folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        pca_dim = 256

    # Tokenizer for esm
    tokenizer = AutoTokenizer.from_pretrained("esm_models/tokenizer/")
    CFG.tokenizer = AutoTokenizer.from_pretrained("esm_models/tokenizer/")
    # path to config file
    CFG.path = Output_Directory
    CFG.config_path = CFG.path + "config.pth"
    # load the fasta file
    test = load_test(file_path)

    def pred(test):
        test_dataset = TestDataset(CFG, test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        predictions_ = []
        for fold in CFG.folds:
            model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
            state = torch.load(
                CFG.path + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                map_location=torch.device("cpu"),
            )
            model.load_state_dict(state["model"])
            prediction = inference_fn(test_loader, model, device)
            predictions_.append(prediction)
            del model, state, prediction
            gc.collect()
            torch.cuda.empty_cache()
        predictions = np.mean(predictions_, axis=0)
        test["predictions"] = predictions
        return predictions

    # protbert predict the fasta sequences
    test["predictions"] = pred(test)
    test["esm"] = torch.sigmoid(torch.tensor(test["predictions"]))
    # save a csv for the protbert predictions
    test.to_csv("esm_prediction")
    return test["esm"]


def peptide_descriptors(test_df):
    test_peptide = pd.DataFrame(
        [peptides.Peptide(s).descriptors() for s in test_df["sequence"]]
    )
    test_propy = pd.DataFrame([propy.CTD.CalculateCTD(s) for s in test_df["sequence"]])
    test_df = pd.concat([test_df, test_peptide], axis=1)
    test_df = pd.concat([test_df, test_propy], axis=1)
    return test_df


def ensemble_predict(test_df):
    autogluon_folds = 15
    xgb_folds = 20
    cat_folds = 20
    lgb_folds = 20
    EXCLUDE = ["sequence", "len", "identifiers"]
    FEATURES = [c for c in test_df.columns if c not in EXCLUDE]
    FEATURES_GB = FEATURES.copy()
    FEATURES_GB.remove("labels")
    # autogluon prediction, take mean of all 15 models
    model = TabularPredictor.load(f"./autogluon_models_reg/fold0/")
    pred_autogluon = model.predict_proba(test_df[FEATURES])
    for f in range(1, autogluon_folds):
        model = TabularPredictor.load(f"./autogluon_models_reg/fold{f}/")
        pred_autogluon += model.predict_proba(test_df[FEATURES])
    pred_autogluon /= autogluon_folds
    # xgboost prediction, mean of all 20 models
    xgb = XGBClassifier()
    xgb.load_model(f"xgb_models/XGB_fold0.JSON")
    pred_xgb = xgb.predict_proba(
        test_df[FEATURES_GB], iteration_range=(0, xgb.best_iteration + 1)
    )[:, 1]
    for f in range(1, xgb_folds):
        xgb.load_model(f"xgb_models/XGB_fold{f}.JSON")
        pred_xgb += xgb.predict_proba(
            test_df[FEATURES_GB], iteration_range=(0, xgb.best_iteration + 1)
        )[:, 1]
    pred_xgb = pred_xgb / xgb_folds
    # catboost prediciton, mean of all 20 models
    cat = load(f"cat_models/CAT_fold0.joblib")
    pred_cat = cat.predict_proba(test_df[FEATURES_GB])[:, 1]
    for f in range(1, cat_folds):
        load(f"cat_models/CAT_fold{f}.joblib")
        pred_cat += cat.predict_proba(test_df[FEATURES_GB])[:, 1]
    pred_cat = pred_cat / cat_folds
    # lgbm prediction, mean of all 20 models
    lgb = load(f"lgb_models/lgb_fold0.joblib")
    pred_lgb = lgb.predict_proba(test_df[FEATURES_GB])[:, 1]
    for f in range(1, lgb_folds):
        load(f"lgb_models/lgb_fold{f}.joblib")
        pred_lgb += lgb.predict_proba(test_df[FEATURES_GB])[:, 1]
    pred_lgb = pred_lgb / lgb_folds
    test_df["autogluon_predictions"] = pred_autogluon[1]
    test_df["xgb_predictions"] = pred_xgb
    test_df["cat_predictions"] = pred_cat
    test_df["lgb_predictions"] = pred_lgb
    # the final prediction is the average of the autogluon, xgboost, catboost, and lgbm predictions.
    test_df["mean_predictions"] = (
        0.25 * pred_autogluon[1] + 0.25 * pred_xgb + 0.25 * pred_cat + 0.25 * pred_lgb
    )
    return test_df[["identifiers", "sequence", "mean_predictions"]]


def predict(file_path):
    test = load_test(file_path)
    print("Loaded Fasta File")
    print("Now Performing Protbert Inference, 10 Inferences Total")
    test["protbert"] = protbert_prediction(file_path)
    print("Protbert Inference Complete")
    print("Now Performing ESM Inference, 10 Inferences Total")
    test["esm"] = esm_prediction(file_path)
    print("ESM Inference Complete")
    test["labels"] = np.zeros
    print("Calculating Peptide Descriptors")
    feature_dataframe = peptide_descriptors(test)
    print("Providing Final Predictions")
    prediction_dataframe = ensemble_predict(feature_dataframe)
    return prediction_dataframe
