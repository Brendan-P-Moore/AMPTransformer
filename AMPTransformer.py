# general libraries
import numpy as np
import pandas as pd
import gc
import os

# for the NLP models
import transformers  # for NLP protein models
from transformers import (
    AutoTokenizer,
    AutoModel,
)
import tokenizers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# tracking progress and loading models
from tqdm.auto import tqdm
import joblib
from joblib import dump, load

# loading fasta files and protein physicochemical descriptors
import Bio
from Bio import SeqIO
import propy
from propy import PyPro
import peptides

# training the final model
import catboost
from catboost import CatBoostClassifier
import lightgbm
from lightgbm import LGBMClassifier
import xgboost
from xgboost import XGBClassifier
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import FeatureMetadata
from autogluon.tabular import TabularDataset, TabularPredictor

# configuration class for NLP models
class CFG:
    num_workers = 1
    gradient_checkpointing = False
    batch_size = 1
    max_len = 100
    folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    pca_dim = 256


# Mean pooling the output of the finetuned protein transformer models
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # adds a new dimension after the last index of the attention mask, then expands it to the size of the last hidden state.
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        # clamps all elements in the sum_mask such that the minimum value is 1e-9, changing zeroes to a small number.
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
        self.model = AutoModel.from_config(self.config)
        # returns the mean of the embeddings, as specified in the MeanPooling() function.
        self.pool = MeanPooling()
        # linear layer, reduces the dimenzionality of the hidden_size to the pca_dim specified in the CFG class
        self.fc1 = nn.Linear(self.config.hidden_size, self.cfg.pca_dim)
        # second linear layer, reduces the dimensonality from the pca_dim to 1
        self.fc2 = nn.Linear(self.cfg.pca_dim, 1)
        # initializes weights of the linear layers
        self._init_weights(self.fc1)
        self._init_weights(self.fc2)

    # weight initialization function for the linear layers
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def feature(self, batch):
        outputs = self.model(batch["input_ids"], batch["attention_mask"])
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, batch["attention_mask"])
        return feature

    def forward(self, batch):
        first_linear_output = self.fc1(self.feature(batch))
        final_output = self.fc2(first_linear_output)
        return final_output


# prepare the sequences input for prediction by the finetuned NLP models
def prepare_input(cfg, text):
    if "esm" in cfg.model:
        tokenizer = AutoTokenizer.from_pretrained(CFG.path + "tokenizer")
    else:
        tokenizer = AutoTokenizer.from_pretrained(CFG.path + "tokenizer")
    inputs = tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=cfg.max_len,
        padding="max_length",
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

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        # prepares the text inputs for use in the bert model (tokenize, pad, truncate, encode)
        inputs1 = prepare_input(self.cfg, self.text[item])
        # returns a dict of all inputs
        return {
            "input_ids": inputs1["input_ids"],
            "attention_mask": inputs1["attention_mask"],
        }


# separate amino acids in the protein sequence by spaces.
def add_spaces(x):
    return " ".join(list(x))


# load the fasta sequence file as a dataframe
def load_fasta(file_path):
    with open(os.path.abspath(file_path)) as fasta_file:
        peptides = []
        seq = []
        for seq_record in SeqIO.parse(fasta_file, "fasta"):  # (generator)
            seq.append(str(seq_record.seq))
            peptides.append(seq_record.id)
    test = pd.DataFrame()
    test["sequence"] = seq
    test["peptides"] = peptides
    test.drop_duplicates(inplace=True)
    test.reset_index(drop=True, inplace=True)
    return test


# predict antimicrobial vs non-antimicrobial on each sequence batch and append to preds
def inference(test_loader, model, device):
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


def nlp_predict(test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test["sequence"] = test.sequence.map(add_spaces)
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
        prediction = inference(test_loader, model, device)
        predictions_.append(prediction)
        del model, state, prediction
        gc.collect()
        torch.cuda.empty_cache()
    predictions = np.mean(predictions_, axis=0)
    test["predictions"] = predictions
    return predictions


# classify as antimicrobial or non-antimicrobial using fine-tuned transformer models


def protbert_predict(file_path, protbert_folder_path):
    # define the path and output directory
    try:
        CFG.path = os.path.abspath(protbert_folder_path) + "/"
    except:
        print("Must download protbert model files, see github instructions for details")
    # configuration for the protbert model
    CFG.model = "Rostlab/prot_bert"
    # path to config file
    CFG.config_path = CFG.path + "config.pth"
    # load the fasta sequence file
    test = load_fasta(file_path)
    # classify peptides using pretrained protbert model
    protbert_pred = torch.sigmoid(torch.tensor(nlp_predict(test)))
    return protbert_pred


def esm_predict(file_path, esm_folder_path):
    # define the path and output directory
    try:
        CFG.path = os.path.abspath(esm_folder_path) + "/"
    except:
        print("Must download ESM model files, see github instructions for details")
    # esm model
    CFG.model = "facebook/esm2_t33_650M_UR50D"
    # path to model configuration file
    CFG.config_path = CFG.path + "config.pth"
    # load fasta file
    test = load_fasta(file_path)
    # classify peptides using pretrained protbert model
    esm_pred = torch.sigmoid(torch.tensor(nlp_predict(test)))
    return esm_pred


# calculating the physicochemical peptide descriptors
def generate_peptide_descriptors(test_df):
    test_peptide = pd.DataFrame(
        [peptides.Peptide(s).descriptors() for s in test_df["sequence"]]
    )
    test_propy = pd.DataFrame([propy.CTD.CalculateCTD(s) for s in test_df["sequence"]])
    test_df = pd.concat([test_df, test_peptide], axis=1)
    test_df = pd.concat([test_df, test_propy], axis=1)
    return test_df


def ensemble_predict(test_df):
    # folds used to train models.
    autogluon_model_count = 15
    gb_model_count = 20
    test_df["labels"] = np.zeros  # an empty column of labels that autogluon expects.
    EXCLUDE = ["sequence", "peptides"]
    FEATURES = [c for c in test_df.columns if c not in EXCLUDE]
    FEATURES_GB = FEATURES.copy()
    FEATURES_GB.remove(
        "labels"
    )  # gradient boosted models do not expect a "labels" feature

    # autogluon prediction, take mean of all 15 models
    model = TabularPredictor.load(f"pretrained_models/autogluon_models_reg/fold0/")
    pred_autogluon = model.predict_proba(test_df[FEATURES])
    for f in range(1, autogluon_model_count):
        model = TabularPredictor.load(
            f"pretrained_models/autogluon_models_reg/fold{f}/"
        )
        pred_autogluon += model.predict_proba(test_df[FEATURES])
    pred_autogluon /= autogluon_model_count

    # xgboost prediction, mean of all 20 models
    xgb = XGBClassifier()
    xgb.load_model(f"pretrained_models/xgb_models/XGB_fold0.JSON")
    pred_xgb = xgb.predict_proba(
        test_df[FEATURES_GB], iteration_range=(0, xgb.best_iteration + 1)
    )[:, 1]
    for f in range(1, gb_model_count):
        xgb.load_model(f"pretrained_models/xgb_models/XGB_fold{f}.JSON")
        pred_xgb += xgb.predict_proba(
            test_df[FEATURES_GB], iteration_range=(0, xgb.best_iteration + 1)
        )[:, 1]
    pred_xgb = pred_xgb / gb_model_count

    # catboost prediciton, mean of all 20 models
    cat = load(f"pretrained_models/cat_models/CAT_fold0.joblib")
    pred_cat = cat.predict_proba(test_df[FEATURES_GB])[:, 1]
    for f in range(1, gb_model_count):
        load(f"pretrained_models/cat_models/CAT_fold{f}.joblib")
        pred_cat += cat.predict_proba(test_df[FEATURES_GB])[:, 1]
    pred_cat = pred_cat / gb_model_count

    # lgbm prediction, mean of all 20 models
    lgb = load(f"pretrained_models/lgb_models/lgb_fold0.joblib")
    pred_lgb = lgb.predict_proba(test_df[FEATURES_GB])[:, 1]
    for f in range(1, gb_model_count):
        load(f"pretrained_models/lgb_models/lgb_fold{f}.joblib")
        pred_lgb += lgb.predict_proba(test_df[FEATURES_GB])[:, 1]
    pred_lgb = pred_lgb / gb_model_count
    # the final prediction is the average of the autogluon, xgboost, catboost, and lgbm predictions.
    test_df["Antimicrobial_Peptide_Prediction"] = (
        0.25 * pred_autogluon[1] + 0.25 * pred_xgb + 0.25 * pred_cat + 0.25 * pred_lgb
    )
    return test_df[["peptides", "sequence", "Antimicrobial_Peptide_Prediction"]]


def predict(fasta_file_path, protbert_folder_path, esm_folder_path):
    test = load_fasta(fasta_file_path)
    print("Calculating Protbert Predictions, 10 Total Models")
    test["protbert"] = protbert_predict(fasta_file_path, protbert_folder_path)
    print("Calculating ESM Predictions, 10 Total Models")
    test["esm"] = esm_predict(fasta_file_path, esm_folder_path)
    print("Calculating Peptide Descriptors")
    feature_dataframe = generate_peptide_descriptors(test)
    print("Calculating Final Antimicrobial Peptide Predictions")
    prediction_dataframe = ensemble_predict(feature_dataframe)
    return prediction_dataframe
