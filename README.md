# AMPTransformer

AMPTransformer classifies peptides as antimicrobial/non-antimicrobial. It uses fine-tuned protein NLP models in combination with physicochemical descriptors to classify a FASTA file of peptide sequences. See example.fasta for an example of this file format. A more detailed description of the model is provided in the Description section, below the usage instructions.

# Dependencies
* python >= 3.9
* numpy 1.21.6
* pandas 1.5.0
* torch 1.13.1
* transformers 4.26.1
* tokenizers 0.13.2
* biopython 1.81
* peptides 0.3.1
* propy3 1.1.1
* scikit-learn 1.1.2
* catboost 1.1.1
* lightgbm 3.3.5
* xgboost 1.6.2
* autogluon 0.7.0
* joblib 1.2.0
* tqdm 4.65.0

# Download NLP Models

Before installation, it is necessary to download the pretrained ProtBERT (1) and ESM_V2 (2) protein NLP models.
They can be downloaded from the following Google Drive:

https://drive.google.com/drive/folders/1HFZvBG0VWW2kO6uJqLSIFCs_TISwjGa-?usp=share_link

You can add a shortcut to the folder in your Google Drive by right-clicking on the "nlp_models" folder title, and selecting "Add Shortcut to Drive".

**Note: There are zipped and unzipped versions of the files. The zipped versions will need to be unzipped prior to use.**

The files can also be found as a dataset on kaggle.

https://www.kaggle.com/datasets/brendanmoore14/amptransformer-models

# Google Colaboratory Notebook

The predictor can be run using the following Google Colaboratory notebook:

https://colab.research.google.com/drive/1HE0n7tk53NVMpdCURF_JFmIsNESn2g-v?usp=sharing

**Please make a copy of the notebook so that your changes can be saved.**

# Installation Guide

1. Clone the GitHub repository.

```
git clone https://github.com/Brendan-P-Moore/AMPTransformer

```
2. Set the current working directory to the AMPTransformer folder.

```
cd AMPTransformer

```
3. Install the required packages listed in the dependencies section and requirements file.

```
pip install -r requirements.txt

```
4. Import the AMPTransformer python file, and the predict function.

```
import AMPTransformer
from AMPTransformer import predict

```
# Predict
To predict the antimicrobial nature of the peptide sequences contained within a fasta file, we can run the following function. The path to the fasta file and model folders must be the absolute path.

**Peptide sequences should be between 5 and 100 amino acids in length. The ES**

```
predict(path_to_the_fasta_file, path_to_the_protbert_file_folder, path_to_the_esm_file_folder)

```

An example from the Google Colaboratory notebook:

```
predict('/content/AMPTransformer/test_data/CAMP_test_neg.fasta', '/content/drive/MyDrive/nlp_models/protbert_models', '/content/drive/MyDrive/nlp_models/esm_models')

```

The output of this prediction is a dataframe with three columns. The first column is the peptide label in the fasta file, the second is the sequence, and the third is the antimicrobial prediction. The prediction is a number between 0 and 1, with values above 0.5 being antimicrobial, and those below 0.5 being non-antimicrobial.

```
prediction_dataframe = predict('/content/AMPTransformer/test_data/CAMP_test_neg.fasta', '/content/drive/MyDrive/nlp_models/protbert_models', '/content/drive/MyDrive/nlp_models/esm_models')

```
**For prediction of many peptide sequences, it is recommended to use a GPU.**
# Description

AMPTransformer is an antimicrobial peptide classifier trained on the length-balanced training datasets used by amPEPpy (3,4). These datasets were selected because of the equal peptide sequence length distribution for the antimicrobial and non-antimicrobial peptides (3,4). Because of this, AMPTransformer is more useful at discerning between antimicrobial/non-antimicrobial peptides of equal length. Other predictors should be used if the peptides of interest have large variation in sequence length.

AMPTransformer was evaluated on eight non-redundant test datasets published by Yan et al., and details of the test sets are described therein (5,6). The predictor's performance compared with amPEPpy, a random forest model trained on the same training dataset, is shown in the table below (3). As the test sets are not length-balanced, and are non-redundant only for amPEPpyy, the performance of the AMPTransformer on the independent test sets cannot be compared directly with other methods.

| Test datasets                |          |          |           |        |       |       |   |                     |      |   |
|------------------------------|----------|----------|-----------|--------|-------|-------|---|---------------------|------|---|
| XUAMP                        | Accuracy | F1 score | Precision | Recall | MCC   | AUC   |   | number of proteins: | 3072 |   |
| amPEPpy                      | 0.636    | 0.523    | 0.760     | 0.398  | 0.310 | 0.714 |   |                     |      |   |
| AMPTransformer               | 0.635    | 0.502    | 0.790     | 0.368  | 0.320 | 0.743 |   |                     |      |   |
|                              |          |          |           |        |       |       |   |                     |      |   |
| DBAASP                       |          |          |           |        |       |       |   | number of proteins: | 356  |   |
| amPEPpy                      | 0.767    | 0.743    | 0.828     | 0.674  | 0.543 | 0.870 |   |                     |      |   |
| AMPTransformer               | 0.744    | 0.700    | 0.848     | 0.596  | 0.512 | 0.858 |   |                     |      |   |
|                              |          |          |           |        |       |       |   |                     |      |   |
| dbAMP                        |          |          |           |        |       |       |   | number of proteins: | 440  |   |
| amPEPpy                      | 0.802    | 0.774    | 0.903     | 0.677  | 0.624 | 0.878 |   |                     |      |   |
| AMPTransformer               | 0.834    | 0.820    | 0.897     | 0.755  | 0.677 | 0.922 |   |                     |      |   |
|                              |          |          |           |        |       |       |   |                     |      |   |
| LAMP                         |          |          |           |        |       |       |   | number of proteins: | 1750 |   |
| amPEPpy                      | 0.728    | 0.662    | 0.874     | 0.533  | 0.495 | 0.826 |   |                     |      |   |
| AMPTransformer               | 0.750    | 0.700    | 0.878     | 0.582  | 0.532 | 0.865 |   |                     |      |   |
|                              |          |          |           |        |       |       |   |                     |      |   |
| DRAMP                        |          |          |           |        |       |       |   | number of proteins: | 2364 |   |
| amPEPpy                      | 0.690    | 0.597    | 0.852     | 0.459  | 0.428 | 0.720 |   |                     |      |   |
| AMPTransformer               | 0.665    | 0.537    | 0.868     | 0.388  | 0.395 | 0.718 |   |                     |      |   |
|                              |          |          |           |        |       |       |   |                     |      |   |
| CAMP                         |          |          |           |        |       |       |   | number of proteins: | 44   |   |
| amPEPpy                      | 0.841    | 0.829    | 0.895     | 0.773  | 0.688 | 0.856 |   |                     |      |   |
| AMPTransformer               | 0.773    | 0.737    | 0.875     | 0.636  | 0.567 | 0.886 |   |                     |      |   |
|                              |          |          |           |        |       |       |   |                     |      |   |
| APD3                         |          |          |           |        |       |       |   | number of proteins: | 354  |   |
| amPEPpy                      | 0.884    | 0.881    | 0.905     | 0.859  | 0.769 | 0.924 |   |                     |      |   |
| AMPTransformer               | 0.864    | 0.860    | 0.886     | 0.836  | 0.730 | 0.927 |   |                     |      |   |
|                              |          |          |           |        |       |       |   |                     |      |   |
| YADAMP                       |          |          |           |        |       |       |   | number of proteins: | 226  |   |
| amPEPpy                      | 0.894    | 0.889    | 0.932     | 0.850  | 0.791 | 0.950 |   |                     |      |   |
| AMPTransformer               | 0.881    | 0.873    | 0.930     | 0.823  | 0.766 | 0.957 |   |                     |      |   |
|                              |          |          |           |        |       |       |   |                     |      |   |
| Weighted Overall Performance |          |          |           |        |       |       |   | total proteins:     | 8606 |   |
| amPEPpy                      | 0.702    | 0.619    | 0.830     | 0.501  | 0.439 | 0.769 |   |                     |      |   |
| AMPTransformer               | 0.698    | 0.602    | 0.845     | 0.479  | 0.440 | 0.788 |   |                     |      |   |

# References

1. Elnaggar, A. et al., ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 7112-7127, 1 Oct. 2022, doi: 10.1109/TPAMI.2021.3095381.
2. Lin, Z. et al., Evolutionary-scale Prediction of Atomic Level Protein Structure With a Language Model, bioRxiv 2022.07.20.500902; doi: doi: 10.1101/2022.07.20.500902.
3. Lawrence, TJ. et al. amPEPpy 1.0: a Portable and Accurate Antimicrobial Peptide prediction tool. Bioinformatics. 2021 Aug 4;37(14):2058-2060. doi: 10.1093/bioinformatics/btaa917.
4. Bhadra, P. et al. AmPEP: Sequence-based Prediction of Antimicrobial Peptides Using Distribution Patterns of Amino Acid Properties and Random Forest. Sci Rep 8, 1697 (2018). doi: 10.1038/s41598-018-19752-w.
5. Yan, K. et al. sAMPpred-GAT: Prediction of Antimicrobial Peptide by Graph Attention Network and Predicted Peptide Structure. Bioinformatics. 2023 Jan 1;39(1):btac715. doi: 10.1093/bioinformatics/btac715.
6. Xu, J. et al. Comprehensive assessment of machine learning-based methods for predicting antimicrobial peptides, Briefings in Bioinformatics, Volume 22, Issue 5, September 2021, bbab083, doi: 10.1093/bib/bbab083.

# FAQ

To be updated as questions arise.
