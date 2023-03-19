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

**Note: The total file size is approximately 30 GB, and the files will need to be unzipped prior to use.**

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
2. Set the current working directory to 

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

AMPTransformer was evaluated on eight non-redundant test datasets published by Yan et al.(5,6). The predictor's performance compared with amPEPpy, a random forest model trained on the same training dataset, is shown in the table below (3). As the test sets are not length-balanced, and are non-redundant only for amPEPpyy, the performance of the AMPTransformer on the independent test sets cannot be compared directly with other methods.

# References

1. Elnaggar, A. et al., ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 7112-7127, 1 Oct. 2022, doi: 10.1109/TPAMI.2021.3095381.
2. Lin, Z. et al., Evolutionary-scale Prediction of Atomic Level Protein Structure With a Language Model, bioRxiv 2022.07.20.500902; doi: doi: 10.1101/2022.07.20.500902.
3. Lawrence, TJ. et al. amPEPpy 1.0: a Portable and Accurate Antimicrobial Peptide prediction tool. Bioinformatics. 2021 Aug 4;37(14):2058-2060. doi: 10.1093/bioinformatics/btaa917.
4. Bhadra, P. et al. AmPEP: Sequence-based Prediction of Antimicrobial Peptides Using Distribution Patterns of Amino Acid Properties and Random Forest. Sci Rep 8, 1697 (2018). doi: 10.1038/s41598-018-19752-w.
5. Yan, K. et al. sAMPpred-GAT: Prediction of Antimicrobial Peptide by Graph Attention Network and Predicted Peptide Structure. Bioinformatics. 2023 Jan 1;39(1):btac715. doi: 10.1093/bioinformatics/btac715.
6. Xu, J. et al. Comprehensive assessment of machine learning-based methods for predicting antimicrobial peptides, Briefings in Bioinformatics, Volume 22, Issue 5, September 2021, bbab083, doi: 10.1093/bib/bbab083.

