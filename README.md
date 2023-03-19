# AMPTransformer

AMPTransformer classifies peptides as antimicrobial/non-antimicrobial. It uses fine-tuned protein NLP models in combination with physicochemical descriptors to classify a FASTA file of peptide sequences. See example.FASTA for an example of this file format. A more detailed description of the model is provided in the Description section, below the usage instructions.

# Dependencies
* python >= 3.9
* biopython 1.81
* numpy 1.21.6
* torch 1.13.1
* pandas 1.5.0
* transformers 4.26.1
* peptides 0.3.1
* catboost 1.1.1
* lightgbm 3.3.5
* xgboost 1.6.2
* joblib 1.2.0
* tokenizers 0.13.2
* autogluon 0.7.0
* propy3 1.1.1
* tqdm 4.65.0

# Download NLP Models

Before installation, it is necessary to download the pretrained ProtBERT and ESM_V2 protein NLP models.
They can be downloaded from the following Google Drive:

https://drive.google.com/drive/folders/1HFZvBG0VWW2kO6uJqLSIFCs_TISwjGa-?usp=share_link

**Note: The total file size is approximately 30 GB, and the files will need to be unzipped prior to use.**

The files can also be found as a dataset on kaggle.

https://www.kaggle.com/datasets/brendanmoore14/amptransformer-models

# Installation

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

An example from google colab:

```
predict('/content/AMPTransformer/example.fasta', '/content/protbert_models', '/content/esm_models')

```

The output of this prediction is a dataframe with three columns. The first column is the peptide label in the fasta file, the second is the sequence, and the third is the antimicrobial prediction. The prediction is a number between 0 and 1, with values above 0.5 being antimicrobial, and those below 0.5 being non-antimicrobial.

```
prediction_dataframe = predict('/content/AMPTransformer/example.fasta', '/content/protbert_models', '/content/esm_models')

```
**For prediction of many peptide sequences, it is recommended to use a GPU.**
# Description

AMPTransformer is an antimicrobial peptide classifier trained on the length-balanced training datasets used by amPEPpy. These datasets were selected because of the equal peptide length distribution for the antimicrobial and non-antimicrobial peptides.

# References
