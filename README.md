# CallGAT: Graph Attention Temporal Transformer for Volatility Prediction
## Project Overview
CallGAT is a multimodal machine learning framework designed to predict stock volatility from earnings calls. It combines textual information from transcripts, acoustic features from audio recordings, and graph-based relationships between companies to generate accurate multi-horizon volatility forecasts (3-day, 7-day, 15-day, and 30-day).

The model leverages three key components:

Audio-Text Fusion: Captures vocal cues and linguistic patterns from earnings calls

Graph Relations: Models inter-company relationships based on sector, industry, and known connections

Temporal Processing: Analyzes historical volatility patterns across multiple timeframes


## Clone the repository
git clone https://github.com/yourusername/CallGAT.git

cd CallGAT

## Create and activate virtual environment
python -m venv callgat_env

source callgat_env/bin/activate  
#### On Windows: 
callgat_env\Scripts\activate

## Install dependencies
pip install -r requirements.txt

## Execution Flow
The model requires three types of input data:

Earnings call audio recordings

Earnings call transcripts

Company relationship data (sectors, industries, wiki-relationships)

### Audio Feature Pipeline

The audio pipeline extracts two sets of features from earnings call recordings:

Praat-based Prosodic Features (26 features)

MFCC Features (26 features)

13 Mel-frequency cepstral coefficients

13 Delta MFCC features (capturing temporal dynamics)

These features are compressed using an autoencoder to create a 64-dimensional representation.


### Generate text embeddings
run the python src/earnings-call_extraction/finbert_line_embedding.py
followed by the ./combine_pkl.py

This helps in generating the 768 dimensional text embeddings using the FinBERT.

### Notebook Execution
Import the files to the notebook and run the notebook

## References

Sawhney, R., Khanna, P., Aggarwal, A., Jain, T., Mathur, P., & Shah, R. (2020). VolTAGE: Volatility forecasting via text-audio fusion with graph convolution networks for earnings calls. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8001-8013.
