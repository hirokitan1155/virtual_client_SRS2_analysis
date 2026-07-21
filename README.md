# Virtual Client SRS-2 Analysis

This repository contains the analysis code for investigating the relationship between psychotherapy dialogue features and Social Responsiveness Scale-2 (SRS-2) scores using natural language processing (NLP) and machine learning approaches.

## Overview



This project investigates whether conversational features extracted from psychotherapy sessions can be used to estimate individual differences in social communication characteristics.



The analysis is part of the Virtual Client research framework, which aims to support the assessment and training of social communication skills through AI-based conversational agents.



The analysis includes:



- Lexical similarity features

  - Jaccard similarity

  - Cosine similarity



- Semantic similarity features

  - Sentence-BERT embeddings

  - Turn-level semantic distance



- Linguistic features

  - Word frequency analysis

  - Lexical richness measures (MTLD)



- LLM-assisted analysis

  - Extraction of conversational agendas from patient utterances



- Machine learning analysis

  - Prediction of SRS-2 scores

  - Cross-validation evaluation

  - Feature importance analysis



## Methods



The analysis pipeline includes the following steps:



1. Dialogue preprocessing

2. Linguistic feature extraction

3. Semantic embedding analysis

4. Conversational interaction analysis

5. Statistical correlation analysis

6. Machine learning-based prediction with cross-validation



## Repository Structure



```



virtual_client_SRS2_analysis/



├── analysis.py

│   Main analysis script



├── requirements.txt

│   Python package dependencies



└── README.md

Project documentation



````



## Data Availability



The psychotherapy dialogue dataset used in this study is not publicly available due to ethical and privacy restrictions.



The analysis code is provided to facilitate methodological transparency and reproducibility.



Researchers interested in applying this framework should use their own ethically approved datasets.



## Requirements



Python 3.11+



Main dependencies:



- pandas

- numpy

- scipy

- scikit-learn

- sentence-transformers

- janome

- lexicalrichness

- openai

- torch

- matplotlib



Install dependencies:



```bash

pip install -r requirements.txt

````



## Usage



Run the analysis script:



```bash

python analysis.py

```



The script performs:



1. Loading demographic and dialogue data

2. Dialogue feature extraction

3. Semantic similarity computation using Sentence-BERT

4. LLM-based conversational agenda extraction

5. Correlation analysis with SRS-2 and K6 scores

6. Machine learning prediction of SRS-2 scores

7. Feature importance analysis and visualization



## Configuration



Before running the analysis, prepare the required data paths in `analysis.py`.



The analysis requires:



* Participant-level demographic data including SRS-2 and K6 scores

* Speaker-separated psychotherapy dialogue transcripts



For LLM-based agenda extraction, set your OpenAI API key as an environment variable:



```bash

export OPENAI_API_KEY="your_api_key"

```



The Sentence-BERT model:



```

intfloat/multilingual-e5-large

```



will be automatically downloaded when running the analysis for the first time.



## Output



The analysis generates:



* Correlation results between dialogue features and clinical measures

* Machine learning prediction performance

* Feature importance scores

* Visualization files



Example outputs include:



```

K6_SRS2_scatter.pdf

SRS2_prediction_scatter.pdf

```



## Citation



If you use this code, please cite the corresponding publication.



```

Citation information will be added after publication.

```



## License



MIT License
