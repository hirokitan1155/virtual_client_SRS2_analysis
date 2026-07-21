# SRS-2 Prediction Analysis from Psychotherapy Dialogue

This repository contains the analysis code for predicting Social Responsiveness Scale-2 (SRS-2) scores from psychotherapy dialogue data using natural language processing (NLP) and machine learning approaches.

## Overview

This project investigates whether conversational features extracted from psychotherapy sessions can be used to estimate individual differences in social communication characteristics.

The analysis includes:

- Lexical similarity features (e.g., Jaccard similarity, cosine similarity)
- Semantic similarity features using Sentence-BERT embeddings
- Turn-level interaction features
- Lexical richness measures (MTLD)
- LLM-based extraction of conversational agendas
- Machine learning-based prediction of SRS-2 scores

## Methods

The analysis pipeline includes:

1. Dialogue preprocessing
2. Linguistic feature extraction
3. Semantic embedding analysis
4. Statistical correlation analysis
5. Machine learning prediction with cross-validation

## Data Availability

The psychotherapy dialogue dataset used in this study is not publicly available due to ethical and privacy restrictions.

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
