# Crowdflower_Search_Results
Crowdflower Search Results Relevance is a Kaggle Competition - https://www.kaggle.com/c/crowdflower-search-relevance/overview. The goal of this project is to create a machine learning model for measuring the relevance of search results. This model can be used to help eCommerce businesses to evaluate the performance of their search algorithms.

## Dataset
The dataset is taken from https://www.kaggle.com/c/crowdflower-search-relevance/data.

## Packages Required
1. pandas==1.1.3
2. tqdm==4.50.2
3. nltk==3.5
4. beautifulsoup4==4.9.3
5. gensim==3.8.3
6. fasttext==0.9.2
7. numpy==1.19.2
8. scikit_learn==0.24.1

## Results
Word2Vec word embeddings gave the best MCC (Matthews correlation coefficient) score of 0.285 with RandomForestClassifier.
GloVe and FastText word embeddings performed the best with ExtratreesClassifier and gave the MCC (Matthews correlation coefficient) score of 0.3275 and 0.332 respectively.