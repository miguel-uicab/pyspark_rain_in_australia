# !/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from pyspark.mllib.stat import Statistics
from pyspark.mllib.linalg import Vectors


def get_correlation_matrix(data=None,
                           list_feature_names=None):
    """
    Se calcula la matriz de correlación de Spearman.
    """
    df = data.select(list_feature_names)

    # Convert the DataFrame into an RDD of Vectors
    rdd_vectors = df.rdd.map(lambda row: Vectors.dense(row))

    # Calculate the Pearson correlation matrix using the RDD of Vectors
    correlation_matrix = Statistics.corr(rdd_vectors, method="spearman")
    correlation_df = pd.DataFrame(correlation_matrix,
                                  columns=list_feature_names,
                                  index=list_feature_names)
    return correlation_df


def get_features_names_drop_by_corr(pd_corr=None,
                                    threshold=None):
    """
    Devuelve una lista de variables que sobrepasan cierto valor de correlación
    con respecto a otras variables.
    """
    upper = pd_corr.where(np.triu(np.ones(pd_corr.shape),k=1).astype(bool))
    drop_by_corr = [column for column in upper.columns if any(upper[column] > threshold)]

    return drop_by_corr