import pandas as pd
import numpy as np
from scipy import stats
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy import stats

from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def outlier_boxplot(col_df_prueba, df):
    
    name_variable = pd.DataFrame(col_df_prueba).columns[0]
    tipo = df.loc[name_variable, 'tipo_variable']
    
    list_outliers = []
    if tipo == 'Numérica':

        qtile_25 = col_df_prueba.quantile(0.25)
        mediana = col_df_prueba.quantile(0.5)
        qtile_75 = col_df_prueba.quantile(0.75)
        ITQ = qtile_75 - qtile_25
        max_value = mediana + 1.5 * ITQ
        min_value = mediana - 1.5 * ITQ

        values = col_df_prueba.dropna().values.squeeze()

        if len(np.unique(values)) != 2:   
            for value in values:
                if (value > max_value) or (value < (min_value)):
                    list_outliers.append(value)
        list_outliers = list(np.unique(list_outliers))        
    return list_outliers
def len_outliers(list_values):
    
    try:
        if type(list_values) == list:
            len_out = len(list_values)
        else:
            len_out = 0
    except:
        len_out = 0
    
    return len_out
def mean_mode(col_df_prueba):
    values = col_df_prueba.dropna().values.squeeze()
    
    if apply_tipo(col_df_prueba) != 'Numérica':
        result = stats.mode(values)[0][0]
    elif apply_tipo(col_df_prueba) == 'Numérica':
        result = np.mean(values)
    return result
def label_encoding(df_original, variables_categorical):
    df_original2 = df_original.copy()
    
    for i, variable in enumerate(variables_categorical):
    
        values = df_original[variable].dropna().unique()
        
        #Categóricas dicotómica descriptiva
        if (len(values) == 2) and (np.sum([type(value) == str for value in values]) != 0):
            df_original2[variable].replace(values[0], 0, inplace = True)
            df_original2[variable].replace(values[1], 1, inplace = True)
        
        #Categórica dicotómica numérica
        elif (len(values) == 2) and (np.sum([type(value) == str for value in values]) == 0):
            pass
        
        #Categórica > 2 clases
        elif len(values) >= 2:

            total_target1 = df_original2[[variable, 'desercion']].dropna().groupby(variable).sum().sum()[0]
            df_mean_label = df_original2[[variable, 'desercion']].dropna().groupby(variable).sum() / total_target1
            dict_target = df_mean_label.to_dict()['desercion']

            for key, value in dict_target.items():
                df_original2[variable].replace(key, value, inplace = True)

    return df_original2
def apply_tipo(df_column):
    '''Esta función indica el tipo de variable de entrada (Categórica, Numérica, Categórica Dicotómica)
     Args:
         df_column: serie de entrada'''
    
    values = df_column.dropna().values.squeeze()
    if len(np.unique(values)) == 2:
        tipo = 'Categórica Dicotómica'
    else:
        if np.sum([type(value) != str for value in values]) != 0:
            tipo = 'Numérica'
        else:
            tipo = 'Categórica'
    return tipo

def is_normal(df_column):
    '''Esta función asigna el nombre de la columna a la serie introducida df_column.'''
    
    if stats.shapiro(df_column.dropna()).pvalue < 0.05:
        result = 'NO NORMAL'
    else:
        result = 'NORMAL'
    return result

def name(df_column, name_col):
    '''Esta función asigna el nombre de la columna a la serie introducida.
    Args:
    
        df_column: serie (columna de dataframe) de entrada de la variable explicativa
        name_col : nuevo nombre de la column '''
    
    df_named = pd.DataFrame(df_column, columns = [name_col])[name_col]
    if df_named.empty:
        df_named = pd.DataFrame(df_column.copy())
        df_named.columns = [name_col]
        df_named = df_named[name_col]        
        
    return df_named



def gini_function(col_df_prueba, col_df_target,df, mode_outliers = 0 ):
    
#     print(col_df_prueba)
    name_variable = pd.DataFrame(col_df_prueba).columns[0]
    tipo = df.loc[name_variable, 'tipo_variable']
    df_join = pd.concat([col_df_prueba, col_df_target], axis = 1).replace([np.inf, -np.inf], np.nan).dropna()
    X_train = df_join.iloc[:,:-1].values
    y_train = df_join.iloc[:,-1].values.squeeze()
    
    #Este if es para quitar los outliers según el modo
    if ((mode_outliers == 1 or mode_outliers == 2) and tipo == 'Numérica'):
        if mode_outliers == 1:
            col_outlier = 'outlier_std'
        elif mode_outliers == 2:
            col_outlier = 'outlier_boxplot'

        outliers = [value for value in df.loc[name_variable, col_outlier]]
        list_index_outliers = []
        _ = [list_index_outliers.extend(np.where(X_train.squeeze() == value)[0]) for value in outliers] 
        list_index_outliers = np.unique(list_index_outliers)
 
        if len(list_index_outliers) != 0:
            X_train = np.delete(X_train, list_index_outliers).reshape(-1,1)
            y_train = np.delete(y_train, list_index_outliers)
            #del y_train[list_index_outliers]
    
    # try: 
    lr = LogisticRegression(max_iter=200)
    model_res = lr.fit(X_train, y_train)
    auc = roc_auc_score(y_train, lr.predict_proba(X_train)[:,1])
    gini = 2 * auc - 1
    if auc <= 0.5:
        gini = 0
    GINI = np.round(gini, 3)
    # except:
    #     GINI = None
        
    
    
    return GINI