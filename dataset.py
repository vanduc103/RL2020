#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing


# load data by dataset_id
def load_data(dataset, dataset_id):
    df = pd.read_csv(dataset)
    if dataset_id == '23':
        X = df.loc[:, df.columns != 'Contraceptive_method_used'].to_numpy()
        y = df.loc[:, df.columns == 'Contraceptive_method_used'].to_numpy()
    elif dataset_id == '38':
        df = df.where(df == '?', None) 
        X = df.loc[:, (df.columns != 'Class') & (df.columns != 'TBG')].to_numpy()
        y = df.loc[:, df.columns == 'Class'].to_numpy()
    elif dataset_id == '46':
        X = df.loc[:, (df.columns != 'Class') & (df.columns != 'Instance_name')].to_numpy()
        y = df.loc[:, df.columns == 'Class'].to_numpy()
    elif dataset_id == '181':
        X = df.loc[:, (df.columns != 'class_protein_localization')].to_numpy()
        y = df.loc[:, df.columns == 'class_protein_localization'].to_numpy()
    elif dataset_id == '184':
        X = df.loc[:, (df.columns != 'game')].to_numpy()
        y = df.loc[:, df.columns == 'game'].to_numpy()
    elif dataset_id == '185':
        X = df.loc[:, (df.columns != 'Hall_of_Fame') & (df.columns != 'Player')].to_numpy()
        y = df.loc[:, df.columns == 'Hall_of_Fame'].to_numpy()
    elif dataset_id == '273':
        X = df.loc[:, (df.columns != 'Drama')].to_numpy()
        y = df.loc[:, df.columns == 'Drama'].to_numpy()
    elif dataset_id == '679':
        X = df.loc[:, (df.columns != 'sleep_state')].to_numpy()
        y = df.loc[:, df.columns == 'sleep_state'].to_numpy()
    elif dataset_id == '715':
        X = df.loc[:, (df.columns != 'binaryClass')].to_numpy()
        y = df.loc[:, df.columns == 'binaryClass'].to_numpy()
    elif dataset_id == '718':
        X = df.loc[:, (df.columns != 'binaryClass')].to_numpy()
        y = df.loc[:, df.columns == 'binaryClass'].to_numpy()
    elif dataset_id in ['720', '722', '723', '727', '728', '734', '735', '737', '740', '741', '743', '751', '752', '761','772','797','799','803','806','807'
                        ,'813','816','819','821','822','823','833','837','843','845','846','847','849','866','871','881','901','903','904',
                         '910', '912','913','917','971','976','977','978','979','980','1019','1020','1021']:
        X = df.loc[:, (df.columns != 'binaryClass')].to_numpy()
        y = df.loc[:, df.columns == 'binaryClass'].to_numpy()
    elif dataset_id == '914':
        X = df.loc[:, (df.columns != 'binaryClass') & (df.columns != 'id')].to_numpy()
        y = df.loc[:, df.columns == 'binaryClass'].to_numpy()
    elif dataset_id in ['1036', '1040']:
        X = df.loc[:, (df.columns != 'label')].to_numpy()
        y = df.loc[:, df.columns == 'label'].to_numpy()
    elif dataset_id in ['1049','1050','1056','1069']:
        X = df.loc[:, (df.columns != 'c')].to_numpy()
        y = df.loc[:, df.columns == 'c'].to_numpy()
    elif dataset_id in ['1053','1067','1068']:
        X = df.loc[:, (df.columns != 'defects')].to_numpy()
        y = df.loc[:, df.columns == 'defects'].to_numpy()
    elif dataset_id == '1120':
        X = df.loc[:, (df.columns != 'class:') & (df.columns != 'ID')].to_numpy()
        y = df.loc[:, df.columns == 'class:'].to_numpy()
    elif dataset_id in ['1128','1130','1134','1138','1139','1142','1146','1161','1166']:
        X = df.loc[:, (df.columns != 'Tissue') & (df.columns != 'ID_REF')].to_numpy()
        y = df.loc[:, df.columns == 'Tissue'].to_numpy()
    elif dataset_id == '1abalone':
        X = df.loc[:, (df.columns != 'sex')].to_numpy()
        y = df.loc[:, df.columns == 'sex'].to_numpy()
    elif dataset_id == '1amazon':
        X = df.loc[:, (df.columns != 'class')].to_numpy()
        y = df.loc[:, df.columns == 'class'].to_numpy()
    else:
        X = df.loc[:, df.columns != 'class'].to_numpy()
        y = df.loc[:, df.columns == 'class'].to_numpy()
    feat_type = None

    le = preprocessing.LabelEncoder()
    if dataset_id == '3':
        feat_type = ['Categorical' for i in range(X.shape[1])]
        for i in range(X.shape[1]):
            X[:,i] = le.fit_transform(X[:,i])
        y = le.fit_transform(y)
    elif dataset_id == '6':
        feat_type = ['Numerical' for i in range(X.shape[1])]
        y = le.fit_transform(y)
    elif dataset_id == '12':
        feat_type = ['Numerical' for i in range(X.shape[1])]
    elif dataset_id == '14':
        feat_type = ['Numerical' for i in range(X.shape[1])]
    elif dataset_id == '16':
        feat_type = ['Numerical' for i in range(X.shape[1])]
    elif dataset_id == '18':
        feat_type = ['Numerical' for i in range(X.shape[1])]
    elif dataset_id == '21':
        feat_type = ['Categorical' for i in range(X.shape[1])]
        for i in range(X.shape[1]):
            X[:,i] = le.fit_transform(X[:,i])
        y = le.fit_transform(y)
    elif dataset_id == '23':
        feat_type = ['Numerical', 'Categorical', 'Categorical', 'Numerical', 'Categorical', 'Categorical', 'Categorical', 'Categorical', 'Categorical']
        for i in [1, 2, 4, 5, 6, 7, 8]:
            X[:,i] = le.fit_transform(X[:,i])
        y = le.fit_transform(y)
    elif dataset_id == '24':
        feat_type = ['Categorical' for i in range(X.shape[1])]
        for i in range(X.shape[1]):
            X[:,i] = le.fit_transform(X[:,i])
        y = le.fit_transform(y)
    elif dataset_id == '26':
        feat_type = ['Categorical' for i in range(X.shape[1])]
        for i in range(X.shape[1]):
            X[:,i] = le.fit_transform(X[:,i])
        y = le.fit_transform(y)
    elif dataset_id == '31':
        feat_type = [None] * 20
        for i in [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]:
            feat_type[i] = 'Categorical'
        for i in [1, 4, 7, 10, 12, 15, 17]:
            feat_type[i] = 'Numerical'
        for i in [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]:
            X[:,i] = le.fit_transform(X[:,i])
        y = le.fit_transform(y)
    elif dataset_id == '36':
        y = le.fit_transform(y)
    elif dataset_id == '38':
        feat_type = ['Categorical'] * 28
        for i in [0, 17, 19, 21, 23, 25]:
            feat_type[i] = 'Numerical'
        not_in_list = [0, 17, 19, 21, 23, 25]
        for i in range(X.shape[1]):
            if i not in not_in_list:
                X[:,i] = le.fit_transform(X[:,i])
        X = X.astype(np.float)
        y = le.fit_transform(y)
    elif dataset_id == '46':
        feat_type = ['Categorical'] * 60
        for i in range(X.shape[1]):
            X[:,i] = le.fit_transform(X[:,i])
        y = le.fit_transform(y)
    elif dataset_id == '181':
        y = le.fit_transform(y)
    elif dataset_id == '182':
        y = le.fit_transform(y)
    elif dataset_id == '184':
        feat_type = ['Categorical'] * 6
        for i in range(X.shape[1]):
            X[:,i] = le.fit_transform(X[:,i])
        y = le.fit_transform(y)
    elif dataset_id == '185':
        feat_type = ['Numerical'] * 16
        for i in [15]:
            feat_type[i] = 'Categorical'
        for i in [15]:
            X[:,i] = le.fit_transform(X[:,i])
    elif dataset_id == '715':
        y = le.fit_transform(y)
    elif dataset_id == '718':
        y = le.fit_transform(y)
    elif dataset_id == '720':
        feat_type = ['Numerical'] * 8
        for i in [0]:
            feat_type[i] = 'Categorical'
        for i in [0]:
            X[:,i] = le.fit_transform(X[:,i])
        y = le.fit_transform(y)
    elif dataset_id in ['722', '723', '727', '728', '734', '735', '737', '740', '743','751', '752', '761','772','797','799','803','806','807',
                        '813','816','819','821','822','823','833','837','843','845','846','847','849','866','871','901','903','904','910','912','913','914','917',
                        '976','977','978','979','980', '1019','1020','1021','1036','1040','1049','1050', '1053','1056','1067','1068','1069',
                        '1120','1128', '1130','1134','1138','1139','1142','1146','1161','1166']:
        y = le.fit_transform(y)
    elif dataset_id == '741':
        feat_type = ['Numerical', 'Categorical']
        for i in [1]:
            X[:,i] = le.fit_transform(X[:,i])
        y = le.fit_transform(y)
    elif dataset_id == '881':
        feat_type = ['Numerical'] * 10
        for i in [2, 6, 7]:
            feat_type[i] = 'Categorical'
        for i in [2, 6, 7]:
            X[:,i] = le.fit_transform(X[:,i])
        y = le.fit_transform(y)
    elif dataset_id in ['1abalone', '1amazon', '1gisette', '1madelon', '1yeast']:
        y = le.fit_transform(y)
    elif dataset_id == '1car':
        feat_type = ['Categorical'] * 6
        for i in range(X.shape[1]):
            X[:,i] = le.fit_transform(X[:,i])
        y = le.fit_transform(y)
    elif dataset_id == '1germancredit':
        feat_type = ['Categorical'] * 20
        for i in [1,4,7,10,12,15,17]:
            feat_type[i] = 'Numerical'
        for i in [0,2,3,5,6,8,9,11,13,14,16,18,19]:
            X[:,i] = le.fit_transform(X[:,i])
        y = le.fit_transform(y)
    elif dataset_id == '1krvskp':
        feat_type = ['Categorical'] * 36
        for i in range(X.shape[1]):
            X[:,i] = le.fit_transform(X[:,i])
        y = le.fit_transform(y)

    X = np.array(X, dtype=float)
    print(X.shape)
    print(y.shape)
    return X, y, feat_type

if __name__ == "__main__":
    main()
