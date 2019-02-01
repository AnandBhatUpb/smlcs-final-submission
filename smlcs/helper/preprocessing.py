from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

class Preprocessing:
    def handle_missing_data(self, X):
        imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imputer = imputer.fit(X)
        return imputer.transform(X)

    def encode_categorical_data(self, X):
        encoded_df = pd.DataFrame()
        for i in range(X.shape[1]):
            le_i = LabelEncoder()
            encoded_df[X.iloc[:,i].name+'_encoded'] = le_i.fit_transform(X.iloc[:,i])
        ohe_encoded_df = pd.DataFrame()
        for j in range(encoded_df.shape[1]):
            ohe = OneHotEncoder()
            m = ohe.fit_transform(encoded_df.iloc[:,j].values.reshape(-1, 1)).toarray()
            df = pd.DataFrame(m, columns=[X.iloc[:,j].name+str(int(k)) for k in range(m.shape[1])])
            ohe_encoded_df = pd.concat([ohe_encoded_df, df], axis=1)
        return ohe_encoded_df
        #ohe_encoded_df.to_csv('encoded_frame.csv', sep=',', encoding='utf-8')


    def encode_labels(self, Y):
        labelencoder_y = LabelEncoder()
        return labelencoder_y.fit_transform(Y)

