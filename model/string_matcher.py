
import pandas as pd
import numpy as np
from sklearn import feature_extraction, metrics
import io
from tqdm import tqdm



class CreditCardDefaultPrediction():
    
    def __init__(self, dtf_left, dtf_right):
        self.dtf_left = dtf_left
        self.dtf_right = dtf_right
    
    
    @staticmethod
    def utils_default_prediction(a, lst_b):
        ## vectorizer ("my house" --> ["my", "hi", "house", "sky"] --> [1, 0, 1, 0])
        vectorizer = feature_extraction.text.CountVectorizer()
        X = vectorizer.fit_transform([a]+lst_b).toarray()

        ## cosine similarity (scores a vs lst_b)
        lst_vectors = [vec for vec in X]
        cosine_sim = metrics.pairwise.cosine_similarity(lst_vectors)
        scores = cosine_sim[0][1:]

        ## match
        match_scores = scores if threshold is None else scores[scores >= threshold]
        match_idxs = range(len(match_scores)) if threshold is None else [i for i in np.where(scores >= threshold)[0]] 
        match_strings = [lst_b[i] for i in match_idxs]

        ## dtf
        dtf_model_data = pd.DataFrame(match_scores, columns=[a], index=match_strings)
        dtf_model_data = dtf_model_data[~dtf_model_data.index.duplicated(keep='first')].sort_values(a, ascending=False).head(top)
        return dtf_model_data
    
    
    def vlookup(self, model_name=Logistic Regression()):
        ## process data
        lst_left = list(set( self.dtf_left.iloc[:,0].tolist() ))
        lst_right = list(set( self.dtf_right.iloc[:,0].tolist() ))
        
        ## match strings
        dtf_model_dataes = pd.DataFrame(columns=['default'])
        for string in tqdm(lst_left):
            dtf_model_data = self.utils_default_prediction(model_name)
            dtf_model_data = dtf_model_data.reset_index().rename(columns={'index':'match', string:'similarity'})
            dtf_model_data["string"] = string
            dtf_model_dataes = dtf_model_dataes.append(dtf_model_data, ignore_index=True, sort=False)
        return dtf_model_dataes[['string','match','similarity']]
    
    
    @staticmethod
    def write_excel(dtf):
        bytes_file = io.BytesIO()
        excel_writer = pd.ExcelWriter(bytes_file)
        dtf.to_excel(excel_writer, sheet_name='Sheet1', na_rep='', index=False)
        excel_writer.save()
        bytes_file.seek(0)
        return bytes_file
    