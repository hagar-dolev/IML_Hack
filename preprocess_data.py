import os.path
import pandas as pd
import numpy as np
import pathlib

import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#
#
# def remove_not_labeled(data, label):
#     # data.dropna(how='all')
#     wher_none = np.ones(data.shape[0]).astype(np.bool)
#     for l in label:
#         wher_none = wher_none & data[l].isna()
#     return data.drop(np.nonzero(wher_none == True)[0], axis=0)
#
#
# def replace_nan_with_div(data):
#
#     values = data[data.notna()]
#     if values.shape[0] == data.shape[0]:
#         # data.fillna(0, inplace=True)
#         return data
#     size = values.shape[0]
#     count = Counter(values)
#
#     values = np.unique(values)
#     if values.shape[0] == 0:
#         # data.fillna(0, inplace=True)
#         return data
#
#     nan_amount = data.shape[0] - size
#     probabileties = np.array([count[key]/size for key in count.keys()])
#     new_vals = np.random.choice(values, size=nan_amount, p=probabileties)
#     for ind, val in zip(np.nonzero(data.isnull())[0], new_vals):
#         data.iat[ind] = val
#
#     return data
#
#
# def schoolDays(x):
#     if x == 996:
#         return 365
#     if x== 997 or x==999:
#         return -1
#     return x
#
#
# def pre_process_data():
#     my_path = os.path.abspath(os.path.dirname(__file__))
#
#     # questions_path = os.path.join(my_path,
#     #                               "../learning_disabilities/child_survey_data/questionier_child.pdf")
#     # questions = extract_from(questions_path)
#     # # print(*[(x,questions[x]) for x in questions.keys()], sep="\n\n")
#     # vars_colmns_path = os.path.join(my_path,
#     #                                 "../learning_disabilities/child_survey_data/CSV_samchild_summary_variables_desc.csv")
#     # vars_to_doc_data = pd.read_csv(vars_colmns_path,
#     #                                names=['Question', 'Column1', 'Recode', 'Instrument Variable Name', 'FinalDocName',
#     #                                       'Processing Variable Label', 'Location', 'Column2', 'Length'])
#
#     ##############################################################
#     # not_categorial = ["ADD2", "ADD2N", "WTIA_SC", "WTFA_SC", "TOTOZ_P", "BWTGRM_P", "CHGHT_TC", "CWGHT_TC",
#     #                   "BMI_SC", "AGE_P", "CBALAGE", "SCHDAYR1"]
#     # child_data_set = remove_not_labeled(child_data_set, ['ADD2', 'ADD2N'])
#     # target = np.logical_or(child_data_set['ADD2'] == 0, child_data_set['ADD2N'] == 0).astype(np.int)
#     #
#     # child_data_set.dropna(axis=1, how="all")
#     # child_data_set.dropna(axis=1, thresh=int(child_data_set.shape[0] * (6 / 10)))
#     #
#     # child_data_set = child_data_set.apply(lambda x: replace_nan_with_div(x))
#     #
#     # child_data_set[not_categorial] = child_data_set[not_categorial].apply(lambda x: x.astype(np.float64))
#     # child_data_set["SCHDAYR1"].apply(schoolDays)
#     #
#     # for column in child_data_set:
#     #     # print()
#     #     if column not in not_categorial:
#     #         print(column, child_data_set[column].cat.categories)
#     #
#     # # child_data_set[not_categorial].astype(np.float64)
#
#     ##############################################################
#     data_set_path = os.path.join(my_path, "../learning_disabilities/child_survey_data/samchild.csv")
#     child_data_set = pd.read_csv(data_set_path , dtype='category')
#     child_data_set.drop(labels=["RECTYPE", "SRVY_YR", "HHX", "INTV_QRT", "INTV_MON", "FMX", "FPX"], axis=1, inplace=True)
#
#     not_categorial = ["ADD2", "ADD2N", "WTIA_SC" ,"WTFA_SC", "TOTOZ_P", "BWTGRM_P", "CHGHT_TC", "CWGHT_TC", "BMI_SC", "AGE_P", "CBALAGE"]
#     child_data_set[not_categorial] = child_data_set[not_categorial].apply(lambda x: x.cat.codes)
#
#     for column in child_data_set:
#         # print()
#         if column not in not_categorial:
#             print(column, child_data_set[column].cat.categories)
#
#     child_data_set = remove_not_labeled(child_data_set, ['ADD2', 'ADD2N'])
#     target = np.logical_or(child_data_set['ADD2'] == 0, child_data_set['ADD2N'] == 0).astype(np.int)
#
#     child_data_set.dropna(axis=1, how="all")
#     child_data_set.dropna(axis=1, thresh=int(child_data_set.shape[0]*(6/10)))
#
#     child_data_set = child_data_set.apply(lambda x: replace_nan_with_div(x))
#
#     child_data_set[not_categorial].astype(np.float64)
#
#     child_data_set = child_data_set.drop(columns=['ADD2', 'ADD2N'])
#
#     child_data_set = pd.get_dummies(child_data_set)
#
#     child_data_set['target'] = target
#
#     features = child_data_set.columns.values
#     features = np.delete(features, np.where(features == 'target'))
#
#     child_data_set.to_csv(os.path.join(my_path, "../learning_disabilities/child_survey_data/sam_chiled_cleanedupnow.csv"))
#
#     return child_data_set, features
#

# def unify_samples(files_lines):


def main():
    my_path = os.path.abspath(os.path.dirname(__file__))
    code_samples_path = os.path.join(my_path,"../IML_Hack/Task2_files")
    # define lib the path
    files_lines = []
    currentDirectory = pathlib.Path(code_samples_path)
    for i, currentFile in enumerate(currentDirectory.iterdir()):
        with open(currentFile, 'r') as infile:
            with open(code_samples_path+'/tagged{}.txt'.format(i), 'w') as outfile:
                for line in infile:
                    line = line.replace(os.linesep, '{}'.format(i) + os.linesep)
                    outfile.write(line)
        # curr = open(currentFile)
        # files_lines.append(curr.readlines())

    # data, features = pre_process_data()



if __name__ == '__main__':
    main()