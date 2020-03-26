import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from BertSequenceClassification_streamline.bert_downstream_classification import *

class AppCommentData():
    def __init__(self, df, mode, tokenizer = None, batch_size = None):
        """
        Args:
            mode: in ["train", "test", "val", "all"]
            tokenizer: one of bert tokenizer
        """
        self.df = df
        self.mode = mode
        self.tokenizer = tokenizer
        self.batch_size = batch_size
    
    def reindex(self, label2index):
        """Reindex the df given the mapping in label2index so that 
        it can be fed to model
        Called to reindex the train val test data to the label in "all" data
        """
        df_reindex = self.df.copy()
        def getindex4label(label):
            return label2index[label]
        df_reindex["index"] = df_reindex["index"].apply(getindex4label)
        self.df_reindex = df_reindex

    def get_index2label(self):
        """
        Return:
            A dictionary {class_index: class_label}
        """
        index = self.df['index']
        index2label = {idx:val for idx, val in enumerate(index.unique())}
        return index2label

    def get_label2index(self):
        index2label = self.get_index2label()
        return {lab:idx for idx, lab in index2label.items()}

    def get_num_index(self):
        return len(self.get_index2label())

    def get_index_dist(self, verbose = False):
        """得到此 dataframe 的訓練分布 (分類:筆數) return the mapping of value and occurrence
           df: "pd.DataFrame" type
        """
        each_count = self.df['index'].value_counts()
        if verbose:
            print(self.mode + " data 各類分布: \n")
            for index in each_count.index:
                print("{:15} 數量: {}".format(index, each_count[index] ))
        return each_count

    def plot_pie(self, fontprop):
        plt.figure(figsize=(10, 10), facecolor="w")
        index_dist = self.get_index_dist()

        plt.title("{} data Distribution ({} data)".format(self.mode, sum(index_dist)), fontsize=16)
        patches,l_text, p_text = plt.pie(index_dist, autopct="%1.1f%%",
                                        textprops = {"fontsize" : 12}, labeldistance=1.05)
        for i, t in enumerate(l_text): 
            t.set_text(index_dist.index[i])
            t.set_fontproperties(fontprop)
            #t.set_color('red')
            pct = float(p_text[i].get_text().strip('%'))
            if pct < 2:
                p_text[i].set_text("")
            #p_text[i].set_color('red')
        plt.show()

    def get_dataset(self):
        """label2index contains label to index mapping as in the all dataset"""
        return OnlineQueryDataset(self.mode, self.df_reindex, self.tokenizer)

    def get_dataloader(self):
        """Return a dataloader that can be fed to """
        shuffle = True if self.mode == "train" else False
        return DataLoader(self.get_dataset(), batch_size=self.batch_size, shuffle = shuffle,  
                            collate_fn=create_mini_batch)
    """Usage:
    app_data = AppCommentData(df)
    app_data.plot_pie()
    """

def preprocess_app_comment(path, verbose = False):
    """讀取 app comment，轉化為後續 model 可以理解的格式，並輸出清理過後的dataset
    Args:
        path (str): path to the app comment file
        verbose (int): If True, print 各類別對應的筆數
    Returns:
        DataFrame: The DataFrame with only relevant data
    """
    df_all = pd.read_excel(path)
    df_all.rename(columns={"評論內容": "question", "類別": "index"}, inplace = True)
    df_all.dropna(axis = 0, how = 'any', subset=["question", "index"], inplace = True)
    if verbose:
        print(df_all['index'].value_counts())
    df_all = df_all.loc[:, ["index", "question"]]       # get 'index' and 'question' coluumn
    return df_all

def df2binary(df):
    df_new = df.copy()
    df_new['index'] = df_new['index'].apply(lambda x: 1 if x == '稱讚' else 0)
    return df_new

def df_without_bin(df):
    return df[df['index'] != '稱讚']

def train_val_test_split(df, train_size = 0.75):
    train, test = train_test_split(
        df, train_size=train_size)
    # train, val = train_test_split(
    #     rain, test_size=(1/9), random_state=1)
    val, test = train_test_split(
        test, test_size=0.5)
    return train, val, test

def get_confusion_matrix(true_label, predictions, num_index):
    """Return a group-to-group comparison matrix and a list of list storing
       the index of comments that are assigned to a wrong group
    Args:
        true_label: an array that stores the truth label of the test set
        predictions: an array that stores the model prediction
        num_index: how many classes used in the model
    Returns:
        class_matrix (list of list): The confusion matrix
        false_group (list of list): false_group[i] - comments in group i that are
            assigned to other groups
    """
    class_matrix = np.zeros(shape=(num_index, num_index))
    false_group = [[] for _ in range(num_index)]
    for idx, true, pred in zip(range(len(predictions)),true_label, predictions):
        class_matrix[true][pred] += 1
        if true != pred:
            false_group[true].append(idx)
    return class_matrix, false_group

def print_acc(class_matrix):
    """print the accuracy given a confusion matrix"""
    total = 0
    num_index = len(class_matrix)
    for i in range(num_index):
        total += class_matrix[i][i]
    print("Accuracy: {0}%".format(100 * total/np.sum(class_matrix)))

def print_summary_i(df, i, class_matrix, index2label, false_group, predictions, firstK = 3):
    """print 出第 i 組的分類概覽
       包括所有此類的問題， 被錯誤分類成哪個組別
    Args:
        df (Dataframe): The test data that you want to summarize
        i (int): the i-th class
        class_matrix (list of list): the confusion matrix
        index2label (dict): the class index to label mapping 
        false_group (list of list): false_group[i] - comments in group i that are
            assigned to other groups
        firstK (int): How many wrongly classified comments you want to present 
    """
    num_group = np.sum(class_matrix[i])
    #num_group = num_each_group[i]
    print("類別: {0} \t 測試集筆數: {1}".format(index2label[i], int(num_group)))
    #print("正確分類筆數: {0}".format(class_matrix[i][i]))
    print("被分類器正確分類的機率: {0} %".format(100 * class_matrix[i][i]/num_group) )
    sorted_idxs = np.argsort(class_matrix[i])
    if sorted_idxs[-1] == i:
        idx = sorted_idxs[-2]
    else:
        idx = sorted_idxs[-1]
    
    if class_matrix[i][idx] != 0:
        print("最常被分錯的組別: {0}  筆數: {1}\n".format(index2label[idx], class_matrix[i][idx]))
        count = 0
        for f in false_group[i]:
            if count >= firstK: break
            print(df.iloc[f]['question'])
            print("\t被分成: {0}".format(index2label[predictions[f]]))
            count += 1

def plot_dist(index_dist ,label_list, fontprop = None):
    """Plot the distribution given in index_dist
    Args:
        index_dist (list): A list of distribution
    """
    total = np.sum(index_dist)
    num_index = len(index_dist)
    y_value = np.array(index_dist, dtype = int)
    x_axis = range(num_index)

    plt.figure(figsize = (10, 5), facecolor="w")
    plt.bar(x_axis, height = y_value)
    
    plt.xticks(x_axis, label_list, fontproperties=fontprop)
    plt.xticks(rotation=270, fontproperties=fontprop)
    plt.xlabel("被分類的組別", fontproperties=fontprop)
    for j in range(num_index):
        perc = round(100 * y_value[j]/total, 2)
        plt.text(j - 0.4, y_value[j], str(perc) + "%", color='blue', fontproperties=fontprop)
    #plt.subplot(432)
    plt.show()