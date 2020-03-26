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

# df = preprocess_app_comment(path, verbose = False) # path stores the raw app comment data

# def get_dataset_dist(mode, dataset, index2label, verbose = False):
#     """得到此 dataset 的訓練分布 (分類:筆數)
#        dataset: OnlineQueryDataset type
#     """
#     def get_index2label(i):
#         return index2label[i]
#     index_list = dataset.df['index'].apply(get_index2label) 

#     each_count = index_list .value_counts()
#     if verbose:
#         print(mode + "set 各類分布: \n")
#         for idx in each_count.index:
#             print("{:15} 數量: {}".format(index2label[idx], each_count[idx] ))
#     return each_count


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
    for i in range(num_index):
        total += class_matrix[i][i]
    print("Accuracy: {0}%".format(100 * total/np.sum(class_matrix)))

# def print_summary_all(class_matrix, index2label, false_group):
#     num_each_group = np.sum(class_matrix, axis = 1)
#     for i in range(num_index):
#         print("類別: {0} \t 測試集筆數: {1}".format(index2label[i], int(num_each_group[i]) ))
#         print("被分類器正確分類的機率: {0} %".format(100 * class_matrix[i][i]/num_each_group[i]) )
#         sorted_idxs = np.argsort(class_matrix[i])
#         if sorted_idxs[-1] == i:
#             idx = sorted_idxs[-2]
#         else:
#             idx = sorted_idxs[-1]
#         if class_matrix[i][idx] != 0:
#             print("最常被分錯的組別: {0}  筆數: {1}".format(index2label[idx], class_matrix[i][idx]) )
#             print("錯誤分組範例: {0}".format( testset.df.iloc[false_group[i][0]]['question'] ))
#             print("\t被分成: {0}".format(index2label[ predictions[false_group[i][0]] ] ))
#         print()

def print_summary_i(i, class_matrix, index2label, false_group, firstK = 3):
    """print 出第 i 組的分類概覽
       包括所有此類的問題， 被錯誤分類成哪個組別
    Args:
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
            print(testset.df.iloc[f]['question'])
            print("\t被分成: {0}".format(index2label[predictions[f]]))
            count += 1

"""畫出第 i 組評論的分類分佈"""
def plot_dist(index_dist):
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
    #plt.xticks(range(num_index),('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'))
    plt.xticks(x_axis, y_value, fontproperties=fontprop)
    #plt.title("原始組別: " + label_list[i], fontproperties=fontprop)
    plt.xticks(rotation=270, fontproperties=fontprop)
    plt.xlabel("被分類的組別", fontproperties=fontprop)
    for j in range(num_index):
        perc = round(100 * y_value[j]/total, 2)
        plt.text(j - 0.4, y_value[j], str(perc) + "%", color='blue', fontproperties=fontprop)
    #plt.subplot(432)
    plt.show()

if __name__ == "__main__":
    # 為了讓 colab 能 print 中文
    fontprop = let_colab_print_chinese()
    #fm.FontProperties(fname='/usr/share/fonts/truetype/NotoSansCJKkr-Medium.otf', size= 12)
    args = Args()
    path = "公版評論回覆與分類表@20200313.xlsx"
    df = preprocess_app_comment(path, verbose = False)
    df = filter_toofew_toolong(df, args.min_each_group, args.maxlength)
    df.to_csv("app_comment" + "_me" + str(args.min_each_group) + "_ml" + str(args.maxlength) + ".csv")

    # investigate all data
    app_data = AppCommentData(df)
    app_data.plot_pie()
    app_data.get_index_dist(verbose = True)
    num_labels = app_data.get_num_index()
    index2label = app_data.get_index2label()

    df_train, df_val, df_test = train_val_test_split(df)
    

    PRETRAINED_MODEL_NAME = "bert-base-chinese"
    # 取得此預訓練模型所使用的 tokenizer
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    # investigate train set
    app_train = AppCommentData(df_train, "train", tokenizer, 64)
    app_train.plot_pie()

    # investigate test set
    app_val = AppCommentData(df_val, "val", tokenizer, 64)
    app_test = AppCommentData(df_test, "test", tokenizer, 64)
    app_test.plot_pie()
    
    trainloader = app_train.get_dataloader()
    valloader = app_val.get_dataloader()
    testloader = app_test.get_dataloader()

    model = train(trainloader, valloader, PRETRAINED_MODEL_NAME, num_labels, args.epoch)
    save_model(args, args.model_output, model, tokenizer)

    # testing
    predictions = get_predictions(model, testloader).detach().cpu().numpy()

    # Presentation
    true_label = app_test.df["index"]
    class_matrix, false_group = get_confusion_matrix(true_label, predictions, num_labels)
    print_acc("測試集 Accuracy: ", class_matrix)

    label_list = [index2label[key] for key in index2label]
    label2index = {val:key for key, val in index2label.items()}
    summary = pd.DataFrame(class_matrix, dtype = int, columns=label_list, index = label_list)
    print("測試集 confusion matrix:")
    summary

    # 個別檢查
    for i in range(num_labels):
        print_summary_i(i, class_matrix, index2label, false_group, 3)

    # labels: ('未知問題','操作問題','稱讚','建議-台幣','抱怨','建議-信用卡','建議-APP','顧客疑問','系統問題-bug','第三方問題','系統問題-Keypasco')
    i = label2index['抱怨']
    plot_group_i(i, class_matrix)
    plot_dist(class_matrix[i])
    #print_summary_i(i, class_matrix, index2label)


"""set the arguments if not using a command line """
class Args:
    data_path = 'all_app_comment.csv'
    epoch = 25
    batch_size = 64
    min_each_group = 5                      # 每組至少要有幾筆才被放入訓練集
    maxlength = 50                          # 若app comment評論超過此長度則刪除
    model_output = 'model_ep25_eg5_ml50'
    #model_start = "app_all1"

"""由於 “第三方問題” “建議-台幣” “顧客疑慮” 筆數過少，利用設 "args.min_each_group" 參數來將之排除在分類器的訓練集裡。若將來有更多資料，就可重新放入訓練"""

# if args.model_prediction:
#     write_prediction(args.model_prediction, predictions)

# if 'index' in testset.df:      # If we have labels on test set, we can calculate the accuracy
#     # 用分類模型預測測試集
#     test_label = testset.df['index']
#     print("Testset accuracy: %f" % plain_accuracy(test_label, predictions))



"""## Approach 2: Binary calssification followed by a multiclass classification on non-compliment comments

(1) Binary calssification (分出稱讚以及非稱讚的評論)
"""
def df2binary(df):
    df['index'] = df['index'].apply(lambda x: 1 if x == '稱讚' else 0)
    return df

"""(2) Multiclass classification on non-compliment comments."""
def df_without_bin(df):
    return df[df['index'] != '稱讚']