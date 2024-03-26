# step 1: extract features from the data (preprocessing) 
# step 2: split the dataset into train (10%) and test (90%)
# step 3: train the SVM model on the training data
# step 4: caculate models confidence on the test data and decide which ones can be labeled
# step 5: choose 2% samples for human annotation
# step 6: re-train the model with the new annotated data
# step 7: if any sample is unlabeled, go to step 4


import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
from svm_classifier_function import preprocessing
import os
import streamlit as st


csv_path__ = "APP_TEST.csv"
sub_csv_path__ = "sub_APP_TEST.csv"
#*************************************utils**************************************
def get_image_files():
    df = pd.read_csv(sub_csv_path__)
    return list(df["generated"])

def load_labels():
    return pd.read_csv(sub_csv_path__)

def update_labels(df_labels, image_file, selected_label):
    if image_file in df_labels["generated"].values:
        if selected_label == "Ok":
            df_labels.loc[df_labels["generated"] == image_file, "Label"] = 1
        elif selected_label == "Lost":
            df_labels.loc[df_labels["generated"] == image_file, "Label"] = 0
        df_labels.loc[df_labels["generated"] == image_file, "human-machine"] = 'h'
    df_labels.to_csv(sub_csv_path__, index=False)
#********************************************************************************
class status_monitor:
    def __init__(self):
        self.ARR = {"create_csv_file": "",
                    "preprocessing2": "",
                    "create_sub_dataset": "",
                    "model_train": "",
                    "predict": "",
                    "merge_datasets": "",
                    "check_sub_dataset_compeleted": "",
                    "model_train": ""}

STATUS = status_monitor()
#********************************************************************************

def create_csv_file(org_img_path, gen_img_path):
    if os.path.exists(csv_path__):
        print("CSV file already exists!")
        return "CSV file already exists!"
    list_org = os.listdir(org_img_path)
    list_gen = os.listdir(gen_img_path)
    org_list = []
    gen_list = []
    for i in range(len(list_gen)):
        if list_gen[i].find("_torchvision") != -1:
            org = list_gen[i].replace("_torchvision", "")
            for j in range(len(list_org)):
                if org == list_org[j]:
                    org_list.append(str(list_org[j]))
                    gen_list.append(str(list_gen[i]))
                    break
    hm = ["m" for i in range(len(org_list))]
    df = pd.DataFrame({
        "original" : org_list,
        "generated" : gen_list, 
        "Lable" : list(-1*np.ones(len(org_list))),
        "human-machine" : hm,
    })
    df.to_csv(csv_path__)
    s = f'CSV file "{csv_path__}" was created successfully!'
    print(s)
    return s


def preprocessing2(org_img_path, gen_img_path):
    df = pd.read_csv(csv_path__)
    size = df.shape[0]
    if "PSNR" in df.columns:
        print("Preprocessing is already done!")
        return "Preprocessing is already done!"
    psnr = []
    ssim = []
    cpl = []
    cs = []
    for i in tqdm(range(size)):
        psnr_, ssim_, cpl_, cs_ = preprocessing(str(org_img_path+"\\"+df["original"][i]), str(gen_img_path+"\\"+df["generated"][i]))
        psnr.append(psnr_)
        ssim.append(ssim_)
        cpl.append(cpl_)
        cs.append(cs_)
    df["PSNR"] = psnr 
    df["SSIM"] = ssim 
    df["CPL"] = cpl 
    df["CS"] = cs 
    df.to_csv(csv_path__)
    print("size: ", size)
    print("Preprocessing is done successfully!")
    return "Preprocessing is done successfully!"


def create_sub_dataset(state, split1=0.2, split2=0.05):
    df = pd.read_csv(csv_path__)
    size = df.shape[0]
    if state["sub_dataset"] == 1:
        df2 = df[:int(size*split1)]
        df2.to_csv(sub_csv_path__)

    elif state["sub_dataset"] == 2:
        new_df = df.loc[df["Label"] == -1] 
        if new_df.shape[0] > int(size*split2):
            new_df = new_df[:int(size*split2)]
        new_df.to_csv(sub_csv_path__)
        # print("dgh: new sub-dataset shape is: ", new_df.shape) 
    print("New sub dataset was created successfully!")
    return "New sub dataset was created successfully!"


def handle_training():
    df = pd.read_csv(csv_path__)
    size = df.shape[0]

    #create training data
    x_train = []
    y_train = []
    for i in range(size):
        if df["Label"][i] != -1 and df["human-machine"][i] == 'h':
            x_train.append(np.array([df["PSNR"][i], df["SSIM"][i], df["CPL"][i], df["CS"][i]]))
            y_train.append(df["Label"][i])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    # print("dgh: ", x_train.shape, y_train.shape)

    #train the model
    model, s1 = model_train(x_train, y_train)

    #evaluate the model
    # __new_model_eval__(model, csv_path__, len(y_train))

    #prediction and automatic labeling
    _, s2 = predict(model, csv_path__, threshold=0.9)
    return s1, s2
    

def merge_datasets():
    df_main = pd.read_csv(csv_path__)
    df_sub = pd.read_csv(sub_csv_path__)
    l =  np.copy(df_main["Label"])
    hm = np.copy(df_main["human-machine"])
    for i in range(len(df_sub)):
        for j in range(len(df_main)):
            if df_sub["generated"][i] == df_main["generated"][j]:
                l[j] = df_sub["Label"][i]
                hm[j] = df_sub["human-machine"][i]
                break
    df_main["Label"] = l
    df_main["human-machine"] = hm
    df_main.drop(df_main.columns[df_main.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    df_main.to_csv(csv_path__)
    print("Sub-dataset and the main dataset are merged!")
    return "Sub-dataset and the main dataset are merged!"


def check_sub_dataset_compeleted():
    df = pd.read_csv(sub_csv_path__)
    if -1 in df["Label"].values:
        return 0
    else:
        print("Sub dataset is compelete! :)")
        return 1
    

def model_train(x_train, y_train):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True, random_state=42))
    clf.fit(x_train, y_train)
    # print("dgh: model training is done! ")
    return clf, "Model training is done! "


def predict(model, df_name, threshold=0.95):
    #this function uses the trained model to label the data that we are confident about.
    #it writes the labels in the csv file.
    df = pd.read_csv(df_name)

    cnt = 0
    labels = np.copy(df["Label"])
    for i in range(len(labels)):
        if labels[i] == -1 and df["human-machine"][i] == 'm':
            x = np.array([[df["PSNR"][i], df["SSIM"][i], df["CPL"][i], df["CS"][i]]])
            pred = model.predict_proba(x)
            if pred[0, 1] > threshold:
                labels[i] = 1
                cnt += 1
            elif pred[0, 0] > threshold:
                labels[i] = 0
                cnt += 1
    df["Label"] = labels
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    df.to_csv(df_name)
    s1 = str(str(cnt) + " new data are labeled automatically!")
    print(s1)
    s2 = str(str(number_of_unlabeled_data(df_name)) + " data are unlabeled. \n\n")
    print(s2)
    return cnt, str(s1 + " ,  " + s2)
            

def number_of_unlabeled_data(df_name):
    df = pd.read_csv(df_name)
    return np.sum(df["Label"] == -1)


def __new_model_eval__(model, df_name, n):
    pass
    #it should compare the data to human labels

    # df = pd.read_csv(df_name)
    # x_test = np.array([np.array([df["PSNR"][i], df["SSIM"][i], df["CPL"][i], df["CS"][i]]) for i in range(n)])
    # y_test = np.array(df["Label"][:n])
    # ans = model.predict(x_test)
    # print("Accuracy of this model is: ", 100*np.sum(ans == y_test)/n)


def report():
    ## final eval
    df = pd.read_csv(csv_path__)
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    df.to_csv(csv_path__)
    number_of_h = len(df.loc[df["human-machine"] == "h"])
    number_of_m = len(df.loc[df["human-machine"] == "m"])
    st.write("📑   FINAL REPORT")
    st.write("+ Number of automatically labeled data: ", str(number_of_h), " out of ", str(number_of_h + number_of_m))
    st.write("+ Numan effort is: ", str(100*number_of_h/(number_of_h + number_of_m))[:4])
    if "true_label" in df.columns:
        df2 = df.loc[df["human-machine"] == 'm'] 
        correct = np.sum(df2["true_label"] == df2["Label"])
        st.write("+ Accuracy: ", str(100*correct/len(df2["true_label"]))[:4])


    

