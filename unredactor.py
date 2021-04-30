import argparse
import glob, os
import os.path
import main

if __name__ == '__main__':
    
    glob_text = "C:/Users/sanary/Desktop/aclImdb_v1.tar/aclImdb/train/pos/*.txt"
    glob_test_text = "C:/Users/sanary/Desktop/aclImdb_v1.tar/aclImdb/test/pos/*.txt"
    
    print("getting training features \n")
    train_df_x,train_df_y = train_features(glob_text)
    print("training model\n")
    clf = DecisionTreeClassifier(random_state=0)
    clf_train=clf.fit(train_df_x.values,train_df_y.values)
    print("training completed\n")
    print("readcting documents from test folder\n")
    redact_test_folder_docs(glob_test_text)
    print("written redacted files into redacted folder\n")
    print("testing begins\n")
    result = unredactor('redactor/*.redacted', 'unredactor/*.unredacted' , clf)
    print("\nreplaced redacted docs with predicted names and saved them to to_be saved folder\n")
    print(result)
