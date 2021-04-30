import argparse
import glob, os
import os.path
from sklearn.tree import DecisionTreeClassifier
from project2 import main


if __name__ == '__main__':
    
    glob_text = "/home/anjalireddytippana/proj2/cs5293sp21-project2/cs5293p21-project2/train/*.txt"
    glob_test_text = "/home/anjalireddytippana/proj2/cs5293sp21-project2/cs5293p21-project2/test/*.txt"
    
    print("getting training features \n")
    train_df_x,train_df_y = main.train_features(glob_text)
    print("training model\n")
    clf = DecisionTreeClassifier(random_state=0)
    clf_train=clf.fit(train_df_x.values,train_df_y.values)
    print("training completed\n")
    print("readcting documents from test folder\n")
    main.redact_test_folder_docs(glob_test_text)
    print("written redacted files into redacted folder\n")
    print("testing begins\n")
    result = main.unredactor('redacted/*.redacted', 'unredactor' , clf)
    print("\nreplaced redacted docs with predicted names and saved them to to_be saved folder\n")
    print(result)
