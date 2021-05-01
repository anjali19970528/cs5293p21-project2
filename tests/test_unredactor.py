import pytest
import io,os
import sys
sys.path.insert(0,"/home/anjalireddytippana/proj2/cs5293sp21-project2/cs5293sp21-project2/project2")
from project2 import main


glob_text = "/home/anjalireddytippana/proj2/cs5293sp21-project2/cs5293p21-project2/train/0_9.txt"
text ='''Bromwell High is a cartoon comedy. It ran at the same time as some other programs about school life, such as "Teachers". My 35 years in the teaching profession lead me to believe that Bromwell High's satire is much closer to reality than is "Teachers". The scramble to survive financially, the insightful students who can see right through their pathetic teachers' pomp, the pettiness of the whole situation, all remind me of the schools I knew and their students.
    When I saw the episode in which a student repeatedly tried to burn down the school, I immediately recalled ......... at .......... High.
    A classic line: INSPECTOR: I'm here to sack one of your teachers. STUDENT: Welcome to Bromwell High.
    I expect that many adults of my age think that Bromwell High is far fetched. What a pity that it isn't!'''

def test_get_entity():
    names_list = main.get_entity(text)
    assert(len(names_list)) == 3

sample_doc="she is a good girl"
names_list = ['she']
def test_redacted_doc():
    redacted_nlp_doc= main.redacted_doc(names_list,sample_doc)
#     print(redacted_nlp_doc)
    assert redacted_nlp_doc==("\u2588\u2588\u2588 is a good girl")

test_name_features=[{'name': 'Bromwell', 'name length': 8, 'no_of_words': 1, 'rating': 9}, {'name': 'Bromwell High', 'name length': 13, 'no_of_words': 2, 'rating': 9}, {'name': 'Bromwell High', 'name length': 13, 'no_of_words': 2, 'rating': 9}]
#glob_text = "C:/Users/sanary/Desktop/aclImdb_v1.tar/aclImdb/train/pos/0_9.txt"
def test_get_name_features():
    name_features = main.get_name_features(glob_text)
    print(name_features)
    assert test_name_features == name_features

#glob_text = "/home/anjalireddytippana/proj2/cs5293sp21-project2/cs5293p21-project2/train/0_9.txt"
#glob_text = "C:/Users/sanary/Desktop/aclImdb_v1.tar/aclImdb/train/pos/0_9.txt"
train_df_x=['name length', 'no_of_words', 'rating']
train_df_y=['name']
def test_train_features():
    X,Y = main.train_features(glob_text)
    assert(X.columns.tolist()==train_df_x)
    assert(Y.columns.tolist()==train_df_y)

#glob_text = "C:/Users/sanary/Desktop/aclImdb_v1.tar/aclImdb/train/pos/0_9.txt"
test_df = ['name length', 'no_of_words', 'rating','start_index','end_index']
def test_test_features():
    with io.open("/home/anjalireddytippana/proj2/cs5293sp21-project2/cs5293p21-project2/redacted/0_10.redacted", 'r', encoding='utf-8') as fyl:
        text = fyl.read()
    
    K = main.test_features(text,"/home/anjalireddytippana/proj2/cs5293sp21-project2/cs5293p21-project2/redacted/0_10.redacted")

    assert(K[0].columns.tolist() == test_df)

