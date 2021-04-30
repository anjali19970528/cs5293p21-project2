import glob
import re
import io,os
import pandas as pd

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
from sklearn.tree import DecisionTreeClassifier 


def get_entity(text):
    """Prints the entity inside of the text."""
    names_list=[]
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':            
                names_list.append((' '.join(c[0] for c in chunk)))
    return names_list

def redacted_doc(names_list,text):
    unic_char = '\u2588'
    if len(names_list) > 0:
        for i,j in enumerate(names_list):
            
            replace_block = ' '.join([len(word)*unic_char for word in word_tokenize(j)]) 
            j= r'\b' + j + r'\b'
            text=re.sub(j,replace_block,text)
    return text

def redact_test_folder_docs(glob_test_text):
    for thefile in glob.glob(glob_test_text)[:3]:
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
        names_list=get_entity(text)
        print(names_list)
        doc=redacted_doc(names_list,text)
        print(doc)
        if not os.path.exists('redacted'):
            os.makedirs('redacted')
        text_file = thefile.split("/")[-1]
        output_file = text_file.replace('.txt', '.redacted')
        print(output_file)
        file_obj = open('redacted/'+output_file, "w",encoding = "utf-8")
        file_obj.write(doc)
        file_obj.close()
        

def get_name_features(glob_text):
    name_features=[]
    for thefile in glob.glob(glob_text)[:3]:
#         print(thefile)
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            names_list=get_entity(text)
            if len(names_list)>0:
                for name in names_list:
                    names_info={}
                    names_info["name"]=name
                    names_info["name length"]=len(name)
                    names_info["no_of_words"]=len(word_tokenize(name))
                    names_info["rating"]=int(re.findall(r'_(\d{1,2}).txt',thefile)[0])
                    name_features.append(names_info)
    return name_features

def train_features(glob_text):
    name_features=get_name_features(glob_text)
    df=pd.DataFrame(name_features)
    df=df.drop_duplicates()
#     print(df)
    train_df_x = df.loc[:,["name length","no_of_words","rating"]]
    train_df_y = df.loc[:,["name"]]
    return train_df_x, train_df_y

def test_features(text, thefile):
    unic_char = '\u2588'
    redacted_list=[]
#             print(text)
    names_list=get_entity(text)
    print(names_list)
    doc=redacted_doc(names_list,text)
    rating = int(re.findall(r'_(\d{1,2}).redacted',thefile)[0])
    red_block = unic_char + r'+\s*' + unic_char + r'+'
    for match in re.finditer(red_block, doc):
        matched_block = match.string
        s = match.start()
        e = match.end()
        red_blocks_dict = {}
        red_blocks_dict['name length'] = len(matched_block)
        red_blocks_dict['no_of_words'] = len(word_tokenize(matched_block))      
        red_blocks_dict['rating'] = rating
        red_blocks_dict['start_index'] = s
        red_blocks_dict['end_index'] = e
        redacted_list.append(red_blocks_dict)
   
    test_df=pd.DataFrame(redacted_list)
    return test_df, doc


def unredactor(redacted_files_folder_path, destination_folder_path , clf):
    for thefile in glob.glob(redacted_files_folder_path):
        print(thefile)
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
        test_features_df, redacted_text = test_features(text, thefile)
        test_features_X = test_features_df[['name length', 'no_of_words', 'rating']]
        predicted_names = clf.predict(test_features_X.values)
    #     filepath = filepath.replace('redacted','unredacted')
        redacted_blocks_indices_list = list(zip(test_features_df.start_index.values.tolist(), test_features_df.end_index.values.tolist()))
        print(redacted_text)
        start_index = 0
        unredacted_text =''
        print(len(predicted_names))
        print(len(test_features_df.shape))
        for i, pred_name in enumerate(predicted_names):
            
            name_start_index = redacted_blocks_indices_list[i][0]
            name_last_index = redacted_blocks_indices_list[i][1]
            print(name_start_index, name_last_index, len(text))
            text_to_be_redacted = redacted_text.replace(redacted_text[name_start_index:name_last_index], pred_name)
            unredacted_text = unredacted_text+' '+text_to_be_redacted[start_index:name_last_index]
            start_index = name_last_index
            
        print(unredacted_text)
        if not os.path.exists(destination_folder_path):
            os.makedirs(destination_folder_path)
        output_file = thefile.split("/")[-1]
        print(output_file)
        output_file = output_file.replace('.redacted', '.txt')
        file_obj = open(destination_folder_path+'/'+output_file, "w",encoding = "utf-8")
        file_obj.write(unredacted_text)
        file_obj.close()
    
    return "success"






