**Name: Anjali Reddy Tippana**

**Project descripiton:**

In this project, I'm going to unredact the redacted names in text files.  

**How to run:**

To run the code first you need to intsall and setup python3 environment with packages mentioned in pipfile or requirements.txt

**Commands used to install python3 and setup python environment:**

pyenv install python 3.8.6

pyenv virtualenv 3.8.6 folder_name

**Commands used to install packages:**

pipenv install nltk

pipenv install pandas  

pipenv install numpy

pipenv install scikit-learn

pipenv install pytest 

**command used to run:**

pipenv run python unredactor.py


**How to test:**

pipenv run python -m pytest.

**Assumptions:**

I'm assuming that all the famous person names that are redacted are unredacted and written into a file.


**Functions used:**

**get_entity(text)**
 
This function will take a txt and labels all the names of people as PERSON using nltk after word tokenizing the given text. This function returns list of all names.     

**redacted_doc(names_list,text)**

This function is going to take list of all names and txt file as input parameters. It masks all words present in txt file that are also present in the names_list with a unicode char. This function returns this redacted text.

**redact_test_folder_docs(glob_test_text)**

This function is going to take a path of a folder which has test text documents and using glob operator it is going to read all the files in that location and by calling  get_entity(text) and redacted_doc(names_list,text) functions it redacts the text files and then creates, writes those redacted text files into another files with the same file name but replaces .txt with .redacted. This function does not return anything. 

** get_name_features(glob_text)**

This function is going to take a path of a folder which has train text documents and using glob operator it is going to read all the files in that location and by calling  get_entity(text) it gets names_list and then it gets features of names like name, name_length, number of words in a name and target value name, rating of that text file(using regex) and appends them into a list of dictionaries with these features as key value pairs. This function returns list of features.

**train_features(glob_text)**

This function is going to take a path of a folder which has train text documents and using glob operator it is going to read all the files in that location and by calling get_name_features(glob_text) function it is going to get all features of names and then using pandas dataframes I got train_df_x dataframe with features "name length", "no_of_words", "rating" and train_df_y dataframe with name itself. This function returns these two dataframes.  

**test_features(text, thefile)** 

This function is going to take name of the redacted file and  redacted txt  as input parameters . This function uses re.findall to match the redacted blocks and also it prepares features that are required for testing like start and end index, name length, number of words and then appends all these features to a list. This function returns pandas dataframe of this list and also redacted document.    

**unredactor(redacted_files_folder_path, destination_folder_path , clf):**

This function is going to take file path of redacted files, path for folder to place all unredacted files and a Decision tree classifier model object as input parameters. Then by calling test_features(text, thefile) it gets all features of redacted block and then using decision tree classifier oblect , it is going to predict redacted names and writes them in the place of redacted block and then writes these files into a into destination_folder_path with same file names.    
    

**References:**

https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
https://www.w3schools.com/python/pandas/pandas_dataframes.asp
https://www.geeksforgeeks.org/python-os-path-splitext-method/
https://www.tutorialspoint.com/How-do-we-use-re-finditer-method-in-Python-regular-expression
https://docs.github.com/en/github/writing-on-github/basic-writing-and-formatting-syntax






