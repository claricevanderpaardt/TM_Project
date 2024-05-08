#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


# In[17]:


books_df = pd.read_csv('book1-100k.csv', delimiter= ",")
books_df.head()


# In[10]:


ratings_df = pd.read_csv('user_rating_0_to_1000.csv', delimiter= ",")
ratings_df.head()


# In[15]:


books_df.columns


# In[30]:


merged_df = pd.merge(books_df, ratings_df, on='Name')
merged_df.head()


# In[16]:


merged_df.columns


# In[19]:


# Renaming the IDs for better interpretation of data as one is referring to the book's ID and the other one to the user rating it
merged_df.rename(columns={'Id': 'bookID'}, inplace=True)
merged_df.rename(columns={'ID': 'userID'}, inplace=True)
merged_df.columns


# In[22]:


# Number of missing values per column 
missing_per_column = merged_df.isnull().sum()
print(missing_per_column)


# In[39]:


# For ISBN and Publisher we can ignore these missing values as they are not really relevant to our analysis. 
# Language, however, can be quite important for a book recommendation system, as it directly affects user preferences 
# and accessibility. 

# Mode imputation for missing values:
mode_language = merged_df['Language'].mode()[0]
merged_df['Language'].fillna(mode_language, inplace=True)


# In[49]:


# Tokenizing and lemmatizing columns:
def tokenize_and_lemmatize(col):
    # ensures col is a string
    col = str(col)
    # Load English stopwords
    stop_words = set(stopwords.words('english'))
    # Initialize the Lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Tokenize the column
    tokens = word_tokenize(col.lower())
    # Filter out stopwords and non-alphabetic characters
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Lemmatize each filtered token
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return lemmatized_tokens


# In[51]:


# Apply the tokenizing and lemmatization function to the "text" cols:
merged_df['Processed_Name'] = merged_df['Name'].apply(tokenize_and_lemmatize)
merged_df['Processed_Authors'] = merged_df['Authors'].apply(tokenize_and_lemmatize)
# Not that relevant for our analysis but just in case we use it later on:
merged_df['Processed_Publisher'] = merged_df['Publisher'].apply(tokenize_and_lemmatize)
merged_df['Processed_Rating'] = merged_df['Rating_y'].apply(tokenize_and_lemmatize)


# In[54]:


print(merged_df[['Rating_y', 'Processed_Rating']].head())


# In[55]:


# Splitting into a training and validation set:
df_train_set, df_valid_set = train_test_split(merged_df, test_size=0.2, random_state=42, shuffle=True) # random state for reproducibility

