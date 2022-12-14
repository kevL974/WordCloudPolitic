import twint  # configuration
import pandas as pd
import nltk
from nltk.corpus import stopwords
from itertools import chain
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pandas import DataFrame
from typing import List

exclure_mots = ['je', 'j\'', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
                'me', 'te', 'le', 'lui', 'la', 'les', 'leur', 'eux', 'moi', 'toi',
                'celui', 'celle', 'ceux', 'celles', 'ceci', 'cela', 'ce', 'c\'', 'ça', 'celui-ci',
                'celui-là', 'celle-ci', 'celle-là', 'ceux-ci', 'ceux-là', 'celles-ci', 'celles-là',
                'qui', 'que', 'quoi', 'dont', 'où',
                'lequel', 'laquelle', 'duquel', 'auquel', 'lesquelles', 'desquelles', 'auxquelles',
                'auquel', 'auxquels', 'auxquelles', 'desquels'
                'le', 'l\'', 'un', 'mon', 'ton', 'son', 'notre', 'votre', 'leur', 'ce', 'cet', 'chaque',
                'la', 'une', 'ma', 'ta', 'sa', 'notre', 'votre', 'cette', 'les', 'des', 'mes', 'tes', 'ses',
                'nos', 'vos', 'leurs', 'ces', 'quelques', 'plusieurs', 'de', 'https://', 'http://', '.', ',', '/', '!', 'https', 'co']

stopwords = stopwords.words('french')
def get_deputy_name_and_surname(df: DataFrame ) -> DataFrame:
    return df[['Prénom','Nom','Groupe politique (abrégé)']]


def remove_special_character(text: str) -> str:
    text.
def preprocessing(text: str) -> str:
    low_text = text.lower()
    low_text = remove_special_character(low_text)

def transform_to_twitter_account_pattern(df: DataFrame) -> DataFrame:
    df['Initials'] = get_initials_of_surname(df)
    df['pattern1'] = df['Prénom']+df['Nom']
    df['pattern2'] = df['Initials'] + df['Nom']
    df['pattern3'] = df['Initials'].apply(str.lower) + df['Nom']
    return df


def get_initials_of_surname(df: DataFrame) -> DataFrame:
    return df['Prénom'].apply(str.split).apply(get_initials)


def get_initials(word: List) -> str:
    if len(word) == 0:
        return ''

    if len(word) == 1:
        return word[0][0]

    return word[0][0]+get_initials(word[1:])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # df_depute = pd.read_csv("ressources/depute.csv", sep=';')
    # df_depute = get_deputy_name_and_surname(df_depute)
    # df_depute = transform_to_twitter_account_pattern(df_depute)
    # df_rn = df_depute[df_depute['Groupe politique (abrégé)'] == 'RN']
    df_depute = pd.read_csv("ressources/twitter_account.csv", sep=';')
    twitter_accounts = df_depute['Username'].values.tolist()
    df_tweet = None

    wordcloud = None
    config = twint.Config()
    for username in twitter_accounts:
        try:
            config.Username = username
            config.Limit = 100  # running search
            config.Store_pandas = True
            config.Pandas = True
            twint.run.Search(config)
            test_df: DataFrame = twint.storage.panda.Tweets_df
            tweets = "".join(chain.from_iterable(test_df['tweet'].values))

            text = preprocessing(tweets)

            wordcloud = WordCloud(background_color = 'white', stopwords = stopwords, max_words = 50).generate(text.lower())
            print(text)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.show()


        except ValueError:
            print(f'{username} not found as twitter account')
