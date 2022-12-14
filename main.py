import twint  # configuration
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pandas import DataFrame
from typing import List


def get_deputy_name_and_surname(df: DataFrame ) -> DataFrame:
    return df[['Prénom','Nom','Groupe politique (abrégé)']]


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

    df_depute = pd.read_csv("ressources/depute.csv", sep=';')
    df_depute = get_deputy_name_and_surname(df_depute)
    df_depute = transform_to_twitter_account_pattern(df_depute)
    df_rn = df_depute[df_depute['Groupe politique (abrégé)'] == 'RN']
    twitter_accounts = df_rn.head(20).values.tolist()
    df_tweet = None
    for i in twitter_accounts:
        n = 4
        while n <= 6:

            try:

                username = i[n]
                config = twint.Config()
                config.Username = username
                config.Limit = 100  # running search
                config.Store_pandas = True
                config.Pandas = True
                twint.run.Search(config)
                test_df : DataFrame = twint.storage.panda.Tweets_df
                if test_df.empty == False:
                    if df_tweet is None:
                        df_tweet = test_df
                    else:
                        df_tweet = pd.concat(df_tweet,test_df)

                print(df_tweet)
                break

            except ValueError:
                print(f'{username} not found as twitter account')
                n += 1


# wordcloud = WordCloud(background_color = 'white', stopwords = exclure_mots, max_words = 50).generate(text)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show();