import twint  # configuration
import pandas as pd
import nltk
import math
from nltk.corpus import stopwords
from itertools import chain
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import numpy as np
from PIL import Image
from functools import reduce
from pandas import DataFrame, Series
from typing import List, Set, Dict

exclure_mots = ['je', 'j\'', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
                'me', 'te', 'le', 'lui', 'la', 'les', 'leur', 'eux', 'moi', 'toi',
                'celui', 'celle', 'ceux', 'celles', 'ceci', 'cela', 'ce', 'c\'', 'ça', 'celui-ci',
                'celui-là', 'celle-ci', 'celle-là', 'ceux-ci', 'ceux-là', 'celles-ci', 'celles-là',
                'qui', 'que', 'quoi', 'dont', 'où',
                'lequel', 'laquelle', 'duquel', 'auquel', 'lesquelles', 'desquelles', 'auxquelles',
                'auquel', 'auxquels', 'auxquelles', 'desquels'
                'le', 'l\'', 'un', 'mon', 'ton', 'son', 'notre', 'votre', 'leur', 'ce', 'cet', 'chaque',
                'la', 'une', 'ma', 'ta', 'sa', 'notre', 'votre', 'cette', 'les', 'des', 'mes', 'tes', 'ses',
                'nos', 'vos', 'leurs', 'ces', 'quelques', 'plusieurs', 'de', 'français', 'française', 'france', 'contre'
                'depuis', 'tous', 'tout', 'toute', 'plus']

STOP_WORDS = stopwords.words('french')
STOP_WORDS = STOP_WORDS + exclure_mots
def get_deputy_name_and_surname(df: DataFrame ) -> DataFrame:
    return df[['Prénom','Nom','Groupe politique (abrégé)']]

def get_tokens(text: str) -> List:
    return text.split()

def remove_stop_words(tokens: List, stop_words: List) -> List:
    clean_token = []
    for token in tokens:
        if token not in stop_words and len(token) > 1:
            clean_token.append(token)
    return clean_token

def remove_special_character(text: str) -> str:
    clean = text.replace("à", "a")
    clean = clean.replace("é", "e")
    clean = clean.replace("è", "e")
    clean = clean.replace("ê", "e")
    clean = clean.replace("ë", "e")
    clean = clean.replace("î", "i")
    clean = clean.replace("ï", "i")
    clean = clean.replace("ô", "o")
    clean = clean.replace("ù", "u")
    return clean

def cleaning(results: List) -> List:
    tweets = []
    for result in results:
        id = result.id
        tweet = result.tweet
        tweet = tweet.lower()
        tweet = re.sub("https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)",
                                  "", tweet)
        tweet = re.sub(r'[^\w\s]', ' ', tweet)
        tweet = re.sub(r'[\s]+', ' ', tweet)
        tweet = remove_special_character(tweet)
        tweets.append((id, tweet))
        #.map(lambda tweet: set(tweet.split(' ')))
    return tweets

def prepocessing(corpus: List) -> List:
    docs = []
    for doc in corpus:
        tokens = get_tokens(doc[1])
        tokens = remove_stop_words(tokens, STOP_WORDS)
        docs.append((doc[0], tokens))
    return docs


def compute_tf(corpus: List) -> List:
    return {doc[0]: term_frequencies(doc[1]) for doc in corpus}


def compute_idf(corpus: List, uniq_terms: List) -> List:
    N = len(corpus)
    return {doc[0]: inverse_document_frequency(doc[1], uniq_terms, N) for doc in corpus}


def compute_tf_idf(uniq_terms: Set, tf: Dict, idf: Dict) -> Dict:
    tdf_idf = {}
    for j in tf.keys():
        tf_j = tf[j]
        idf_j = idf[j]
        for i in uniq_terms:
            if i not in tdf_idf:
                tdf_idf[i] = {}

            if i in tf_j:
                tf_ij = tf_j[i]
                idf_ij = idf_j[i]
                tdf_idf[i][j] = tf_ij * idf_ij

    return tdf_idf

def term_frequencies(doc: List) -> Dict:
    nb_token_in_doc = len(doc)
    return {token: doc.count(token)/nb_token_in_doc for token in doc}

def inverse_document_frequency(text: List, uniq_terms: List, N: int) -> List:
    return {term: math.log(N/(text.count(term)+1)) for term in uniq_terms}

def get_uniq_terms(corpus: List) -> Set:
    bag_of_word = []
    for doc in corpus:
        bag_of_word = bag_of_word+doc[1]
    return set(bag_of_word)



def show(tweets : List) -> None:
    for t in tweets:
        print(t)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # df_depute = pd.read_csv("ressources/depute.csv", sep=';')
    # df_depute = get_deputy_name_and_surname(df_depute)
    # df_depute = transform_to_twitter_account_pattern(df_depute)
    # df_rn = df_depute[df_depute['Groupe politique (abrégé)'] == 'RN']
    df_depute = pd.read_csv("ressources/twitter_account.csv", sep=';')
    twitter_accounts = df_depute['Username'].values.tolist()

    wordcloud = None
    config = twint.Config()
    for username in twitter_accounts:
        try:
            config.Username = username
            config.Limit = 300  # running search
            config.Store_object = True
            twint.run.Search(config)
            results = twint.output.tweets_list

            tweets = cleaning(results)
            show(tweets)

            corpus = prepocessing(tweets)

            uniq_terms = get_uniq_terms(corpus)
            tf = compute_tf(corpus)
            idf = compute_idf(corpus, uniq_terms)

            tdf_idf = compute_tf_idf(uniq_terms, tf, idf)

            score = {term: 0 for term in uniq_terms}
            for term_i in uniq_terms:
                for doc_j in tdf_idf[term_i].keys():
                    #print("{term_i}-{doc_j} -> {score}".format(term_i=term_i, doc_j=doc_j, score=str(tdf_idf[term_i][doc_j])))
                    score[term_i] += tdf_idf[term_i][doc_j]

            sorted_score = dict(sorted(score.items(), key=lambda item: item[1],reverse=True))

            wordcloud = WordCloud(background_color = 'white', max_words = 50).generate_from_frequencies(sorted_score)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.title(username)
            plt.show()


        except ValueError:
            print(f'{username} not found as twitter account')
