import twint  # configuration
import pandas as pd
import nltk
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
                'nos', 'vos', 'leurs', 'ces', 'quelques', 'plusieurs', 'de', "français", "française", "france", "contre"]

STOP_WORDS = stopwords.words('french')
def get_deputy_name_and_surname(df: DataFrame ) -> DataFrame:
    return df[['Prénom','Nom','Groupe politique (abrégé)']]

def get_tokens(text: str) -> List:
    return text.split()

def remove_stop_words(tokens: List, stop_words: List) -> List:
    clean_token = []
    for token in tokens:
        if token not in stop_words:
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
        tweet: str = result.tweet
        tweet = tweet.lower()
        tweet = re.sub("https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)",
                                  "", tweet)
        tweet = re.sub(r'[^\w\s]', ' ', tweet)
        tweet = re.sub(r'[\s]+', ' ', tweet)
        tweet = remove_special_character(tweet)
        tweets.append(tweet)
        #.map(lambda tweet: set(tweet.split(' ')))
    return tweets

def prepocessing(corpus: List) -> List:
    docs = []
    for doc in corpus:
        tokens = get_tokens(doc)
        tokens = remove_stop_words(tokens, STOP_WORDS)
        docs.append(tokens)
    return docs

def compute_tf(corpus: List) -> List:
    return [term_frequencies(doc) for doc in corpus]

def compute_idf(corpus: List, uniq_terms: List) -> List:
    df = {term: corpus.count(term) for term in uniq_terms}


def term_frequencies(doc: List) -> Dict:
    nb_token_in_doc = len(doc)
    return {token: doc.count(token)/nb_token_in_doc for token in doc}

def get_uniq_terms(corpus: List) -> Set:
    bag_of_word = []
    for doc in corpus:
        bag_of_word = bag_of_word+doc
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
    df_depute = pd.read_csv("ressources/twitter_account_dev.csv", sep=';')
    twitter_accounts = df_depute['Username'].values.tolist()

    wordcloud = None
    config = twint.Config()
    for username in twitter_accounts:
        try:
            config.Username = username
            config.Limit = 5  # running search
            config.Store_object = True
            twint.run.Search(config)
            results = twint.output.tweets_list

            tweets = cleaning(results)
            show(tweets)

            corpus = prepocessing(tweets)

            uniq_terms = get_uniq_terms(corpus)
            tf = compute_tf(corpus)
            idf = compute_idf(corpus)





            # s_pre_tweets = preprocessing(s_tweets)
            #
            # uniqwords = reduce(lambda x, y: x.union(y), s_pre_tweets)
            #
            # set_stopwords_fr = set(stopwords_fr)
            # clean_uniqwords = uniqwords - set_stopwords_fr
            # s_uniqwords = Series(list(set_stopwords_fr))
            #
            # corpus_collection = s_pre_tweets.values
            #
            # word_count_collection = []
            # for corpus in corpus_collection :
            #     count_words_corpus = dict.fromkeys(set_stopwords_fr, 0)
            #     for word in corpus:
            #         if word in count_words_corpus:
            #             count_words_corpus[word] += 1
            #     word_count_collection.append(count_words_corpus)
            
            # tweets = "".join(chain.from_iterable(test_df['tweet'].values))
            # stopwords_fr = stopwords_fr + exclure_mots
            # text = preprocessing(tweets)
            # print(text)
            # wordcloud = WordCloud(background_color = 'white', stopwords = stopwords_fr, max_words = 20).generate(text)
            # plt.imshow(wordcloud)
            # plt.axis("off")
            # plt.show()


        except ValueError:
            print(f'{username} not found as twitter account')
