import twint
from abc import ABC, abstractmethod
from typing import List


class Scraper(ABC):
    """
    Abtract class for twitter scraper.
    """
    name = "SCRAPER"

    def __init__(self):
        self._username = None
        self._config = None
        self.configure()

    @abstractmethod
    def configure(self) -> None:
        """
        This method does scraper initial configuration of scraper
        :return:
        """
        pass

    @property
    def username(self) -> str:
        return self._username

    @username.setter
    def username(self, username: str) -> None:
        """
        Set username parameter in scraper engine for searching.
        :param username: Twitter username account
        :return: None
        """
        self._username = username

    @abstractmethod
    def do_scrape(self) -> List:
        """
        This method launchs the scraping on Twitter based on username
        :return:
        """
        pass


class TwintScraper(Scraper):
    """
    A scraper that uses twint module.
    """

    def __init__(self):
        Scraper.__init__(self)

    def configure(self) -> None:
        config = twint.Config()
        config.Limit = 100
        config.Store_object = True
        self._config = config

    def do_scrape(self) -> List:
        self._config.Username = self.username
        twint.run.Search(self._config)
        return twint.output.tweets_list
