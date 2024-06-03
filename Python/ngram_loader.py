import requests


class NGramLoader:
    def __init__(self):
        self.url = "https://books.google.com/ngrams/json"

    def load(self, ngram: str, year_start: int, year_end: int) -> dict[int, int]:
        query_params = {
            "content": ngram,
            "year_start": year_start,
            "year_end": year_end,
            "corpus": 26,
            "smoothing": 0,
            "case_insensitive": True
        }

        response = requests.get(url=self.url, params=query_params).json()

        return {year: freq for year, freq in zip(range(year_start, year_end + 1), response[0]['timeseries'])}
