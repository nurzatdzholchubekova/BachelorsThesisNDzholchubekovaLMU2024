import os
import re
from multiprocessing import Process
from pathlib import Path
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from config import Config


class Dataset:
    def __init__(self, config: Config):
        self.df = None
        self.config = config
        self.base_output_path = self.config.dataset.base_output_folder
        self.feather_path = os.path.join(os.getcwd(),
                                         self.config.output_folder,
                                         f'{self.config.dataset.base_output_folder}_kaggle')
        self.preprocessing_path = os.path.join(os.getcwd(),
                                               self.config.output_folder,
                                               f'{self.config.dataset.base_output_folder}_preprocessed')
        self.years_covered = None

    def load(self):
        self.config.logger.info("Start loading Kaggle dataset from .csv file.")

        df = pd.read_csv(self.config.dataset.filename, usecols=self.config.dataset.cols_of_interest)
        df['id'] = df.index
        self.df = df[['id', 'created_utc', 'body', 'sentiment']]

        self.config.logger.info("Finished loading Kaggle dataset from .csv file.")

    def to_feather(self):
        if not os.path.exists(self.feather_path):
            os.mkdir(self.feather_path)
            self.config.logger.info(f'Created folder \'{self.feather_path}\' for feather files.')
        else:
            self.config.logger.info(f'Folder for feather files exists at {self.feather_path}.')

        if self.df.empty:
            self.load()

        self.config.logger.info('Start converting timestamps to datetime.')
        self.df['created_utc'] = pd.to_datetime(self.df['created_utc'], unit='s')
        self.df.rename(columns={"created_utc": "datetime"}, inplace=True)
        self.config.logger.info('Finished converting timestamps to datetime.')

        self.years_covered = self.df['datetime'].dt.year.unique()
        self.config.logger.info('Determined unique years in dataset.')
        for year in (pbar := tqdm(self.years_covered, total=len(self.years_covered), ncols=120)):
            pbar.set_description(f'Writing .feather file for year {year}')
            df_year = self.df[self.df['datetime'].dt.year == year]
            self.df = self.df.drop(df_year.index)
            path = os.path.join(self.feather_path,
                                f'{self.config.dataset.filename.replace(".csv", "")}_{year}_kaggle.feather')
            df_year.to_feather(path, compression='lz4')

        self.config.logger.info('Finished writing feather files..')
        del self.df

    def preprocess(self):
        # Only feather files are preprocessed
        assert (os.path.exists(self.feather_path)
                and sum(1 for _ in Path(self.feather_path).glob('*.feather')) > 0), \
            'Preprocessing requires feather files. Modify the provided path or create feather files.'

        # Create folder for preprocessed files
        if not os.path.exists(self.base_output_path):
            self.config.logger.info(f'Creating folder for preprocessed feather files at {self.base_output_path}.')
            os.mkdir(self.base_output_path)
            self.config.logger.info('Created folder for preprocessed feather files.')
        else:
            self.config.logger.info(f'Folder for preprocessed feather files exists at {self.base_output_path}.')

        processes = []
        feather_files = [p for p in Path(self.feather_path).glob('*.feather')]
        for i, file_path in enumerate(feather_files):
            year = re.findall(r'(?<!\d)\d{4}(?!\d)', file_path.name)[-1]
            p = Process(target=self._preprocess, args=[file_path, year, i])
            p.daemon = True
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        os.system('cls' if os.name == 'nt' else 'clear')
        self.config.logger.info(f'Finished preprocessing and writing all feather files.')

    def _preprocess(self, path: Path, year: str, tqdm_pos: int):
        logger = Config().logger

        # Initialize lemmatizer
        lemmatizer = WordNetLemmatizer()

        # Read feather file
        df = pd.read_feather(path)
        # Create output path
        output_path = os.path.join(self.preprocessing_path, path.name.replace('_kaggle', '_preprocessed'))
        size = df.shape[0]

        logger.info(f'Start preprocessing {path.name} with tqdm_pos {tqdm_pos}.')
        # Progress bar seems to be bugged, but it's the best I can currently get.
        # Multiple processes feature is currently not fully supported by tqdm.
        for i, row in tqdm(df.iterrows(), total=size, desc=f'Preprocessing {year}', position=tqdm_pos, leave=False,
                           ncols=120):
            # Convert text to lowercase
            text = row['body'].lower()
            # Remove punctuation
            text = re.sub(r'[^\w\s]', '', text)
            # Remove URLs
            text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
            # Remove user mentions, hashtags and line breaks
            text = re.sub(r'\@\w+|\#|\n|\r', '', text)
            # Lemmatize and remove stopwords
            text = text.split()
            text = ' '.join([lemmatizer.lemmatize(w) for w in text if w not in stopwords.words('english')])
            # Write back to row
            df.at[row['id'], 'body'] = text

        logger.info(f'Finished preprocessing {path.name}. Writing to {output_path}.')
        df.to_feather(output_path, compression='lz4')
        logger.info(f'Finished writing to {output_path}.')

    def load_raw(self, year: str):
        return self._load_from(self.feather_path, year)

    def load_preprocessed(self, year: str):
        return self._load_from(self.preprocessing_path, year)

    def get_years_covered(self):
        return [re.findall(r'(?<!\d)\d{4}(?!\d)', p.name)[-1]
                for p in Path(self.feather_path).glob('*.feather')]

    def _load_from(self, directory: str, year: str):
        load_type = 'preprocessed' if 'preprocessed' in directory else 'raw'

        if not self.years_covered or len(self.years_covered) == 0:
            self.years_covered = [re.findall(r'(?<!\d)\d{4}(?!\d)', p.name)[-1]
                                  for p in Path(directory).glob('*.feather')]

        assert year in self.years_covered, f'The data for year {year} are not available.'

        path = [p for p in Path(directory).glob('*.feather') if year in p.name]

        assert len(path) >= 1, f'There is no feather file with {load_type} data from year {year}.'
        assert len(path) == 1, f'There cannot be more than one feather file with {load_type} data from year {year}.'

        return pd.read_feather(path[0])
