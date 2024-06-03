import re
from multiprocessing import Process
from pathlib import Path

import pandas as pd
import os
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from methodologies.methodology import Methodology
from config import Config


class Vader(Methodology):
    def __init__(self, config: Config):
        super().__init__('vader', config.methodologies['vader'])
        folder_prefix = '_preprocessed' if self.config.use_preprocessed_data else '_kaggle'
        self.analyzer = SentimentIntensityAnalyzer()
        self.data_files_path = os.path.join(os.getcwd(),
                                            config.output_folder,
                                            f'{config.dataset.base_output_folder}{folder_prefix}')
        self.vader_output_path = os.path.join(os.getcwd(),
                                              config.output_folder,
                                              f'{self.config.base_output_folder}{folder_prefix}')
        self.logger = config.logger
        self.years_covered = None

    def polarity(self):
        if not os.path.exists(self.vader_output_path):
            os.mkdir(self.vader_output_path)
            self.logger.info(f'Created folder \'{self.vader_output_path}\' for vader files.')
        else:
            self.logger.info(f'Folder for vader files exists at {self.vader_output_path}.')

        assert (os.path.exists(self.data_files_path)
                and sum(1 for _ in Path(self.data_files_path).glob('*.feather')) > 0), \
            'Vader polarity calculation requires feather files. Modify the provided path or create feather files.'

        processes = []
        feather_files = [p for p in Path(self.data_files_path).glob('*.feather')]
        for i, file_path in enumerate(feather_files):
            year = re.findall(r'(?<!\d)\d{4}(?!\d)', file_path.name)[-1]
            p = Process(target=self._polarity, args=[file_path, year, i])
            p.daemon = True
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        os.system('cls' if os.name == 'nt' else 'clear')
        self.logger.info(f'Finished calculating vader polarity and writing all feather files.')

    def _polarity(self, path: Path, year: str, tqdm_pos: int):
        # Read feather file
        df = pd.read_feather(path)
        # Create output path
        output_path = os.path.join(self.vader_output_path, path.name)
        # Get some auxilary data
        data_type = 'kaggle' if path.name.endswith('kaggle.feather') else 'preprocessed'
        size = df.shape[0]

        values = []
        for i, row in tqdm(df.iterrows(), total=size, position=tqdm_pos, leave=False, ncols=120,
                           desc=f'VADER polarity for {data_type} data of {year}',
                           ):
            vader_polarity = self.analyzer.polarity_scores(row['body'])
            values.append([i] + list(vader_polarity.values()))

        logger = Config().logger
        df = pd.DataFrame(values, columns=['id'] + list(vader_polarity.keys()))
        logger.info(f'Finished preprocessing {path.name}. Writing to {output_path}.')
        df.to_feather(output_path, compression='lz4')
        logger.info(f'Finished writing to {output_path}.')

    def write_sentiment_to_source(self):
        assert (os.path.exists(self.data_files_path)
                and sum(1 for _ in Path(self.data_files_path).glob('*.feather')) > 0), \
            'Replacing Kaggle sentiment requires feather files from Kaggle dataset. Modify the provided path or create feather files.'

        assert (os.path.exists(self.data_files_path)
                and sum(1 for _ in Path(self.data_files_path).glob('*.feather')) > 0), \
            'Replacing VADER sentiment requires feather files with VADER sentiment. Modify the provided path or create feather files.'

        feather_files_kaggle = [p for p in Path(self.data_files_path).glob('*.feather')]
        feather_files_vader = [p for p in Path(self.vader_output_path).glob('*.feather')]

        assert len(feather_files_kaggle) == len(feather_files_vader), 'Amount of Kaggle and VADER files must match.'

        self.logger.info(f'Starting to replace Kaggle sentiment with VADER sentiment.')

        for i in range(len(feather_files_kaggle)):
            year = re.findall(r'(?<!\d)\d{4}(?!\d)', str(feather_files_kaggle[i]))[-1]
            self.logger.info(f'Overwriting Kaggle sentiment with VADER sentiment for {year}.')
            # Read existing data
            df_kaggle = pd.read_feather(feather_files_kaggle[i])
            df_vader = pd.read_feather(feather_files_vader[i])
            df_vader.index = df_vader['id']
            # Overwrite Kaggle sentiment with VADER score
            df_kaggle['sentiment'] = df_vader['compound']
            # Write back Kaggle .feather file
            df_kaggle.to_feather(feather_files_kaggle[i], compression='lz4')

        self.logger.info(f'Finished to replace Kaggle sentiment with VADER sentiment.')

    def load_vader_file(self, year: str):
        if not self.years_covered:
            self.years_covered = [re.findall(r'(?<!\d)\d{4}(?!\d)', p.name)[-1]
                                  for p in Path(self.vader_output_path).glob('*.feather')]

        assert year in self.years_covered, f'The data for year {year} are not available.'

        path = [p for p in Path(self.vader_output_path).glob('*.feather') if year in p.name]

        assert len(path) >= 1, f'There is no feather file with vader scores from year {year}.'
        assert len(path) == 1, f'There cannot be more than one feather file with vader scores from year {year}.'

        return pd.read_feather(path[0])
