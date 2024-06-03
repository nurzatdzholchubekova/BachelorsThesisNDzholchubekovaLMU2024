import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from config import Config
from methodologies.methodology import Methodology

pd.options.mode.chained_assignment = None  # default='warn'


class WordCounter(Methodology):
    def __init__(self, name: str, config: Config):
        super().__init__(name, config.methodologies['word_counter'])
        cfg = config.methodologies['word_counter']
        self.folder_prefix = '_preprocessed' if self.config.use_preprocessed_data else '_kaggle'
        self.vectorizer = CountVectorizer(stop_words='english', lowercase=True, max_features=cfg.max_words,
                                          vocabulary=cfg.vocabulary, ngram_range=cfg.ngram_range,
                                          min_df=cfg.min_df, max_df=cfg.max_df)
        self.data_files_path = os.path.join(os.getcwd(),
                                            config.output_folder,
                                            f'{config.dataset.base_output_folder}{self.folder_prefix}')
        self.word_count_output_path = os.path.join(os.getcwd(),
                                                   config.output_folder,
                                                   f'{self.config.base_output_folder}{self.folder_prefix}')
        self.logger = config.logger
        self.max_words = cfg.max_words
        self.ngram_range = cfg.ngram_range
        self.vocabulary = cfg.vocabulary

    def perform(self, type: str) -> pd.DataFrame:
        assert type in ['frequency', 'sentiment'], 'Valid types for word counter are "frequency" and "sentiment"'

        if type == 'sentiment':
            assert self.vocabulary, 'Vocabulary cannot be empty for type "sentiment"'

        if not os.path.exists(self.word_count_output_path):
            os.mkdir(self.word_count_output_path)
            self.logger.info(f'Created folder \'{self.word_count_output_path}\' for word count related files.')
        else:
            self.logger.info(f'Folder for word count related files exists at {self.word_count_output_path}.')

        assert (os.path.exists(self.data_files_path)
                and sum(1 for _ in Path(self.data_files_path).glob('*.feather')) > 0), \
            'Word count calculation requires feather files. Modify the provided path or create feather files.'

        result_dict = {}
        frequencies = []
        feather_files = [p for p in Path(self.data_files_path).glob('*.feather')]
        for file_path in (pbar := tqdm(feather_files, total=len(feather_files), ncols=120)):
            # Get current year
            year = re.findall(r'(?<!\d)\d{4}(?!\d)', file_path.name)[-1]
            #
            pbar.set_description(f'{type.capitalize()} calculation for year {year}')
            # Read feather file
            df = pd.read_feather(file_path)
            #
            output = self.vectorizer.fit_transform(df['body'])
            #
            df_result = pd.DataFrame(data=output.toarray(), columns=self.vectorizer.get_feature_names_out())
            frequencies.append(df_result.shape[0])
            if type == 'frequency':
                result_dict[year] = df_result.sum().to_dict()
            elif type == 'sentiment':
                df = df.reset_index()
                df_result = pd.merge(df_result[df_result.any(axis=1)], df['sentiment'],
                                     left_index=True, right_index=True)
                for column in df_result.columns[:-1]:  # Exclude the last column during the iteration
                    df_result[column] = df_result[column].astype(np.float64)
                    df_result.loc[df_result[column] > 0, column] = df_result['sentiment']

                df_result = df_result.drop(columns='sentiment')
                # Only divide by number of values != 0. Otherwise, the mean also includes documents that don't
                # have the vocabulary in it.
                result_dict[year] = (df_result.sum() / np.count_nonzero(df_result, axis=0)).to_dict()

        self.logger.info(f'Finished calculating word counter for {type} and writing to .csv file.')

        df_result = pd.DataFrame.from_dict(result_dict).T
        df_result = df_result.fillna(-1)

        if type == 'frequency':
            df_result = df_result.astype(np.int64)
            df_result['total_frequency'] = frequencies

        infos = f'top{self.max_words}_ngram{self.ngram_range[0]}-{self.ngram_range[1]}' \
            if self.vocabulary is None \
            else f'vocab{len(self.vocabulary)}_ngram{self.ngram_range[0]}-{self.ngram_range[1]}'
        output_file_path = os.path.join(self.word_count_output_path, f'WordCounter_{type}_{infos}{self.folder_prefix}.csv')
        df_result.to_csv(output_file_path, index=True, sep=';')

        return df_result
