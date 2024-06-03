import os
import json
import pathlib
import logging.config
from logging import Logger


def setup_logging(logger_config_path: str):
    with open(pathlib.Path(logger_config_path)) as f:
        configuration = json.load(f)

    logging.config.dictConfig(configuration)


def create_output_folder(logger: Logger, output_folder: str):
    output_path = os.path.join(os.getcwd(), output_folder)
    if os.path.exists(output_path):
        logger.info(f'Folder {output_folder} for output files and directories already exists.')
    else:
        os.mkdir(output_path)
        logger.info(f'Created folder \'{output_folder}\' for output files and directories.')


class Config:
    def __init__(self):
        # Logger
        self.logger = logging.getLogger("sentiment_analysis")
        self.logger_config_path = 'config_logging.json'
        setup_logging(os.path.join(os.getcwd(), self.logger_config_path))
        self.log_enable_health = False
        self.log_ram = True
        self.log_cpu = True
        self.log_disk = True
        self.log_intervall = 5  # in seconds
        # General
        self.output_folder = 'output'
        create_output_folder(self.logger, self.output_folder)
        # Dataset configuration
        self.dataset = DatasetConfiguration('the-reddit-climate-change-dataset-comments.csv',
                                            ['created_utc', 'body', 'sentiment'],
                                            'dataset_per_year',
                                            overwrite_sentiment_by='vader',
                                            consider=True,
                                            preprocess=False,
                                            create_feather_files=True)
        # Configuration for different methodologies
        self.methodologies = {
            'vader': MethodologyConfiguration('vader',
                                              use_preprocessed_data=False,
                                              recalc=True),
            'word_counter': WordCounterConfiguration('word_counter',
                                                     'frequency',
                                                     use_preprocessed_data=False,
                                                     recalc=True,
                                                     max_words=50,
                                                     ngram_range=(1, 1))
        }

        # ['energy', 'government', 'science', 'technology',
        #  'carbon', 'nuclear', 'money', 'evidence', 'fossil',
        #  'co2', 'oil', 'planet', 'solar', 'wind', 'temperature',
        #  'emissions', 'war']

        # ['carbon dioxide', 'carbon emissions', 'change denial',
        #  'fossil fuel', 'global warming', 'nuclear power',
        #  'renewable energy', 'electric car', 'sea level',
        #  'carbon tax', 'extreme weather', 'greenhouse effect',
        #  'natural gas', 'clean energy', 'access pipeline',
        #  'carbon footprint']

        # Comparator configuration
        self.comparator_folder = 'comparators'
        self.comparator_filename = 'compared_to_kaggle'
        self.comparators_to_kaggle = {'vader': True,
                                      'ngram': False}
        # Plot output and configuration
        self.plot_topic_over_time = True
        self.word_counter_type = 'frequency'
        self.word_counter_filename = 'top50_ngram1-1_kaggle'
        self.vocab_file_path = os.path.join(os.getcwd(),
                                            self.output_folder,
                                            'word_counter_kaggle',
                                            f'WordCounter_{self.word_counter_type}_{self.word_counter_filename}.csv')
        self.plot_output_path = os.path.join(os.getcwd(), self.output_folder, 'plots')
        # NER configuration
        self.docbin_batch_size = 100000


class Configuration:
    def __init__(self, base_output_folder: str, consider: bool = True, recalc: bool = False):
        self.base_output_folder = base_output_folder
        self.consider = consider
        self.recalc = recalc


class MethodologyConfiguration(Configuration):
    def __init__(self, methodology: str, consider: bool = True, recalc: bool = True,
                 use_preprocessed_data: bool = False):
        super().__init__(methodology, consider, recalc)
        self.methodology = methodology
        self.use_preprocessed_data = use_preprocessed_data


class WordCounterConfiguration(MethodologyConfiguration):
    def __init__(self, methodology: str, type: str, consider: bool = True, recalc: bool = True,
                 use_preprocessed_data: bool = False, min_df: [int, float] = 1, max_df: [int, float] = 1.,
                 max_words: int = 25, ngram_range: tuple = (1, 1), vocabulary: list = None):
        super().__init__(methodology, consider, recalc, use_preprocessed_data)
        self.type = type
        self.max_words = max_words
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary = vocabulary

        if self.vocabulary:
            lens = [len(v.split(' ')) for v in self.vocabulary]
            self.ngram_range = (min(lens), max(lens))


class DatasetConfiguration(Configuration):
    def __init__(self, filename: str, cols_of_interest: list, base_output_folder: str,
                 consider: bool = True, preprocess: bool = True, create_feather_files: bool = False,
                 overwrite_sentiment_by: str = None):
        super().__init__(base_output_folder, consider)
        self.filename = filename
        self.cols_of_interest = cols_of_interest
        self.create_feather_files = create_feather_files
        self.preprocess = preprocess
        self.overwrite_sentiment_by = overwrite_sentiment_by
