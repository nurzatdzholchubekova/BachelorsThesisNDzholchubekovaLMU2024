import math
import os
import re
import time
import psutil
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from multiprocessing import Process
import plotly.express as px
from config import Config
from pathlib import Path
from dataset import Dataset
from methodologies.word_counter import WordCounter
from tqdm import tqdm
from methodologies.vader import Vader
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from ngram_loader import NGramLoader
import spacy
from spacy import displacy
from spacy.matcher import Matcher
from spacy.tokens import Span, DocBin
from wordcloud import WordCloud
from nltk.corpus import stopwords
import string
import networkx as nx
from collections import defaultdict, Counter
from spacy.cli import download

# download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')
matplotlib.use('Qt5Agg')

# Set global font sizes
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['font.size'] = 14


def monitor_pc_health():
    config = Config()
    logger_health = config.logger

    while True:
        if config.log_ram:
            memory_usage = psutil.virtual_memory().percent
            if 60 >= memory_usage < 75:
                logger_health.info(f'Memory (RAM) usage is {memory_usage}%!')
            elif memory_usage < 90:
                logger_health.warning(f'Memory (RAM) usage is {memory_usage}%!')
            else:
                logger_health.critical(f'Memory (RAM) usage is {memory_usage}%!')

        if config.log_cpu:
            cpu_usage = psutil.cpu_percent()
            if 60 >= cpu_usage < 75:
                logger_health.info(f'CPU usage is {cpu_usage}%!')
            elif cpu_usage < 90:
                logger_health.warning(f'CPU usage is {cpu_usage}%!')
            else:
                logger_health.critical(f'CPU usage is {cpu_usage}%!')

        if config.log_disk:
            disk_usage = psutil.disk_usage(os.getcwd()).percent
            if 60 >= disk_usage < 75:
                logger_health.info(f'Disk storage usage is {disk_usage}%!')
            elif disk_usage < 90:
                logger_health.warning(f'Disk storage usage is {disk_usage}%!')
            else:
                logger_health.critical(f'Disk storage usage is {disk_usage}%!')

        time.sleep(config.log_intervall)


def kaggle_vs_vader(dataset: Dataset, vader: Vader) -> str:
    # Check whether directory for comparators exists
    comparator_folder_path = os.path.join(os.getcwd(), dataset.config.output_folder, dataset.config.comparator_folder)
    if not os.path.exists(comparator_folder_path):
        os.mkdir(comparator_folder_path)
        dataset.config.logger.info(f'Created folder \'{dataset.config.comparator_folder}\' for comparators.')
    else:
        dataset.config.logger.info(f'Folder for comparators exists at {dataset.config.comparator_folder}.')

    # Check whether directory for vader exists within comparators directory
    vader_folder_path = os.path.join(os.getcwd(),
                                     dataset.config.output_folder,
                                     dataset.config.comparator_folder,
                                     vader.config.base_output_folder)
    if not os.path.exists(vader_folder_path):
        os.mkdir(vader_folder_path)
        dataset.config.logger.info(f'Created folder \'{vader.config.base_output_folder}\' '
                                   f'for vader comparison within comparators directory.')
    else:
        dataset.config.logger.info(f'Folder for vader comparison exists at {vader.config.base_output_folder}.')

    output_to = os.path.join(vader_folder_path, f'{config.comparator_filename}.feather')

    results = []
    years = dataset.get_years_covered()
    for year in tqdm(years, total=len(years), desc='Compare Kaggle sentiment with VADER score', ncols=120):
        output_mismatches = os.path.join(vader_folder_path, f'{config.comparator_filename}_mismatches_{year}.csv')
        # Get already calculated data
        df_data = dataset.load_preprocessed(year) if vader.config.use_preprocessed_data else dataset.load_raw(year)
        df_vader = vader.load_vader_file(year)
        # merge data
        df = pd.merge(df_data, df_vader, on='id')
        # Drop unnecessary columns
        df.drop(columns=['pos', 'neu', 'neg'], inplace=True)
        # Extract mismatches between Kaggle and VADER sentiment
        df_mismatches = df[df['sentiment'] != df['compound']][['body', 'sentiment', 'compound']]
        df_mismatches.columns = ['Text', 'Kaggle sentiment', 'VADER sentiment']
        count_nan_kaggle = df['sentiment'].isna().sum()
        # Remove rows with NaN since they can't be used for calculation of acc, rec, pre, f1, ...
        df = df[df['compound'].notna()]
        count_nan_vader = df['compound'].isna().sum()
        df = df[df['sentiment'].notna()]
        # Define column names for binning options
        col = 'kaggle class'
        # Build classes from sentiment / compound values
        df[col] = pd.cut(df['sentiment'], bins=[-1, -0.05, 0.05, 1],
                         labels=[-1, 0, 1], include_lowest=True, right=True)
        df['vader class'] = pd.cut(df['compound'], bins=[-1, -0.05, 0.05, 1],
                                   labels=[-1, 0, 1], include_lowest=True, right=True)
        # Count absolute entries per class
        class_count = df['vader class'].value_counts().to_dict()
        value_count = class_count[-1] + class_count[0] + class_count[1]
        # Calculate accuracy
        acc = accuracy_score(df['vader class'], df[col])
        # Number of correctly and incorrectly classified samples
        correct_classified = int(accuracy_score(df['vader class'], df[col], normalize=False))
        incorrect_classified = value_count - correct_classified
        # Calculate micro recall, precision and  f1-score
        rec_micro = recall_score(df['vader class'], df[col], average='micro')
        pre_micro = precision_score(df['vader class'], df[col], average='micro')
        f1_micro = f1_score(df['vader class'], df[col], average='micro')
        # Calculate macro recall, precision and  f1-score
        rec_macro = recall_score(df['vader class'], df[col], average='macro')
        pre_macro = precision_score(df['vader class'], df[col], average='macro')
        f1_macro = f1_score(df['vader class'], df[col], average='macro')

        results.append([
            year,
            value_count,
            class_count[-1],
            class_count[0],
            class_count[1],
            class_count[-1] / float(value_count),
            class_count[0] / float(value_count),
            class_count[1] / float(value_count),
            count_nan_kaggle,
            count_nan_vader,
            correct_classified,
            incorrect_classified,
            acc,
            rec_micro, pre_micro, f1_micro,
            rec_macro, pre_macro, f1_macro
        ])

        df_mismatches.to_csv(output_mismatches, index=False, sep=';')

    df_columns = [
        'year',
        'frequency_abs_sum',
        'frequency_abs_neg',
        'frequency_abs_neu',
        'frequency_abs_pos',
        'frequency_rel_neg',
        'frequency_rel_neu',
        'frequency_rel_pos',
        'frequency_abs_nan_kaggle',
        'frequency_abs_nan_vader',
        'correct_classified',
        'incorrect_classified',
        'accuracy',
        'recall_micro',
        'precision_micro',
        'f1_micro',
        'recall_macro',
        'precision_macro',
        'f1_macro'
    ]

    df = pd.DataFrame(results, columns=df_columns)
    dataset.config.logger.info(f'Finished comparison between Kaggle sentiment and VADER score.')
    df.to_feather(output_to, compression='lz4')
    df.to_csv(output_to.replace('.feather', '.csv'), index=False, sep=';')
    dataset.config.logger.info(f'Results were written to {output_to}')


def subreddit_plot(cfg: Config, top_n_subreddits: int = 10, save_img: bool = True):
    df = pd.read_csv(cfg.dataset.filename, usecols=['created_utc', 'subreddit.name'])
    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    df['year'] = df['datetime'].dt.year

    # Count occurrences of each subreddit and get the top 10
    top_subreddits = df['subreddit.name'].value_counts().nlargest(top_n_subreddits).index

    # Filter the DataFrame to include only the top 10 subreddits
    filtered_df = df[df['subreddit.name'].isin(top_subreddits)]

    # Group by 'year' and 'subreddit.name' and count occurrences
    subreddit_counts = filtered_df.groupby(['year', 'subreddit.name']).size().reset_index(name='counts')

    # Pivot the table
    pivot_table = subreddit_counts.pivot(index='year', columns='subreddit.name', values='counts').fillna(0)
    # Convert counts to percentage of the total for each year
    pivot_percent = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

    # Plot a stacked bar chart with percentages
    ax = pivot_percent.plot(kind='bar', stacked=True, figsize=(15, 10))
    plt.title('Percentage of Comments per Subreddit by Year')
    plt.xlabel('Year')
    plt.ylabel('Percentage of Comments')
    plt.xticks(rotation=45)
    # Add Text Annotations with conditional filtering for non-zero values
    for bars in ax.containers:
        # Generate labels, but conditionally display them
        labels = ax.bar_label(bars, fmt='%.1f%%', label_type='center', fontsize=8)
        for label, value in zip(labels, bars.datavalues):
            if value == 0:
                label.set_visible(False)  # Hide label for zero values

        # Optional: Adjust label transparency via the alpha of the label's color
        # This loops through each label setting its alpha value.
        for label in labels:
            label.set_alpha(0.8)  # Set alpha here. Alpha values are between 0 and 1.

    plt.legend(title='Subreddit', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    if save_img:
        output_folder_path = os.path.join(cfg.plot_output_path, 'Frequency')
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_folder_path, f'subreddit_frequency_{top_n_subreddits}.png')
        plt.savefig(output_path)
    else:
        plt.show()

    plt.clf()


def subreddit_plot_update(cfg: Config, top_n_subreddits: int = 5, save_img: bool = True):
    df = pd.read_csv(cfg.dataset.filename, usecols=['created_utc', 'subreddit.name'])
    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    df['year'] = df['datetime'].dt.year

    # Group by year and subreddit, and count occurrences
    subreddit_yearly_counts = df.groupby(['year', 'subreddit.name']).size().unstack(fill_value=0)

    # Collect all subreddits that appear in the top 5 in any year
    top_subreddits = set()
    for year in subreddit_yearly_counts.index:
        top_5 = subreddit_yearly_counts.loc[year].nlargest(top_n_subreddits).index
        top_subreddits.update(top_5)

    # Create a list from the set of all subreddits and add 'Others'
    all_subreddits = sorted(list(top_subreddits)) + ['Others']

    # Process each year
    rows = []
    for year in sorted(subreddit_yearly_counts.index):
        year_data = subreddit_yearly_counts.loc[year]
        top_5 = year_data.nlargest(top_n_subreddits)
        top_subreddits_this_year = top_5.index.tolist()
        # Initialize year data with zeros
        year_normalized = pd.Series(0, index=all_subreddits)
        # Populate the data for existing subreddits
        year_normalized[top_subreddits_this_year] = top_5[top_subreddits_this_year]
        # Calculate 'Others' as the sum of all non-top-5 subreddits
        others = year_data.drop(top_subreddits_this_year, errors='ignore').sum()
        year_normalized['Others'] = others
        # Normalize to 100%
        year_normalized /= year_normalized.sum()
        year_normalized *= 100
        # Append the normalized data for this year to the list
        rows.append(year_normalized)

    # Create the full DataFrame from the list of Series
    normalized_data = pd.DataFrame(rows, index=sorted(subreddit_yearly_counts.index))

    # Plotting
    ax = normalized_data.plot(kind='bar', stacked=True, colormap='nipy_spectral', figsize=(18, 10))
    ax.bar_width = 0.8
    plt.ylabel('Percentage')
    plt.title(f'Top {top_n_subreddits} Frequent Subreddits and Others by Year')
    plt.legend(title='Subreddit', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.xticks(rotation=45)
    # Add Text Annotations with conditional filtering for non-zero values
    for bars in ax.containers:
        # Generate labels, but conditionally display them
        labels = ax.bar_label(bars, fmt='%.1f%%', label_type='center', color='white', fontsize=9)
        for label, value in zip(labels, bars.datavalues):
            if value == 0:
                label.set_visible(False)  # Hide label for zero values
        # Optional: Adjust label transparency via the alpha of the label's color
        # This loops through each label setting its alpha value.
        for label in labels:
            label.set_alpha(0.9)  # Set alpha here. Alpha values are between 0 and 1.

    if save_img:
        output_folder_path = os.path.join(cfg.plot_output_path, 'Frequency')
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_folder_path, f'subreddit_frequency_top{top_n_subreddits}.png')
        plt.savefig(output_path)

        cfg.logger.info(f'Saved images about subreddit distribution per year to {output_folder_path}.')
    else:
        plt.show()

    plt.clf()


def comment_dist_total(cfg: Config, save_img: bool = True):
    df = pd.read_csv(cfg.dataset.filename, usecols=['created_utc'])
    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    df['year_month'] = df['datetime'].dt.strftime('%b %Y')
    labels = df['year_month'].unique()

    plt.figure(figsize=(32, 16))
    plt.hist(df['year_month'], bins=len(labels), range=(0, len(labels)), alpha=0.75, edgecolor='black')
    plt.title(f'Comment Frequency Distribution')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(np.arange(.5, len(labels) + .5), labels=labels)
    plt.gca().invert_xaxis()

    if save_img:
        output_folder_path = os.path.join(cfg.plot_output_path, 'Frequency')
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_folder_path, f'comment_distribution_total.png')
        plt.savefig(output_path)
    else:
        plt.show()

    plt.clf()


def comment_dist_within_year(dataset: Dataset, cfg: Config, save_img: bool = True):
    output_folder_path = os.path.join(cfg.plot_output_path, 'Frequency')
    plt.figure(figsize=(13, 6))

    years = dataset.get_years_covered()
    for year in (pbar := tqdm(years, total=len(years), ncols=120)):
        pbar.set_description(f'Loading data and plotting comment distribution within year {year}.')
        # Get already calculated data
        df = dataset.load_preprocessed(year) if vader.config.use_preprocessed_data else dataset.load_raw(year)
        df['month'] = df['datetime'].dt.month
        plt.hist(df['month'], bins=12, range=(0.5, 12.5), alpha=0.75, edgecolor='black')
        plt.title(f'Distribution of Comment Frequency in {year}')
        plt.xlabel('Month')
        plt.ylabel('Frequency')
        plt.xticks(np.arange(1, 13),
                   ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        if save_img:
            Path(output_folder_path).mkdir(parents=True, exist_ok=True)
            output_path = os.path.join(output_folder_path, f'comment_distribution_{year}.png')
            plt.savefig(output_path)
        else:
            plt.show()

        plt.clf()

    dataset.config.logger.info(f'Saved images about comment distribution to {output_folder_path}.')


def sentiment_per_year_plots(dataset: Dataset, cfg: Config):
    mean_sentiment_per_year(dataset, cfg, save_img=True)
    samples_per_year(dataset, cfg, save_img=True)


def mean_sentiment_per_year(dataset: Dataset, cfg: Config, save_img: bool = True):
    dataset.config.logger.info(
        f'Starting to calculate mean sentiment score per year from kaggle dataset.')

    results = []
    years = dataset.get_years_covered()
    for year in (pbar := tqdm(years, total=len(years), ncols=120)):
        pbar.set_description(f'Loading data of year {year} and calculating mean sentiment score.')
        # Get already calculated data
        df_data = dataset.load_preprocessed(year) if vader.config.use_preprocessed_data else dataset.load_raw(year)
        results.append(df_data['sentiment'].mean())

    df_result = pd.DataFrame({'year': list(map(int, years)),
                              'mean': results})

    plt.figure(figsize=(14, 8))

    sns.set_style("whitegrid", {'grid.linestyle': ':'})
    ax = sns.lineplot(data=df_result, x='year', y='mean')
    ax.set(title='Mean VADER Sentiment Score per Year', xlabel='Year', ylabel='Mean VADER Sentiment Score')

    plt.vlines(x=2012, ymin=df_result['mean'].min() * 1.1, ymax=df_result['mean'].max() * 1.1,
               linestyles='dotted', color='r', linewidth=.8)
    ax.annotate('United Nations\nConference\non Sustainable\nDevelopment (Rio+20)', xy=(2012, 0),
                xytext=(2013.3, -.025), fontsize='small', arrowprops=dict(facecolor='red', shrink=0.05))

    plt.vlines(x=2019, ymin=df_result['mean'].min() * 1.1, ymax=df_result['mean'].max() * 1.1,
               linestyles='dotted', color='r', linewidth=.8)
    ax.annotate('- Australian Bushfires\n- Amazon Rainforest Fires\n- Global Climate Protests and\n   Youth Movements*',
                xy=(2019, 0.04), xytext=(2013.5, .055), fontsize='small', arrowprops=dict(facecolor='red', shrink=0.05))
    plt.figtext(0.12, .01, '* Including "Fridays for Future"', fontsize=8)
    plt.tight_layout()

    if save_img:
        Path(cfg.plot_output_path).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(cfg.plot_output_path, 'mean_sentiment.png')
        plt.savefig(output_path)

        dataset.config.logger.info(f'Saved image to {output_path}.')
    else:
        plt.show()

    plt.clf()


def samples_per_year(dataset: Dataset, cfg: Config, save_img: bool = True):
    vader_folder_path = os.path.join(os.getcwd(),
                                     dataset.config.output_folder,
                                     dataset.config.comparator_folder,
                                     cfg.methodologies['vader'].base_output_folder)

    output_vader_comparison = os.path.join(vader_folder_path, f'{cfg.comparator_filename}.feather')

    assert os.path.exists(output_vader_comparison), f'Necessary file {output_vader_comparison} does not exist.'

    dataset.config.logger.info(
        f'Starting to plot sample frequency.')
    # Load data from vader comparison
    df = pd.read_feather(output_vader_comparison)
    df['total'] = df['frequency_abs_sum'] + df['frequency_abs_nan_kaggle']

    plt.figure(figsize=(16, 8))
    ax = sns.barplot(df, y="year", x="total", orient="y")
    ax.bar_label(ax.containers[0], fontsize=12)
    ax.set_title('Frequency of Reddit comments per year')
    ax.set_xlabel('Absolute Frequency')
    ax.set_ylabel('Year')
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    plt.tight_layout()

    if save_img:
        Path(cfg.plot_output_path).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(cfg.plot_output_path, 'samples_all.png')
        plt.savefig(output_path)

        dataset.config.logger.info(f'Saved image to {output_path}.')
    else:
        plt.show()

    plt.clf()

    data_bars_split = []
    for _, row in df.iterrows():
        data_bars_split.append([row['year'], 'Negative', row['frequency_abs_neg']])
        data_bars_split.append([row['year'], 'Neutral', row['frequency_abs_neu']])
        data_bars_split.append([row['year'], 'Positive', row['frequency_abs_pos']])

    df_bar_split = pd.DataFrame(data_bars_split, columns=['year', 'Sentiment Class', 'frequency'])

    plt.figure(figsize=(16, 11))

    ax = sns.barplot(df_bar_split,
                     y="year", x="frequency", hue="Sentiment Class", orient="y",
                     palette={'Negative': 'tomato', 'Neutral': 'gold', 'Positive': 'lightgreen'})
    ax.set_title('Frequency of Reddit comments per year by Sentiment Class', fontsize=18)
    # Setting the x-axis and y-axis labels with fontsize 16
    ax.set_xlabel('Absolute Frequency', fontsize=16)
    ax.set_ylabel('Year', fontsize=16)

    ax.bar_label(ax.containers[0], fontsize=12)
    ax.bar_label(ax.containers[1], fontsize=12)
    ax.bar_label(ax.containers[2], fontsize=12)
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    plt.tight_layout()

    if save_img:
        output_path = os.path.join(cfg.plot_output_path, 'samples_all_by_class.png')
        plt.savefig(output_path)

        dataset.config.logger.info(f'Saved image to {output_path}.')
    else:
        plt.show()

    plt.clf()


def interactive_average_sentiment_per_subreddit(cfg: Config, top_n_subreddits: int = 10):
    df = pd.read_csv(cfg.dataset.filename, usecols=['created_utc', 'subreddit.name', 'sentiment'])
    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')

    # Get the top 10 most frequent subreddits
    top_subreddits = df['subreddit.name'].value_counts().nlargest(top_n_subreddits).index.tolist()

    # Filter DataFrame to include only the top 10 subreddits
    filtered_df = df[df['subreddit.name'].isin(top_subreddits)]

    # Convert 'datetime' to the appropriate period
    filtered_df['month'] = filtered_df['datetime'].dt.to_period('M')

    # Group by month and subreddit and calculate average score
    monthly_sentiment = filtered_df.groupby(['month', 'subreddit.name'])['sentiment'].mean().reset_index()
    monthly_sentiment['month'] = monthly_sentiment['month'].dt.to_timestamp()  # Convert Period to Timestamp for Plotly

    # Create an interactive line plot with Plotly
    fig = px.line(monthly_sentiment, x='month', y='sentiment', color='subreddit.name',
                  title='Average Sentiment Score Over Time by Subreddit',
                  labels={'sentiment': 'Average Sentiment Score', 'month': 'Month'})

    fig.update_xaxes(rangeslider_visible=True)
    fig.show()


def hashtags_frequency_per_year(cfg: Config, top_n_hashtags: int = 10, save_img: bool = True):
    df = pd.read_csv(cfg.dataset.filename, usecols=['created_utc', 'body'])
    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    # Extract year from datetime
    df['year'] = df['datetime'].dt.year

    # Extract hashtags and convert to lowercase for case insensitivity
    df['hashtags'] = df['body'].apply(lambda x: re.findall(r"#([A-Za-z0-9_]{3,})", x.lower()))
    df = df.explode('hashtags')

    # Remove empty strings and NaN values from hashtags
    df = df[df['hashtags'].notna() & (df['hashtags'] != '')]
    df = df[~df['hashtags'].str.fullmatch(r'[0-9a-fA-F]+')]  # Removing hexadecimal patterns

    # Remove hashtags that still contain special characters (other than underscore)
    df = df[~df['hashtags'].str.contains(r"[^\w]")]  # \w matches letters, digits, and underscores
    # Filter out hashtags that are purely numeric or specific unwanted strings
    df = df[~df['hashtags'].str.fullmatch(r'\d+')]  # Removes purely numeric hashtags
    # Add hashtags that should be removed manully
    df = df[~df['hashtags'].isin(
        ['x200b', 'x0026', 'x200c', 'x27', 'cite_note', 'sthash', 'the', 'ref', 'amp_tf', 'wiki_'])]

    # Group by year and hashtag, count occurrences
    hashtag_counts = df.groupby(['year', 'hashtags']).size().reset_index(name='counts')

    # Calculate total entries per year
    total_entries_per_year = df.groupby('year')['hashtags'].size()

    # Normalize hashtag counts by the number of entries per year
    hashtag_counts = hashtag_counts.merge(total_entries_per_year, on='year', suffixes=('', '_total'))
    hashtag_counts['normalized_count'] = hashtag_counts['counts'] / hashtag_counts['hashtags_total']

    # Get the most frequent hashtags per year, sorted by normalized counts
    most_frequent_hashtags = hashtag_counts.sort_values(by=['year', 'normalized_count'], ascending=[True, False])
    most_frequent_hashtags = most_frequent_hashtags.groupby('year').head(top_n_hashtags)  # Top 10 hashtags per year

    # Determine common x-axis limits
    max_normalized_count = most_frequent_hashtags['normalized_count'].max()

    # Determine the number of unique years to create enough subplots
    years = most_frequent_hashtags['year'].unique()
    n_cols = 5
    n_rows = (len(years) + n_cols - 1) // n_cols  # Calculate required rows

    # Create the figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 5 * n_rows), squeeze=False)
    axes = axes.flatten()  # Flatten to ease the iteration

    # Track the last index for placing the x-axis title
    last_idx = len(axes) - 1

    for i, ax in enumerate(axes):
        if i < len(years):
            year = years[i]
            data = most_frequent_hashtags[most_frequent_hashtags['year'] == year]
            sns.barplot(x='normalized_count', y='hashtags', data=data, ax=ax, palette='viridis')
            ax.set_title(f'Year {year}')
            ax.set_xlim(0, max_normalized_count)  # Set uniform x-axis limits

            # Set y-axis title only for the first column
            if i % n_cols == 0:
                ax.set_ylabel('Hashtags')
            else:
                ax.set_ylabel('')  # Remove y-axis label for other plots

            # Set x-axis title only for the last row
            if i >= last_idx - (last_idx % n_cols):
                ax.set_xlabel('Normalized Frequency')
            else:
                ax.set_xlabel('')  # Remove x-axis label for other plots

            # Set a smaller font size for y-axis labels
            ax.tick_params(axis='y', labelsize=8)

        else:
            ax.axis('off')  # Hide unused subplots

    if save_img:
        output_path = os.path.join(cfg.plot_output_path, 'Frequency', 'hashtag_frequency_per_year.png')
        plt.savefig(output_path)

        cfg.logger.info(f'Saved image to {output_path}.')
    else:
        plt.show()

    plt.clf()


def wordcloud_per_class(cfg: Config, save_img: bool = True):
    # Basic text preprocessing
    def preprocess_text(text):
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        # Convert to lowercase
        text = text.lower()
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])

        return text

    df = pd.read_csv(cfg.dataset.filename, usecols=['created_utc', 'body', 'sentiment'])
    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    # Extract year from datetime
    df['year'] = df['datetime'].dt.year
    df['class'] = df['sentiment'].apply(lambda x: 'Negative' if x < -0.05 else 'Positive' if x > 0.05 else 'Neutral')

    # df = df.sample(n=500000, random_state=1)  # Adjust sample size as needed

    df['processed_body'] = df['body'].apply(preprocess_text)

    # Prepare the figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 25))  # 1 row, 3 columns

    for i, sentiment in enumerate(['Positive', 'Neutral', 'Negative']):
        cfg.logger.info(f'Starting wordcloud calculation for sentiment {sentiment}.')

        ax = axes[i]
        subset = df[df['class'] == sentiment]
        text = ' '.join(subset['processed_body'])

        # Create the word cloud using default colors
        wordcloud = WordCloud(width=800, height=400, background_color='white', prefer_horizontal=0.6).generate(text)

        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f'{sentiment} Sentiment')
        ax.axis('off')  # Hide the axes

    if save_img:
        output_path = os.path.join(cfg.plot_output_path, 'wordcloud_per_class.png')
        plt.savefig(output_path)

        cfg.logger.info(f'Saved image to {output_path}.')
    else:
        plt.show()

    plt.clf()


def samples_per_year_stacked(cfg: Config, save_img: bool = True):
    df = pd.read_csv(cfg.dataset.filename, usecols=['created_utc', 'sentiment'])
    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    df['class'] = df['sentiment'].apply(lambda x: 'Negative' if x < -0.05 else 'Positive' if x > 0.05 else 'Neutral')

    # Extract year from datetime
    df['year'] = df['datetime'].dt.year

    # Group by year and class, then count frequencies
    count_data = df.groupby(['year', 'class']).size().unstack(fill_value=0)

    mean_per_class = count_data.mean()
    std_per_class = count_data.std()

    # Convert count data to percentage
    percentage_data = count_data.div(count_data.sum(axis=1), axis=0) * 100

    # Plotting
    ax = percentage_data.plot(kind='bar', stacked=True, figsize=(12, 8), width=0.8,
                              color=['tomato', 'gold', 'lightgreen'])
    plt.title('Frequency of Sentiment Classes by Year')
    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.legend(title='Class', bbox_to_anchor=(1.02, 1), loc='upper left')

    # Add Text Annotations for the bars
    for bars in ax.containers:
        labels = ax.bar_label(bars, fmt='%.1f%%', label_type='center', fontsize=11)
        for label in labels:
            label.set_alpha(0.8)

    plt.tight_layout()

    if save_img:
        output_path = os.path.join(cfg.plot_output_path, 'samples_all_by_class_stacked.png')
        plt.savefig(output_path)
        cfg.logger.info(f'Saved image to {output_path}.')
    else:
        plt.show()

    plt.clf()


def kaggle_vs_NGram(df: pd.DataFrame, cfg: Config, save_img: bool = True):
    ngram = NGramLoader()
    years = list(df.index)

    result = {}
    for column in df.columns:
        if column != 'total_frequency':
            result[column] = ngram.load(column, year_start=years[0], year_end=years[-1])

    df_ngram = pd.DataFrame.from_records(result)

    df[df.columns] = df[df.columns].div(df['total_frequency'], axis=0)
    df = df.drop(columns=['total_frequency'])

    for column in df.columns:
        plt.plot(df[column], label='Kaggle')
        plt.plot(df_ngram[column], label='NGram')
        plt.title(F'Frequency Kaggle dataset vs. NGram: "{column}"')
        plt.xlabel('Year')
        plt.ylabel('Logarithmic rel. Frequency')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)

        if save_img:
            Path(cfg.plot_output_path).mkdir(parents=True, exist_ok=True)
            output_path = os.path.join(cfg.plot_output_path, f'frequency_kaggle_vs_NGram_{column}.png')
            plt.savefig(output_path)

            config.logger.info(f'Saved image to {output_path}.')
        else:
            plt.show()

        plt.clf()


def topic_over_time_line(df: pd.DataFrame, cfg: Config, ngrams_to_plot: list[str] = [],
                         is_freq: bool = False, save_img: bool = True):
    ngram_len = len(list(df.columns)[0].split())
    match ngram_len:
        case 1:
            ngram = 'Unigram'
        case 2:
            ngram = 'Bigram'
        case 3:
            ngram = 'Trigram'
        case 4:
            ngram = 'Quadgram'
        case _:
            ngram = 'To High'

    f, ax = plt.subplots(figsize=(10 + 2 * ngram_len, 6))

    columns = ngrams_to_plot if len(ngrams_to_plot) > 0 else list(df.columns)
    for column in columns:
        if column in df.columns:
            ax.plot(df[column], label=column.title())

    if is_freq:
        plt.title(f'Frequency of {len(columns)} Sampled {ngram}s')
        plt.ylabel('Frequency [%]')
    else:
        plt.hlines(y=0.05, xmin=2010, xmax=2022, linestyles='dashed', color='r', linewidth=3,
                   label='Limits Neutral\nSentiment')
        plt.hlines(y=-0.05, xmin=2010, xmax=2022, linestyles='dashed', color='r', linewidth=3)
        plt.title(f'VADER Sentiment Score of {len(columns)} Sampled {ngram}s')
        plt.ylabel('VADER Score')

    plt.xlabel('Year')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if save_img:
        ngram_len = len(list(df.columns)[0].split())
        match ngram_len:
            case 1:
                ngram = 'Unigram'
            case 2:
                ngram = 'Bigram'
            case 3:
                ngram = 'Trigram'
            case 4:
                ngram = 'Quadgram'
            case _:
                ngram = 'To High'

        output_folder = os.path.join(cfg.plot_output_path, config.word_counter_type.capitalize(), ngram)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_folder, f'{config.word_counter_type}_per_topic_{len(columns)}{ngram}.png')
        plt.savefig(output_path)

        config.logger.info(f'Saved image to {output_path}.')
    else:
        plt.show()

    plt.clf()


def topic_over_time_bar(df: pd.DataFrame, cfg: Config, save_img: bool = True):
    ngram_len = len(list(df.columns)[0].split())
    match ngram_len:
        case 1:
            ngram = 'Unigram'
        case 2:
            ngram = 'Bigram'
        case 3:
            ngram = 'Trigram'
        case 4:
            ngram = 'Quadgram'
        case _:
            ngram = 'To High'

    output_folder_path = os.path.join(cfg.plot_output_path, config.word_counter_type.capitalize(), ngram)

    for column in df.columns:
        ax = sns.barplot(df, x=df.index, y=column)
        # ax.bar_label(ax.containers[0], fontsize=10)
        ax.set(title=f'Frequency of "{column.title()}" per Year', ylabel='Frequency [%]', xlabel='Year')

        if save_img:
            Path(output_folder_path).mkdir(parents=True, exist_ok=True)
            output_path = os.path.join(output_folder_path, f'frequency_bar_{column}.png')
            plt.savefig(output_path)

            config.logger.info(f'Saved image to {output_path}.')
        else:
            plt.show()

        plt.clf()


def normalize_entity(entity: str):
    # Mapping of variations to a standard form
    normalization_map = {
        'us': 'USA',
        'usa': 'USA',
        'u.s.': 'USA',
        'united states': 'USA',
        'the united states': 'USA',
        'america': 'America',
        'americas': 'America',
        'latin america': 'South America',
        'north america': 'North America',
        'asia': 'Asia',
        'prc': 'China',
        'nra': 'China',
        'gore': 'Al Gore',
        'al.': 'Al Gore',
        'al': 'Al Gore',
        'al gore\'s': 'Al Gore',
        'trump': 'Donald Trump',
        'donald trump': 'Donald Trump',
        'trumps': 'Donald Trump',
        'donald': 'Donald Trump',
        'j.': 'Donald Trump',
        'paul': 'Ron Paul',
        'ron paul': 'Ron Paul',
        'ron': 'Ron Paul',
        'barton': 'Jake Barton',
        'fox news': 'Fox News',
        'fox': 'Fox News',
        'nasa': 'NASA',
        'congress': 'US Congress',
        'perry': 'Rick Perry',
        'ipcc': 'IPCC',
        'the intergovernmental panel on climate change': 'IPCC',
        'intergovernmental panel on climate change': 'IPCC',
        'epa': 'Environmental Protection Agency',
        'bush': 'George W. Bush',
        'george bush': 'George W. Bush',
        'gw': 'George W. Bush',
        'romney': 'Mitt Romney',
        'aoc': 'Affiliate Organizations Council',
        'sagan': 'Scott Sagan',
        'mitt romney': 'Mitt Romney',
        'spencer': 'Richard Spencer',
        'richard spencer': 'Richard Spencer',
        'hanson': 'Victor Davis Hanson',
        'saddam': 'Saddam Hussein',
        'oprah': 'Oprah Winfrey',
        'teslas': 'Tesla',
        'tesla': 'Tesla',
        'google': 'Google',
        'alphabet': 'Google',
        'covid': 'COVID-19',
        'covid19': 'COVID-19',
        'covid-19': 'COVID-19',
        'romm': 'Joseph Romm',
        'curry': 'Judith Curry',
        'santer': 'Benjamin Santer',
        'benjamin santer': 'Benjamin Santer',
        'nancy': 'Nancy Pelosi',
        'barack': 'Barack Obama',
        'obama': 'Barack Obama',
        'barack obama': 'Barack Obama',
        'mugabe': 'Robert Mugabe',
        'obama': 'Barack Obama',
        'merkel': 'Angela Merkel',
        'joe': 'Joe Biden',
        'apple': 'Apple',
        'apple inc': 'Apple',
        'apple inc.': 'Apple',
        'microsoft': 'Microsoft',
        'microsoft corporation': 'Microsoft',
        'new york': 'New York',
        'biden': 'Joe Biden',
        'bezos': 'Jeff Bezos',
        'bernie': 'Bernie Sanders',
        'sanders': 'Bernie Sanders',
        'hilary': 'Hillary Clinton',
        'hillary': 'Hillary Clinton',
        'stein': 'Jill Stein',
        'scalia': 'Antonin Scalia',
        'lindzen': 'Richard Lindzen',
        'nixon': 'Richard Nixon',
        'cheney': 'Dick Cheney',
        'lisiecki': 'Lorraine Lisiecki',
        'johnson': 'Boris Johnson',
        'akasofu': 'Syun-Ichi Akasofu',
        'burch': 'Sarah Burch',
        'collins': 'Murray Collins',
        'braswell': 'Danny Braswell',
        'allan': 'Myles Allen',
        'zágoni': 'Miklós Zágoni',
        'zagoni': 'Miklós Zágoni',
        'lovelock': 'James Lovelock',
        'clinton': 'Hillary Clinton',
        'clintons': 'Hillary Clinton',
        'rudd': 'Kevin Rudd',
        'kevin rudd': 'Kevin Rudd',
        'scheer': 'Andrew Scheer',
        'pete': 'Pete Buttigieg',
        'buttigieg': 'Pete Buttigieg',
        'thunberg': 'Greta Thunberg',
        'xi': 'Xi Jinping',
        'jinping': 'Xi Jinping',
        'greta': 'Greta Thunberg',
        'yang': 'Andrew Yang',
        'trudeau': 'Justin Trudeau',
        'warren': 'Warren Buffett',
        'cain': 'John McCain',
        'mccain': 'John McCain',
        'harper': 'Stephen Harper',
        'briner': 'Jason Briner',
        'abbott': 'Greg Abbott',
        'liang': 'Xinfeng Liang',
        'assad': 'Baschar al-Assad',
        'scafetta': 'Nicola Scafetta',
        'einstein': 'Albert Einstein',
        'briffa': 'Keith Briffa',
        'marc hiessen': 'Marc Thiessen',
        'thiessen': 'Marc Thiessen',
        'jon stewart': 'Jon Stewart',
        'david thorne': 'David Thorne',
        'j. geophys': 'Journal of Geophysics',
        'geophys': 'Journal of Geophysics',
        'geophysics': 'Journal of Geophysics',
        'j. clim': 'Journal of Climate',
        'lett': 'Journal of Chemistry Letters',
        'ghosh': 'Amitav Ghosh',
        'evans': 'Tom Evans',
        'dyson': 'Freeman John Dyson',
        'monbiot': 'George Monbiot',
        'wang': 'Wang Tao',
        'abbot': 'Greg Abbott',
        'hansen': 'James Hansen',
        'jones': 'Phil Jones',
        'gillett': 'Nathan Gillett',
        'hegerl': 'Gabriele Hegerl',
        'gillard': 'Julia Gillard',
        'osama': 'Osama bin Laden',
        'blair': 'Tony Blair',
        'tony': 'Tony Blair',
        'britain': 'United Kingdom',
        'uk': 'United Kingdom',
        'dc': 'Washington DC',
        'Washington': 'Washington DC',
        'cruz': 'Ted Cruz',
        'ted': 'Ted Cruz',
        'mann': 'Michael Mann',
        'feser': 'Edward Feser',
        'ryan': 'Paul Ryan',
        'jac': 'Jac Smit',
        'jfk': 'John F. Kennedy',
        'kennedy': 'John F. Kennedy',
        'murdoch': 'Rupert Murdoch',
        'santorum': 'Rick Santorum',
        'reagan': 'Ronald Reagan',
        'friedman': 'Milton Friedman',
        'cohen': 'Steven Cohen',
        'agw': 'Anthropogenic Global Warming',
        'gmo': 'Genetically Modified Food',
        'cap': 'Common Agricultural Policy',
        'mark': 'Mark Jacobson',
        'koch': 'Charles Koch',
        'putin': 'Wladimir Putin',
        'hitler': 'Adolf Hitler',
        'adolf': 'Adolf Hitler',
        'adolf hitler': 'Adolf Hitler',
        'bill': 'Bill Clinton',
        'newman': 'Rebecca Newman',
        'jeremy': 'Jeremy Clarkson',
        'gandhi': 'Mahatma Gandhi',
        'nye': 'Bill Nye',
        'cook': 'Charles Cook',
        'elon': 'Elon Musk',
        'musk': 'Elon Musk',
        'un': 'United Nations',
        'ccp': 'Climate Change Programme',
        'cnn': 'CNN',
        'bbc': 'BBC',
        'ccs': 'Carbon Capture and Storage',
        'bp': 'British Petroleum',
        'bric': 'BRIC',  # brazil, russia, india, china
        'ghg': 'Greenhouse Gases',
        'cru': 'Climatic Research Unit',
        'sun': 'The Sun',
        'eu': 'European Union',
        'aps': 'Announced Pledges Scenario',
        'jesus': 'Jesus Christ',
        'exxon': 'ExxonMobile',
        'the paris agreement': 'Paris Agreement',
        'americans': 'American',
        'africans': 'African',
        'republican': 'Republican',
        'republicans': 'Republican',
        'gop': 'Republican',
        'the republican party': 'Republican',
        'democrats': 'Democrat',
        'dem': 'Democrat',
        'edward snowden': 'Edward Ssnowden',
        'nsa': 'NSA',
        'dems': 'Democrat',
        'christians': 'Christian',
        'christianity': 'Christian',
        'australians': 'Australian',
        'canadians': 'Canadian',
        'muslims': 'Muslim',
        'europeans': 'European',
        'jews': 'Jew',
        'conservatives': 'Conservative',
        'british conservative': 'Conservative',
        'germans': 'German',
        'lamb': 'Hubert Lamb',
        'quebec': 'Québec',
        'poles': 'Pole',
        'indians': 'Indian',
        'mao': 'Mao Zedong',
        'sahel': 'Sahel Region',
        'knut': 'Polar Bear Knut',
        'francis': 'Francis Fukuyama',
        'pruitt': 'Scott Pruitt',
        'fukuyama': 'Francis Fukuyama',
        'carson': 'Ben Carson',
        'kerry': 'John Kerry',
        'morris': 'Dick Morris',
        'inhofe': 'Jim Inhofe',
        'stalin': 'Joseph Stalin',
        'Joaquin': 'Joaquin Castro',
        'hardison': 'Preston Hardison',
        'reich': 'Peter Reich',
        'foster': 'Bill Foster',
        'wigley': 'Tom Wigley',
        'howard': 'Howard Dean',
        'dean': 'Howard Dean',
        'rubio': 'Marco Rubio',
        'howard dean': 'Howard Dean',
        'sato': 'Sato',
        'tpp': 'Trans-Pacific Partnership',
        'clif': 'Clif',
        'reddit': 'Reddit',
        'acc': 'American Control Conference',
        'oecd': 'OECD',
        'fed': 'Federal Reserve',
        'iaea': 'International Atomic Energy Agency',
        'fsb': 'FSB',
        'cia': 'CIA',
        'house': 'US House of Representatives',
        'labor': 'US Department of Labor',
        'doe': 'Department of Energy',
        'nrc': 'Nuclear Regulatory Research',
        'lftr': 'Liquid Fluoride Thorium Reactor',
        'gm': 'General Motors',
        'ev': 'EV',
        'breitbart': 'Breitbart',
        'macron': 'Emmanuel Macron',
        'beck': 'Paul Beck',
        'eisenhower': 'Dwight D. Eisenhower',
        'otto': 'Rebecca Otto',
        'lewis': 'John Lewis',
        'twitter': 'Twitter',
        'tillerson': 'Rex Tillerson',
        'malthus': 'Thomas Robert Malthus',
        'aca': 'Affordable Care Act',
        'princeton': 'Princeton University',
        'mercer': 'Mercer',
        'nsw': 'US Naval Special Warfare Command',
        'dnc': 'Democratic National Committee',
        'gnc': 'Green New Deal',
        'senate': 'US Senate',
        'bernie bot': 'Bernie Bot',
        'noaa': 'National Oceanic and\nAtmospheric Administration',
        'china': 'China',
        'monckton': 'Christopher Monckton',
        'milankovitch': 'Milutin Milankovitch',
        'american': 'American',
        'uan': 'University of Alabama in Huntsville',
        'mclntyre': 'Steve Mclntyre',
        'united nations': 'United Nations',
        'democrat': 'Democrat',
        'mit': 'Massachusetts Institute of Technology',
        'zhou': 'Liming Zhou',
        'lovins': 'Amory Lovins',
        'malthus': 'Thomas Robert Malthus',
        'marshall': 'George Marshall',
        'zimmermann': 'Arthur Zimmermann',
        'roberts': 'Julia Roberts',
        'liu': 'Wei-Min Hao Liu',
        'amurikan': 'American',
        'cameron': 'David Cameron',
        'stewart': 'Jerome Namias Stewart',
        'broecker': 'Wallace Smith Broecker',
        'christy': 'John R. Christy',
        'schlesinger': 'Daniel Schlesinger',
        'ahmadinejad': 'Mahmoud Ahmadinejad',
        'walton': 'Walton Family Foundation',
        'lenin': 'Wladimir Lenin',
        'philipona': 'Philippe Philipona',
        'lockwood': 'Mike Lockwood',
        'Keeling': 'Charles David Keeling',
        'christian': 'Christian',
        'chinese': 'Chinese',
        'arrhenius': 'Svante Arrhenius',
        'dessler': 'Andrew Dessler',
        'bernanke': 'Ben Bernanke',
        'carter': 'Jimmy Carter',
        'greens': 'Green',
        'middleton': 'Kate Middleton',
        'choi': ' Yoon-Jung Choi',
        'sestak': 'Joe Sestak',
        'andrew': 'Andrew Revkin',
        'tepco': 'Tokyo Electric Power Company',
        'pell': 'George Pell',
        'darwin': 'Charles Darwin',
        'gavin': 'Gavin Schmidt',
        'loehle': 'Craig Loehle',
        'harries': 'John Harries',
        'german': 'German',
        'holdren': 'John Holdren',
        'ledbetter': 'Lilly Ledbetter',
        'gleick': 'Peter Gleick',
        'meehl': 'Gerald Meehl',
        'mongols': 'Mongol',
        'marx': 'Karl Marx',
        'mckibben': 'Bill McKibben',
        'storch': 'Hans von Storch',
        'fourier': 'Joseph Fourier',
        'hubbert': 'M. King Hubbert',
        'howarth': 'Robert Howarth',
        'miller': 'Gifford Miller',
        'adam': 'Adam Smith',
        'rahmstorf': 'Stefan Rahmstorf',
        'allen': 'Richard Alley',
        'russian': 'Russian',
        'herzberg': 'Gerhard Herzberg',
        'tyson': 'Neil deGrasse Tyson',
        'teller': 'Edward Teller',
        'pole': 'Pole',
        'tyndall': 'John Tyndall',
        'taylor': 'Professor Peter Taylor',
        'svante': 'Svante Arrhenius',
        'brown': 'Jerry Brown',
        'reiss': 'Mike Reiss',
        'jackson': 'Lisa P. Jackson',
        'anderegg': 'William R.L. Anderegg',
        'notley': 'Rachel Notley',
        'dudley': 'Robert Dudley',
        'chen': 'Fei Chen',
        'griggs': 'James Griggs',
        'feldman': 'Michael Feldman',
        'jill': 'Jill Stein',
        'doran': 'Peter Doran',
        'paris': 'Paris',
        'drumpf': 'Donald Trump',
        'flynn': 'Michael Flynn',
        'jacobson': 'Mark Z. Jacobson',
        'robinson': 'Mary Robinson',
        'tucker': 'Tucker Carlson',
        'peterson': 'Peter G. Peterson',
        'kim': 'Kim Jong-un',
        'pompeo': 'Mike Pompeo',
        'stevenson': 'Adlai Stevenson',
        'charney': 'Jule Charney',
        'harris': 'Kamala Harris',
        'leyen': 'Ursula von der Leyen',
        'zichal': 'Heather Zichal',
        'african': 'African',
        'blm': 'African',
        'mlk': 'Martin Luther King',
        'bidens': 'Joe Biden',
        'nelson': 'Nelson Mandela',
        'james': 'James Madison',
        'stennis': 'John C. Stennis',
        'mbna': 'MBNA America Bank',
        'mcconnell': 'Mitch McConnell',
        'mueller': 'Robert Mueller',
        'mnuchin': 'Steven Mnuchin',
        'javadekar': 'Prakash Javadekar',
        'scott': 'Scott Morrison',
        'gladys': 'Gladys Berejiklian',
        'wwii': 'World War',
        'world war': 'World War',
        'world war 2': 'World War',
        'ww ii': 'World War',
        'the civil war': 'Civil War',
        'civil war': 'Civil War',
        'vietnam war': 'Vietnam War',
        'the vietnam war': 'Vietnam War',
        'the cold war': 'Cold War',
        'cold war': 'Cold War',
        'the iraq war': 'Iraq War',
        'iraq war': 'Iraq War',
        'the iraq war and operation iraqi freedom': 'Iraq War',
        'the great depression': 'Great Depression',
        'great depression': 'Great Depression',
        'the great recession': 'Great Recession',
        'great recession': 'Great Recession',
        'the industrial revolution': 'Industrial Revolution',
        'katrina': 'Hurricane Katrina',
        'sandy': 'Hurricane Sandy',
        'the war on drugs': 'War on Drugs',
        'the drug war': 'War on Drugs',
        'drug war': 'War on Drugs',
        'the arab spring': 'Arab Spring',
        'the dust bowl': 'Dust Bowl',
        'the great barrier reef': 'Great Barrier Reef',
        'williams': 'Serena Williams'
    }

    return normalization_map.get(entity.lower(), entity)


# Correction function for entity labels
def correct_entity_label(entity, label):
    correction_map = {
        'fukushima': 'GPE',
        'solyndra': 'ORG',
        'clif': 'ORG',
        'journal of geophysics': 'ORG',
        'journal of chemistry letters': 'ORG',
        'common agricultural policy': 'ORG',
        'anthropogenic global warming': 'ORG',
        'hurricane sandy': 'EVENT',
        'sato': 'ORG',
        'apple': 'ORG',
        'microsoft': 'ORG',
        'greenpeace': 'ORG',
        'barack obama': 'PERSON',
        'bernie sanders': 'PERSON',
        'donald trump': 'PERSON',
        'breitbart': 'ORG',
        'twitter': 'ORG',
        'mercer': 'ORG',
        'american control conference': 'EVENT',
        'republican': 'NORP',
        'dem': 'NORP',
        'reddit': 'ORG',
        'copenhagen': 'GPE',
        'australia': 'GPE',
        'canada': 'GPE',
        'india': 'GPE',
        'christopher monckton': 'PERSON',
        'milutin milankovitch': 'PERSON',
        'muslim': 'NORP',
        'catholic': 'NORP',
        'central intelligence agency': 'ORG',
        'vikings': 'NORP',
        'university of alabama in huntsville': 'ORG',
        'michael Mann': 'PERSON',
        'steve mclntyre': 'PERSON',
        'general motors': 'ORG',
        'intergovernmental panel on climate change': 'ORG',
        'democrat': 'NORP',
        'OECD': 'ORG',
        'massachusetts institute of technology': 'ORG',
        'liming zhou': 'PERSON',
        'amory lovins': 'PERSON',
        'wei-min hao liu': 'PERSON',
        'american': 'NORP',
        'cameron wake': 'PERSON',
        'chen zhu': 'PERSON',
        'massachusetts': 'GPE',
        'siberia': 'GPE',
        'jerome namias stewart': 'PERSON',
        'john r. christy': 'PERSON',
        'daniel schlesinger': 'PERSON',
        'mahmoud ahmadinejad': 'PERSON',
        'walton family foundation': 'ORG',
        'jerry brown': 'PERSON',
        'hubert au': 'PERSON',
        'wladimir lenin': 'PERSON',
        'sahel region': 'GPE',
        'philippe philipona': 'PERSON',
        'richard harries': 'PERSON',
        'cern': 'ORG',
        'mike lockwood': 'PERSON',
        'charles david keeling': 'PERSON',
        'wikipedia': 'ORG',
        'christian': 'NORP',
        'braganza': 'GPE',
        'svante arrhenius': 'PERSON',
        'andrew dessler': 'PERSON',
        'george w. bush': 'PERSON',
        'ben bernanke': 'PERSON',
        'jimmy carter': 'PERSON',
        'green': 'NORP',
        'krakatoa': 'GPE',
        'john tyndall': 'PERSON',
        'kate middleton': 'PERSON',
        'yoon-jung choi': 'PERSON',
        'eugene': 'GPE',
        'pickens': 'GPE',
        'joe sestak': 'PERSON',
        'andrew revkin': 'PERSON',
        'tokyo electric power company': 'ORG',
        'george pell': 'PERSON',
        'charles darwin': 'PERSON',
        'gavin schmidt': 'PERSON',
        'craig loehle': 'PERSON',
        'john harries': 'PERSON',
        'al gore': 'PERSON',
        'john holdren': 'PERSON',
        'vattenfall': 'ORG',
        'julia gillard': 'PERSON',
        'islam': 'NORP',
        'mongol': 'NORP',
        'cancun': 'GPE',
        'davos': 'GPE',
        'cuadrilla': 'ORG',
        'bernie Sanders': 'PERSON',
        'forbes': 'ORG',
        'kensington': 'GPE',
        'massachoosetts': 'GPE',
        'ursula von der leyen': 'PERSON',
        'heather zichal': 'PERSON',
        'latinos': 'NORP',
        'martin luther king': 'PERSON',
        'bloomberg': 'ORG',
        'mbna america bank': 'ORG',
        'grundfos': 'ORG',
        'europe': 'GPE',
        'america': 'GPE',
        'south america': 'GPE',
        'asia': 'GPE',
        'africa': 'GPE',
        'north america': 'GPE',
        'the sun': 'ORG',
        'danfoss': 'ORG',
        'covid-19': 'EVENT',
        'african': 'GPE',
        'great barrier reef': 'LOC'
    }

    return correction_map.get(entity.lower(), label)


def entity_exclusion_list():
    # Exclusion list
    return ['marijuana', 'evolution', 'coal', 'abuse', 'excerpt', 'supplementary', 'co2', 'brexit', 'c02', 'fossil',
            'bullshit', 'holocene', 'jesus christ', 'anthropogenic global warming', 'common agricultural policy',
            'environmental protection agency', 'op', 'h20', 'ice', 'ba', 'f', 'lftr', 'mwp', 'lr', 'helped', 'venus',
            'mandated', 'smog', 'signed', 'methane', 'anhang', 'nuclear', 'max', 'celsius', 'kelvin', 'fig',
            'extreme', 'l07608', 'greenhouse', 'north', 'idiot', 'meh', 'reddit', 'climate', 'ev', 'justice',
            'genetically modified food', 'greenhouse gases', 'green new deal', 'science', 'bernie bot', 'earth',
            'agrarian', 'h2o', 'mans', 'newtonian', 'obese', 'astroturf', 'albedo', 'huzzah', 'yucca', 'helluva',
            'climate', 'greenhouse gases', 'islam', 'climategate', 'announced pledges scenario', 'islam',
            'carbon capture and storage', 'nimby', 'ir', 'ch4', 'speech', 'liquid fluoride thorium reactor',
            'nuclear regulatory research', 'long', 'carbon capture and storage', 'tmi', 'anwr',
            'genetically modified food', 'liquid fluoride thorium reactor', 'treasury', 'geothermal', 'conroy', 'pro',
            'bullshit', 'marijuana', 'hmm', 'sigh', 'calculating', 'roberts', 'möller', 'sycamore', 'sabatier', 'mini',
            'slash', 'pascal', 'trains', 'monsignor', 'haddad', 'plato', 'freedomWorks', 'corbell', 'bob', 'religion',
            'lu', 'albedo', 'cagw', 'griggs', 'anomaly', 'carbon', 'mans', 'wean', 'exactly', 'no2', 'g8', 'pp',
            'ipad', 'uhh', 'sres', 'ets', 'amo', 'cfl', 'genetically modified food', 'medicare', 'pdf', 'huge', 'amo',
            'thorium', 'meteor', 'fun', 'ref', 'ets', 'Lia', 'inso', 'best', 'soooo', 'gasland', 'question', 'sulphur',
            'greed', 'stupid', 'dystopia', 'hummer', 'reduce', 'misericordiam', 'greed', 'hadley', 'strawman',
            'reptilian', 'peak', 'oceans', 'sandy', 'ecs', 'judicial', 'celsius', 'mw', 'btu', 'ddt', 'imho',
            'medicaid', 'reinstated', 'created', 'appointed', 'restored', 'avta', 'science', 'meh', 'surface',
            'meteorol', 'marijuana', 'ethanol', 'fahrenheit', 'banned', 'lgm', 'eemian', 'ppm', 'cagw', 'intro',
            'anomaly', 'willis', 'nbn', 'entirely', 'science', 'ng', 'lot', 'ev', 'cc', 'likely', 'undersea', 'hunt',
            'jack', 'discussed', 'awww', 'lu', 'christy', 'tmt', 'Ot5n9m4whaw', 'biodiesel', 'david', 'ken',
            'treasuer', 'undersea', 'devolve', 'kinda', 'speed', 'nat', 'leed', 'oxygen', 'humans', 'enso', 'hydro',
            'ai', 'chemical', 'ret', 'nhs', 'papers', 'tsunami', 'cosmos', 'level', 'iris', 'fahrenheit', 'zeitgeist',
            'nah', 'moon', 'hmm', 'ban', 'articles', 'tpp', 'state', 'rss', 'geoengineer', 'learn', 'jack', 'ocean',
            'carbon', 'genetically modified food', 'nafta', 'obamacare', 'bc', 'state', 'aca', 'grl', 'commerce',
            'seventh', 'sixth', 'aca', 'ttp', 'rrs', 'ref', 'veterans', 'fracking', 'infowars', 'agenda', 'energy',
            'education', 'ppm', 'ecs', 'tcr', 'commerce', 'education', 'hydroelectric', 'luckey', 'm2', 'harvey',
            'irma', 'econ', 'ppm', 'iphones', 'energy', 'science', 'bitcoin', 'goldblatt', 'eocene', 'wind', 'idk',
            'dryas', 'oceans', 'soviet', 'gnd', 'carbontax', 'labour', 'deccan', 'ubi', 'rcp', 'moon', 'medieval',
            'cinemas', 'qdefrwjfc9c', 'coachella', 'creepy', 'bans', 'swachh', 'justice', 'creep', 'ample',
            'rowe', 'ro', 'hr5005', 'fosta', 'glass', 'moon', 'Ångström', 'tamoxiphen', 'nature16494', 'ref',
            'subsidy', 'biofuels', 'lupus', 'tcja', 'congresswomen', 'km', 'firefighter', 'bandaid', 'kickatinalong',
            'posta', 'p.s.', 'el niño', 'r.', 'cache:6jqq7kPhOVg)', 'change ](http://www.metoffice.gov.uk', 'humans',
            'designs', 'anti-yucca', 'd.', 'municipal](http://en.wikipedia.org', 'global warming',
            'the dakota access pipeline', 'el nino', 'the day after', 'international year of the planet',
            'global climate change', 'removal](http://www.reddit.com', '^^original ^^reddit', 'affordable care act',
            'trans-pacific partnership', 'summary**](http://np.reddit.com', 'healthcare']


def valid_entity_labels():
    return ['PERSON', 'GPE', 'ORG', 'NORP', 'LOC', 'EVENT']


# Function to analyze entities in documents
def analyze_entities(docs, df, search_term):
    search_text, search_label = search_term

    sentiments = []
    for i, doc in enumerate(docs):
        ents = []
        labels = []
        for ent in doc.ents:
            normalized_text = normalize_entity(ent.text)
            corrected_label = correct_entity_label(normalized_text, ent.label_)

            ents.append(normalized_text)
            labels.append(corrected_label)

        if i == 125:
            a = 0
        if any(e == search_text and l == search_label for e, l in zip(ents, labels)):

            # if any(normalize_entity(ent.text) == search_text
            #      and correct_entity_label(normalize_entity(ent.text), ent.label_) == search_label
            #     for ent in doc.ents):
            sentiment = df.iloc[i]['sentiment']
            if pd.notna(sentiment):
                sentiments.append(sentiment)

    mean_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    doc_count = len(docs)
    frequency = len(sentiments)

    return frequency, 100 * frequency / doc_count, mean_sentiment


def entity_heatmap(dataset: Dataset, cfg: Config, entities_with_labels: [(str, str)],
                   save_img: bool = True):
    results = []
    for year in dataset.get_years_covered():
        docs_folder_path = os.path.join(cfg.plot_output_path, 'Frequency', 'Entity', year)
        Path(docs_folder_path).mkdir(parents=True, exist_ok=True)

        cfg.logger.info(f'Load dataset of year {year}.')
        df = dataset.load_preprocessed(year) if vader.config.use_preprocessed_data else dataset.load_raw(year)
        df.reset_index(inplace=True)

        batch_cnt = math.ceil(len(df) / cfg.docbin_batch_size)
        cfg.logger.info(
            f'[Search Entity: Heatmap] Starting entity frequency and sentiment calculation (with spaCy) for {year} with {len(df)} comments using {batch_cnt} batch(es).')

        docs = []
        if len([file for file in os.listdir(str(docs_folder_path)) if file.endswith('.bin')]) > 0:
            cfg.logger.info(f'Found spaCy .bin files in {docs_folder_path}. Start loading spaCy .bin files.')
            for idx in range(batch_cnt):
                doc_bin = DocBin().from_disk(os.path.join(docs_folder_path, f'spaCy_docs_{year}_{idx}.bin'))
                docs_loaded = list(doc_bin.get_docs(nlp.vocab))
                docs.extend(docs_loaded)
                cfg.logger.info(f'Loaded batch of {len(docs_loaded):06} ({idx + 1} / {batch_cnt}) docs.')
        else:
            cfg.logger.info(f'Didn\'t find spaCy .bin files in {docs_folder_path}. Start calculating spaCy bin files.')
            batches = [df['body'][i:i + cfg.docbin_batch_size].tolist() for i in
                       range(0, len(df), cfg.docbin_batch_size)]
            for idx, batch in enumerate(batches):
                docs_batch = [nlp(text) for text in batch]
                docs.extend(docs_batch)
                doc_bin = DocBin(docs=docs_batch)
                doc_bin.to_disk(os.path.join(docs_folder_path, f'spaCy_docs_{year}_{idx}.bin'))
                cfg.logger.info(
                    f'Saved batch of {len(batch):06} ({idx + 1} / {len(batches)}) docs to .bin file in {docs_folder_path}.')

        for entity_with_label in entities_with_labels:
            frequency_abs, frequency_rel, sentiment = analyze_entities(docs, df, entity_with_label)
            entity_text, entity_label = entity_with_label
            results.append((entity_text, int(year), sentiment, frequency_rel, frequency_abs))

    df_results = pd.DataFrame(results,
                              columns=['Entity', 'Year', 'Sentiment', 'Relative Frequency', 'Absolute Frequency'])

    # Extract the ordered list of entity labels for reindexing
    ordered_entities = [entity_text for entity_text, _ in entities_with_labels]

    # Pivot the data to create matrices for the heatmap
    heatmap_data_sentiment = df_results.pivot(index='Year', columns='Entity', values='Sentiment')
    heatmap_data_relative_frequency = df_results.pivot(index='Year', columns='Entity', values='Relative Frequency')
    heatmap_data_absolute_frequency = df_results.pivot(index='Year', columns='Entity', values='Absolute Frequency')

    # Reindex the columns to ensure they are sorted by the provided order of entities
    heatmap_data_sentiment = heatmap_data_sentiment.reindex(columns=ordered_entities)
    heatmap_data_relative_frequency = heatmap_data_relative_frequency.reindex(columns=ordered_entities)
    heatmap_data_absolute_frequency = heatmap_data_absolute_frequency.reindex(columns=ordered_entities)

    # Sort the index to ensure years are ordered correctly in descending order
    heatmap_data_sentiment = heatmap_data_sentiment.sort_index(ascending=False)
    heatmap_data_relative_frequency = heatmap_data_relative_frequency.sort_index(ascending=False)
    heatmap_data_absolute_frequency = heatmap_data_absolute_frequency.sort_index(ascending=False)

    # Combine relative and absolute frequency for annotations, formatting relative frequency to 5 decimal places
    annotations = heatmap_data_relative_frequency.map(
        lambda x: f"{x:.5f}%") + "\n(" + heatmap_data_absolute_frequency.astype(int).astype(str) + ")"

    # Create a combined heatmap with sentiment as color and combined frequencies as annotation
    plt.figure(figsize=(1 + 2.5 * len(entities_with_labels), 10))
    ax = sns.heatmap(heatmap_data_sentiment, annot=annotations, fmt="", cmap='coolwarm',
                     center=0, linewidths=.5, vmin=-1, vmax=1, annot_kws={"size": 10})
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.title('Sentiment and Frequency of Mentions Over Years')
    plt.yticks(rotation=0)  # Make y-axis ticks horizontal
    plt.tight_layout()

    if save_img:
        output_path = os.path.join(cfg.plot_output_path, 'Frequency', 'Entity',
                                   f'heatmap_sentiment_and_frequency_{len(entities_with_labels)}entities_{entity_label}.png')
        plt.savefig(output_path)
        config.logger.info(f'Saved image to {output_path}.')
    else:
        plt.show()

    plt.clf()


def entity_frequency_and_sentiment(dataset: Dataset, cfg: Config, search_terms: {str: str}, save_img: bool = True):
    results = []
    for year in dataset.get_years_covered():
        docs_folder_path = os.path.join(cfg.plot_output_path, 'Frequency', 'Entity', year)
        Path(docs_folder_path).mkdir(parents=True, exist_ok=True)

        cfg.logger.info(f'Load dataset of year {year}.')
        df = dataset.load_preprocessed(year) if vader.config.use_preprocessed_data else dataset.load_raw(year)
        df.reset_index(inplace=True)

        batch_cnt = math.ceil(len(df) / cfg.docbin_batch_size)
        cfg.logger.info(
            f'[Search Entity: Line Plot] Starting entity frequency and sentiment calculation (with spaCy) for {year} with {len(df)} comments using {batch_cnt} batch(es).')

        docs = []
        if len([file for file in os.listdir(str(docs_folder_path)) if file.endswith('.bin')]) > 0:
            cfg.logger.info(f'Found spaCy .bin files in {docs_folder_path}. Start loading spaCy .bin files.')
            for idx in range(batch_cnt):
                doc_bin = DocBin().from_disk(os.path.join(docs_folder_path, f'spaCy_docs_{year}_{idx}.bin'))
                docs_loaded = list(doc_bin.get_docs(nlp.vocab))
                docs.extend(docs_loaded)  # Combine all documents into a single list if needed
                cfg.logger.info(f'Loaded batch of {len(docs_loaded):06} ({idx + 1} / {batch_cnt}) docs.')
        else:
            cfg.logger.info(f'Didn\'t find spaCy .bin files in {docs_folder_path}. Start calculating spaCy bin files.')
            # Create batches of texts to process them separately
            batches = [df['body'][i:i + cfg.docbin_batch_size].tolist() for i in
                       range(0, len(df), cfg.docbin_batch_size)]
            # Create file for each batch
            for idx, batch in enumerate(batches):
                docs_batch = [nlp(clean_text(text)) for text in batch]
                docs.extend(docs_batch)
                doc_bin = DocBin(docs=docs_batch)
                doc_bin.to_disk(os.path.join(docs_folder_path, f'spaCy_docs_{year}_{idx}.bin'))
                cfg.logger.info(
                    f'Saved batch of {len(batch):06} ({idx + 1} / {len(batches)}) docs to .bin file in {docs_folder_path}.')

        for k, v in search_terms.items():
            cfg.logger.info(f'Starting to analyze entity \'{k}\' with label \'{v}\'.')
            # Analyze entities
            frequency_abs, frequency_rel, sentiment = analyze_entities(docs, df, (k, v))
            # Append results for the current year
            results.append((int(year), frequency_abs, frequency_rel, sentiment, k))

    df_results = pd.DataFrame(results, columns=['Year', 'Absolute Frequency', 'Relative Frequency',
                                                'Mean Sentiment', 'Search Term'])

    # Save the results as CSV file and do the plots
    output_folder_path = os.path.join(cfg.plot_output_path, 'Frequency', 'Entity')
    Path(output_folder_path).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_folder_path, 'entity_frequency.csv')

    df_results.to_csv(output_path, sep=';', index=False, header=not os.path.exists(output_path))

    for k, v in search_terms.items():
        df_results_search_term = df_results[df_results['Search Term'] == k]

        fig, ax1 = plt.subplots(figsize=(20, 10))

        # Plot "Mean Sentiment" on the primary y-axis with different line style and color
        ax1.plot(df_results_search_term['Year'], df_results_search_term['Mean Sentiment'], label='Mean Sentiment',
                 marker='o', linestyle='-', linewidth=2, color='blue', alpha=0.8, markeredgewidth=2,
                 markeredgecolor='black')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Sentiment', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.tick_params(axis='x', labelcolor='black')

        # Create secondary y-axis for "Relative Frequency" on the left side with different line style and color
        ax2 = ax1.twinx()
        ax2.plot(df_results_search_term['Year'], df_results_search_term['Relative Frequency'],
                 label='Frequency', marker='x', linestyle='--', linewidth=2, color='green', alpha=0.8,
                 markeredgewidth=2, markeredgecolor='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.spines['left'].set_position(('outward', 60))  # Offset the secondary y-axis to avoid overlap
        ax2.set_ylabel(f'Frequency [%]', color='black')

        # Combine legends from all axes and automatically determine the best location
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', frameon=True, framealpha=1, borderpad=1)

        # Add grid
        ax1.grid(True, linestyle='--', linewidth=0.5)
        ax2.grid(True, linestyle='--', linewidth=0.5)

        # Title without font changes
        plt.title(f'Entity Frequency and Sentiment for \"{k}\" with label \"{v}\"')

        # Remove unnecessary spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        # Highlight specific points if needed (example: annotate a peak and lowest point)
        max_sentiment_idx = df_results_search_term['Mean Sentiment'].idxmax()
        max_sentiment_year = df_results_search_term.loc[max_sentiment_idx, 'Year']
        max_sentiment_value = df_results_search_term.loc[max_sentiment_idx, 'Mean Sentiment']
        ax1.annotate(f'Peak: {max_sentiment_value:.2f}',
                     xy=(max_sentiment_year, max_sentiment_value),
                     xytext=(max_sentiment_year, max_sentiment_value - max(.07, .2 * max_sentiment_value)),
                     textcoords='data',
                     ha='center',
                     arrowprops=dict(facecolor='gray', edgecolor='gray', arrowstyle='->', linewidth=2, shrinkA=5,
                                     shrinkB=5))

        min_sentiment_idx = df_results_search_term['Mean Sentiment'].idxmin()
        min_sentiment_year = df_results_search_term.loc[min_sentiment_idx, 'Year']
        min_sentiment_value = df_results_search_term.loc[min_sentiment_idx, 'Mean Sentiment']
        ax1.annotate(f'Lowest: {min_sentiment_value:.2f}',
                     xy=(min_sentiment_year, min_sentiment_value),
                     xytext=(min_sentiment_year, min_sentiment_value + max(.07, .2 * min_sentiment_value)),
                     textcoords='data',
                     ha='center',
                     arrowprops=dict(facecolor='gray', edgecolor='gray', arrowstyle='->', linewidth=2, shrinkA=5,
                                     shrinkB=5))

        # Add absolute frequency annotations at the markers
        max_freq_idx = df_results_search_term['Relative Frequency'].idxmax()
        max_freq_year = df_results_search_term.loc[max_freq_idx, 'Year']
        max_freq_value = df_results_search_term.loc[max_freq_idx, 'Relative Frequency']
        abs_max_freq = df_results_search_term.loc[max_freq_idx, 'Absolute Frequency']
        ax2.annotate(abs_max_freq,
                     xy=(max_freq_year, max_freq_value),
                     xytext=(max_freq_year, max_freq_value - max(.005, .05 * max_freq_value)),
                     textcoords='data', ha='center', va='top', fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        min_freq_idx = df_results_search_term['Relative Frequency'].idxmin()
        min_freq_year = df_results_search_term.loc[min_freq_idx, 'Year']
        min_freq_value = df_results_search_term.loc[min_freq_idx, 'Relative Frequency']
        abs_min_freq = df_results_search_term.loc[min_freq_idx, 'Absolute Frequency']
        ax2.annotate(abs_min_freq,
                     xy=(min_freq_year, min_freq_value),
                     xytext=(min_freq_year, min_freq_value + max(.005, .05 * min_freq_value)),
                     textcoords='data', ha='center', va='bottom', fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # Tight layout
        plt.tight_layout()

        if save_img:
            output_path = os.path.join(output_folder_path, f'entity_frequency_{k}.png')
            plt.savefig(output_path)

            config.logger.info(f'Saved image to {output_path}.')
        else:
            plt.show()

        plt.clf()


def entities_in_context_of_search_term(dataset: Dataset, cfg: Config, search_terms: list = [],
                                       save_img: bool = True):
    search_terms.insert(0, '')
    for year in dataset.get_years_covered():
        docs_folder_path = os.path.join(cfg.plot_output_path, 'Frequency', 'Entity', year)
        Path(docs_folder_path).mkdir(parents=True, exist_ok=True)

        cfg.logger.info(f'Load dataset of year {year}.')
        df = dataset.load_preprocessed(year) if vader.config.use_preprocessed_data else dataset.load_raw(year)
        df.reset_index(inplace=True)

        batch_cnt = math.ceil(len(df) / cfg.docbin_batch_size)
        cfg.logger.info(
            f'[Search Entity in Context] Starting frequency and sentiment calculation with entity recognizer for {year} with {len(df)} comments using {batch_cnt} batch(es).')

        docs = []
        if len([file for file in os.listdir(docs_folder_path) if file.endswith('.bin')]) > 0:
            for idx in range(batch_cnt):
                doc_bin = DocBin().from_disk(os.path.join(docs_folder_path, f'spaCy_docs_{year}_{idx}.bin'))
                docs_loaded = list(doc_bin.get_docs(nlp.vocab))
                docs.extend(docs_loaded)  # Combine all documents into a single list if needed
                cfg.logger.info(f'Loaded batch of {len(docs_loaded):06} ({idx + 1} / {batch_cnt}) docs.')
        else:
            # Create batches of texts to process them separately
            batches = [df['body'][i:i + cfg.docbin_batch_size].tolist() for i in
                       range(0, len(df), cfg.docbin_batch_size)]
            # Create file for each batch
            for idx, batch in enumerate(batches):
                docs_batch = [nlp(clean_text(text)) for text in batch]
                docs.extend(docs_batch)
                doc_bin = DocBin(docs=docs_batch)
                doc_bin.to_disk(os.path.join(docs_folder_path, f'spaCy_docs_{year}_{idx}.bin'))
                cfg.logger.info(
                    f'Saved batch of {len(batch):06} ({idx + 1} / {len(batches)}) docs to .bin file in {docs_folder_path}.')

        for search_term in search_terms:
            used_docs = docs
            logger_desc = f'on search term \'{search_term}\'' if search_term != '' else '(all)'
            cfg.logger.info(f'Starting entity recognizer {logger_desc} [{year}].')
            if search_term != '':
                # Find indices where the text contains the search word (case-insensitive)
                indices = df[df['body'].str.contains(search_term, case=False, na=False)].index
                # Use indices to access values from the docs list
                used_docs = [docs[i] for i in indices]

            entities_with_sentiment = []
            for i, doc in enumerate(used_docs):
                seen_entities = set()
                for ent in doc.ents:
                    normalized_text = normalize_entity(ent.text)
                    label = correct_entity_label(normalized_text, ent.label_)
                    if label in valid_entity_labels() \
                            and normalized_text.lower() not in entity_exclusion_list():
                        if normalized_text not in seen_entities:
                            entities_with_sentiment.append((normalized_text, label, df['sentiment'].iloc[i]))
                            seen_entities.add(normalized_text)
            # and not re.search(r'[^\w]', ent.text, re.UNICODE)
            results = {}
            for entity_type in valid_entity_labels():
                entities_of_type = [(ent, sentiment)
                                    for ent, label, sentiment in entities_with_sentiment
                                    if label == entity_type]

                # Count frequencies and sum sentiments
                entity_freq = Counter([ent for ent, sentiment in entities_of_type])
                entity_sentiment_sum = defaultdict(float)
                entity_sentiment_count = defaultdict(int)

                for ent, sentiment in entities_of_type:
                    entity_sentiment_sum[ent] += sentiment
                    entity_sentiment_count[ent] += 1

                # Calculate average sentiment
                entity_avg_sentiment = {ent: entity_sentiment_sum[ent] / entity_sentiment_count[ent] for ent in
                                        entity_freq}

                # Get top 15 entities by frequency
                top_15_entities = dict(entity_freq.most_common(15))

                results[entity_type] = {
                    'labels': list(top_15_entities.keys()),
                    'frequencies': [top_15_entities[label] for label in top_15_entities],
                    'avg_sentiments': [entity_avg_sentiment[label] for label in top_15_entities]
                }

            # Plotting for each entity type
            for entity_type, data in results.items():
                labels = data['labels']
                frequencies = data['frequencies']
                avg_sentiments = data['avg_sentiments']

                if len(labels) > 0:
                    plot_entity_frequencies_and_sentiment(labels, frequencies, avg_sentiments, len(docs),
                                                          entity_type, search_term, year, save_img,
                                                          cfg.plot_output_path)


def network_analysis(dataset: Dataset, cfg: Config, entity_limit: int = 20, save_img: bool = True):
    def extract_entity_sentiments(docs, df):
        entity_pairs = []
        entity_sentiments = defaultdict(lambda: {'sentiment_sum': 0, 'count': 0})

        for i, doc in enumerate(docs):
            seen_pairs = set()
            entities = [normalize_entity(ent.text)
                        for ent in doc.ents
                        if correct_entity_label(normalize_entity(ent.text), ent.label_) in valid_entity_labels()
                        and normalize_entity(ent.text).lower() not in entity_exclusion_list()]

            for j in range(len(entities)):
                for k in range(j + 1, len(entities)):
                    if entities[j] != entities[k]:
                        pair = tuple(sorted([entities[j], entities[k]]))
                        if pair not in seen_pairs:
                            entity_pairs.append(pair)
                            seen_pairs.add(pair)

            # Aggregate sentiment for entities
            for entity in set(entities):
                sentiment = df.iloc[i]['sentiment']
                entity_sentiments[entity]['sentiment_sum'] += sentiment
                entity_sentiments[entity]['count'] += 1

        return entity_pairs, entity_sentiments

    for year in dataset.get_years_covered():
        # Folder for reading and writing .doc files
        docs_folder_path = os.path.join(cfg.plot_output_path, 'Frequency', 'Entity', year)
        Path(docs_folder_path).mkdir(parents=True, exist_ok=True)
        # Folder for output of network analysis images
        output_folder_path = os.path.join(cfg.plot_output_path, 'Network Analysis', year, f'Entities{entity_limit}')
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)

        cfg.logger.info(f'Load dataset of year {year}.')
        df = dataset.load_preprocessed(year) if vader.config.use_preprocessed_data else dataset.load_raw(year)
        df.reset_index(inplace=True)

        batch_cnt = math.ceil(len(df) / cfg.docbin_batch_size)
        cfg.logger.info(
            f'Starting network analysis with entity recognizer (with spaCy) for {year} with {len(df)} comments using {batch_cnt} batch(es).')

        docs = []
        if len([file for file in os.listdir(docs_folder_path) if file.endswith('.bin')]) > 0:
            for idx in range(batch_cnt):
                doc_bin = DocBin().from_disk(os.path.join(docs_folder_path, f'spaCy_docs_{year}_{idx}.bin'))
                docs_loaded = list(doc_bin.get_docs(nlp.vocab))
                docs.extend(docs_loaded)  # Combine all documents into a single list if needed
                cfg.logger.info(f'Loaded batch of {len(docs_loaded):06} ({idx + 1} / {batch_cnt}) docs.')
        else:
            # Create batches of texts to process them separately
            batches = [df['body'][i:i + cfg.docbin_batch_size].tolist() for i in
                       range(0, len(df), cfg.docbin_batch_size)]
            # Create file for each batch
            for idx, batch in enumerate(batches):
                docs_batch = [nlp(clean_text(text)) for text in batch]
                docs.extend(docs_batch)
                doc_bin = DocBin(docs=docs_batch)
                doc_bin.to_disk(os.path.join(docs_folder_path, f'spaCy_docs_{year}_{idx}.bin'))
                cfg.logger.info(
                    f'Saved batch of {len(batch):06} ({idx + 1} / {len(batches)}) docs to .bin file in {docs_folder_path}.')

        valid_indices = df['sentiment'].notna()
        df = df[valid_indices].reset_index(drop=True)
        docs = [docs[i] for i in range(len(docs)) if valid_indices[i]]

        entity_pairs, entity_sentiments = extract_entity_sentiments(docs, df)

        entity_frequency = {entity: data['count'] for entity, data in entity_sentiments.items()}
        top_n_entities = {entity for entity, _ in Counter(entity_frequency).most_common(entity_limit)}

        filtered_entity_pairs = []
        for pair in entity_pairs:
            if pair[0] in top_n_entities and pair[1] in top_n_entities:
                filtered_entity_pairs.append(pair)

        filtered_pair_counts = Counter(filtered_entity_pairs)
        G = nx.Graph()

        for pair, count in filtered_pair_counts.items():
            G.add_edge(pair[0], pair[1], weight=count)

        node_sentiments = {entity: entity_sentiments[entity]['sentiment_sum'] / entity_sentiments[entity]['count']
                           for entity in G.nodes() if entity_sentiments[entity]['count'] > 0}

        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc).copy()

        norm = plt.Normalize(vmin=-1, vmax=1)
        node_colors = plt.cm.coolwarm(norm(list(node_sentiments.values())))

        max_degree = max(dict(G_sub.degree()).values())
        node_sizes = [3000 * (G_sub.degree(n) / max_degree) for n in G_sub.nodes()]

        edge_weights = [G_sub[u][v]['weight'] for u, v in G_sub.edges()]
        max_weight = max(edge_weights) if edge_weights else 1

        if edge_weights:
            edge_weight_threshold = np.percentile(edge_weights, 20)
        else:
            edge_weight_threshold = 0
        edges_to_draw = [(u, v) for u, v in G_sub.edges() if G_sub[u][v]['weight'] >= edge_weight_threshold]
        edge_weights_to_draw = [0.5 + 2 * (G_sub[u][v]['weight'] / max_weight) for u, v in edges_to_draw]

        for layout in ['spring', 'kamada_kawai', 'circular', 'spectral', 'shell']:
            if layout == 'spring':
                pos = nx.spring_layout(G_sub, k=0.3, iterations=1000, seed=42, scale=3.0)
            elif layout == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G_sub)
            elif layout == 'circular':
                pos = nx.circular_layout(G_sub)
            elif layout == 'spectral':
                pos = nx.spectral_layout(G_sub)
            elif layout == 'shell':
                pos = nx.shell_layout(G_sub)
            elif layout == 'graphviz_neato':
                pos = nx.nx_agraph.graphviz_layout(G_sub, prog='neato')
            elif layout == 'graphviz_fdp':
                pos = nx.nx_agraph.graphviz_layout(G_sub, prog='fdp')
            else:
                cfg.logger.info(f'Unsupported layout type: {layout}. Skipped')
                continue

            cfg.logger.info(f'Starting network analysis with layout \'{layout}\'.')

            fig, ax = plt.subplots(figsize=(17, 11))

            nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, node_size=node_sizes, cmap='coolwarm', alpha=0.7,
                                   ax=ax)
            nx.draw_networkx_edges(G_sub, pos, edgelist=edges_to_draw, width=edge_weights_to_draw, alpha=0.6,
                                   edge_color='grey', ax=ax)
            nx.draw_networkx_labels(G_sub, pos, font_size=10, font_family='sans-serif', ax=ax)

            sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Sentiment Score')

            ax.set_title(f'Network of Climate Change Discourse in {year} with {entity_limit} Entities')
            ax.axis('off')
            plt.tight_layout()

            if save_img:
                output_path = os.path.join(output_folder_path,
                                           f'network_analysis_top{entity_limit}_{year}_{layout}.png')
                plt.savefig(output_path)

                config.logger.info(f'Saved image to {output_path}.')
            else:
                plt.show()

            plt.clf()


def clean_text(text):
    # Step 1: Remove patterns like "--gt&;", "gt&;", etc.
    text = re.sub(r'-*&gt;', '', text)
    # Step 2: Replace multiple whitespaces with a single whitespace
    text = re.sub(r'\s+', ' ', text)
    # Step 3: Trim any leading or trailing whitespaces
    text = text.strip()

    return text


def plot_entity_frequencies_and_sentiment(labels, frequencies, avg_sentiments, doc_count, title_specific, search_term,
                                          year, save_img: bool, base_output_folder_path: str):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Normalize sentiment for color mapping (0 to 1)
    norm = plt.Normalize(vmin=-1, vmax=1)
    colors = plt.cm.coolwarm(norm(avg_sentiments))

    bars = ax.bar(labels, 100 * np.array(frequencies) / doc_count, color=colors)
    ax.set_ylabel('Frequency [%]')

    # Set the x-ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha='right')

    # Add absolute frequency labels
    for bar, value in zip(bars, frequencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{value}', ha='center', va='center', color='black',
                fontsize=10, rotation='vertical')

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Sentiment Score')

    ax.set_ylabel('Frequency [%]')
    plt.xticks(rotation=60, ha='right', fontsize=12)

    title_desc = '' if search_term == '' else f' for \'{search_term}\''
    plt.suptitle(f'Frequency and Sentiment of {title_specific} mentioned{title_desc} in {year}', y=.95, fontsize=18)
    plt.title('(Numbers within the bars are the absolute frequencies)', fontsize=10)

    fig.tight_layout()

    if save_img:
        output_folder_path = os.path.join(base_output_folder_path, 'Frequency', 'Entity', year, title_specific)

        Path(output_folder_path).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_folder_path,
                                   f'{title_specific}{"" if search_term == "" else f"_{search_term}"}_{year}.png')
        plt.savefig(output_path)
    else:
        plt.show()

    plt.clf()


def entity_displacy_plot(dataset: Dataset, cfg: Config, search_terms: [str]):
    output_path_csv = os.path.join(cfg.output_folder, 'spaCy_displacy', 'spaCy_displacy.csv')
    for search_term in search_terms:
        output_folder_path = os.path.join(cfg.output_folder, 'spaCy_displacy', search_term)

        for year in dataset.get_years_covered():
            dataset.config.logger.info(f'Starting entity displacy on year {year} for search term \'{search_term}\'.')
            df = dataset.load_preprocessed(year) if vader.config.use_preprocessed_data else dataset.load_raw(year)
            df = df[df['sentiment'].notna()]

            df['term'] = df['body'].apply(lambda text: search_term.lower() in text.lower())
            filtered_df = df[df['term']]

            if filtered_df.size > 0:
                # Sort the filtered DataFrame by sentiment scores
                filtered_df_sorted = filtered_df.sort_values(by='sentiment')

                # Select the entries with the lowest and highest sentiment scores
                entry_lowest_sentiment = filtered_df_sorted.head(1)
                entry_highest_sentiment = filtered_df_sorted.tail(1)

                Path(output_folder_path).mkdir(parents=True, exist_ok=True)

                visualize_entities(entry_lowest_sentiment['body'].iloc[0], search_term, year,
                                   f'{search_term}_lowest', output_folder_path)
                visualize_entities(entry_highest_sentiment['body'].iloc[0], search_term, year,
                                   f'{search_term}_highest', output_folder_path)

                # tbd
                result = []
                result.append([year, search_term, 'negative',
                               entry_lowest_sentiment['id'].iloc[0], entry_lowest_sentiment['sentiment'].iloc[0]])
                result.append([year, search_term, 'positive',
                               entry_highest_sentiment['id'].iloc[0], entry_highest_sentiment['sentiment'].iloc[0]])

                df_result = pd.DataFrame(result, columns=['year', 'search_term', 'class', 'id', 'sentiment'])
                df_result.to_csv(output_path_csv, mode='a', sep=';', index=False,
                                 header=not os.path.exists(output_path_csv))
            else:
                dataset.config.logger.warning(
                    f'There is no comment available for search term \'{search_term}\' in {year}.')


def visualize_entities(text: str, search_term: str, year: str, id: str, out: str):
    def correct_entity_labels(doc):
        ents = []
        for ent in doc.ents:
            corrected_label = correct_entity_label(normalize_entity(ent.text), ent.label_)
            ents.append(Span(doc, ent.start, ent.end, label=corrected_label))
        doc.ents = ents  # Update the doc's entities with the corrected ones

        return doc

    # Output
    output_to = os.path.join(out, f'displacy_{id}_{year}.html')
    # Define a function to process and visualize entities in text
    doc = nlp(clean_text(text))
    # Initialize the Matcher with the shared vocabulary
    matcher = Matcher(nlp.vocab)
    # Add custom entities ti
    doc.ents = add_custom_entities(doc, matcher, search_term)
    # Correct entity labels
    doc = correct_entity_labels(doc)
    # Use displacy to render the document
    html = displacy.render(doc, style='ent', page=True,
                           options={'ents': valid_entity_labels() + ['CLIMATE CHANGE', 'SEARCH TERM']})

    with open(output_to, 'w', encoding='utf-8') as f:
        f.write(html)


def add_custom_entities(doc, matcher, search_term: str) -> str:
    search_term = search_term.split()

    # Define multiple patterns
    if 'climate change' in doc.text.lower():
        patterns1 = [
            [{"LOWER": "climate"}, {"LOWER": "change"}]
        ]
    else:
        patterns1 = [
            [{"LOWER": "climate"}],
            [{"LOWER": "change"}]
        ]

    pattern2 = []

    for term in search_term:
        pattern2.append({"LOWER": term})

    # Add the pattern to the matcher
    matcher.add("CLIMATE CHANGE", patterns1)
    matcher.add("SEARCH TERM", [pattern2])

    # Find matches in the doc
    matches = matcher(doc)

    # Sort matches by their start position, and prioritize longer matches
    matches = sorted(matches, key=lambda x: (x[1], x[2] - x[1]), reverse=True)

    # Create a new list of entities, starting with existing ones if necessary
    new_ents = list(doc.ents)  # Start with existing entities

    # Check for overlaps
    seen_tokens = set()
    for ent in doc.ents:
        seen_tokens.update(range(ent.start, ent.end))

    for match_id, start, end in matches:
        if any(start <= token < end for token in seen_tokens):
            continue  # Skip this match as it overlaps with an existing entity
        # Create a new Span for each match and assign the label accordingly
        span = Span(doc, start, end, label=nlp.vocab.strings[match_id])
        new_ents.append(span)

    return new_ents


def boxplot_comment_length(cfg: Config, save_img: bool = True):
    df = pd.read_csv(cfg.dataset.filename, usecols=['created_utc', 'body', 'sentiment'])
    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    # Extract year from datetime
    df['year'] = df['datetime'].dt.year
    # Determine sentiment class
    df['class'] = df['sentiment'].apply(lambda x: 'Negative' if x < -0.05 else 'Positive' if x > 0.05 else 'Neutral')

    # Create a new column for the number of characters per comment
    df['char_count'] = df['body'].apply(len)

    # Calculate the mean and standard deviation of the character count per sentiment class
    char_count_stats = df.groupby('class')['char_count'].agg(['mean', 'std']).reset_index()

    # Ensure the 'year' column is in the correct format if necessary
    df['year'] = df['year'].astype(int)

    # Set up the custom color palette
    palette = {"Positive": "green", "Neutral": "gray", "Negative": "red"}

    # Set up the figure and axis
    plt.figure(figsize=(20, 10))

    sns.boxplot(x='year', y='char_count', hue='class', data=df,
                palette=palette, showfliers=False, showmeans=True, showcaps=True, notch=True,
                medianprops={"color": "orange", "linewidth": 3}, meanprops={"marker": (8, 1, 0),
                                                                            "markerfacecolor": "yellow",
                                                                            "markeredgecolor": "yellow",
                                                                            "markersize": "8"})

    # Add titles and labels
    plt.title('Number of Characters per Comment by Year', fontsize=20)
    plt.legend(title='Sentiment Class')
    plt.tight_layout()
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Number of Characters', fontsize=16)

    if save_img:
        output_path = os.path.join(cfg.plot_output_path, 'comment_length_boxplot.png')
        plt.savefig(output_path)

        cfg.logger.info(f'Saved image to {output_path}.')
    else:
        plt.show()

    plt.clf()

    # Set up the figure and axis
    plt.figure(figsize=(20, 10))

    # Create the boxen plot
    sns.boxenplot(x='year', y='char_count', hue='class', data=df,
                  showfliers=False, width=0.3, palette=palette)

    # Add titles and labels
    plt.title('Number of Characters per Comment by Year', fontsize=20)
    plt.tight_layout()
    plt.legend(title='Sentiment Class', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Number of Characters', fontsize=16)

    if save_img:
        output_path = os.path.join(cfg.plot_output_path, 'comment_length_boxenplot.png')
        plt.savefig(output_path)

        cfg.logger.info(f'Saved image to {output_path}.')
    else:
        plt.show()

    plt.clf()

    # Set up the figure and axis
    plt.figure(figsize=(25, 20))

    # Ensure the 'class' column is in categorical format with a specified order
    df['class'] = pd.Categorical(df['class'], categories=["Positive", "Neutral", "Negative"], ordered=True)

    # Calculate the 99th percentile for the y-axis limit
    y_limit = df['char_count'].quantile(0.9999)

    # Set up the FacetGrid with hue for class
    g = sns.FacetGrid(df, col="year", col_wrap=5, height=4, sharey=True)

    # Map the boxenplot and stripplot to each facet with hue and dodge
    # g.map_dataframe(sns.boxenplot, x="class", y="char_count", hue="class", dodge=True, palette=palette,
    #               showfliers=False, linewidth=1.5)
    g.map_dataframe(sns.stripplot, x="class", y="char_count", hue="class", jitter=True, size=1.5, alpha=0.2,
                    palette=palette, dodge=True)

    # Set the y-axis limit for all plots
    for ax in g.axes.flat:
        ax.set_ylim(0, y_limit)

    # Remove duplicate legends
    for ax in g.axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend().remove()

    # Set titles and axis labels
    g.set_titles(col_template="{col_name}")
    g.fig.suptitle('Number of Characters per Comment by Year')
    plt.tight_layout()
    g.set_axis_labels("Class", "Number of Characters")

    if save_img:
        output_path = os.path.join(cfg.plot_output_path, 'comment_length_facetgrid.png')
        plt.savefig(output_path)

        cfg.logger.info(f'Saved image to {output_path}.')
    else:
        plt.show()

    plt.clf()


if __name__ == '__main__':
    # Init configuration with logger
    config = Config()

    # Start logger
    if config.log_enable_health:
        logger = config.logger
        p = Process(target=monitor_pc_health)
        p.daemon = True
        p.start()
        logger.info('Started computer health monitoring.')

    ### Load and preprocess datset
    ds = Dataset(config)
    if config.dataset.consider:
        if config.dataset.create_feather_files:
            ds.load()
            ds.to_feather()

        if config.dataset.preprocess:
            ds.preprocess()

    ### VADER
    vader_config = config.methodologies['vader']
    vader = Vader(config)
    if vader_config.consider and vader_config.recalc:
        vader.polarity()

    # Compare kaggle sentiment with vader compound
    if 'vader' in config.comparators_to_kaggle.keys() and config.comparators_to_kaggle['vader']:
        kaggle_vs_vader(ds, vader)

    if config.dataset.overwrite_sentiment_by == 'vader':
        vader.write_sentiment_to_source()

    ### Word Counter
    word_counter_config = config.methodologies['word_counter']
    if word_counter_config.recalc:
        word_counter_config = config.methodologies['word_counter']
        word_counter = WordCounter('word_counter', config)
        word_counter_out = word_counter.perform(word_counter_config.type)
    else:
        assert os.path.exists(config.vocab_file_path), \
            f'There is no word counter vocabulary file at {config.vocab_file_path}.'

        word_counter_out = pd.read_csv(config.vocab_file_path, sep=';', index_col=0)

    if config.word_counter_type == 'frequency':
        for column in word_counter_out.columns[:-1]:
            word_counter_out[column] = 100 * word_counter_out[column] / word_counter_out['total_frequency']
        word_counter_out = word_counter_out.drop(columns='total_frequency')

    if config.plot_topic_over_time:
        is_freq_data = config.word_counter_type == 'frequency'
        topic_over_time_line(word_counter_out,
                             config,
                             ngrams_to_plot=['sea level', 'greenhouse effect', 'extreme weather', 'carbon tax',
                                             'carbon dioxide'],
                             is_freq=is_freq_data,
                             save_img=True)
        if is_freq_data:
            topic_over_time_bar(word_counter_out, config, True)

    ### Google NGram
    if 'ngram' in config.comparators_to_kaggle.keys() and config.comparators_to_kaggle['ngram']:
        kaggle_vs_NGram(word_counter_out, config, True)

    ### Plots for General overview plots
    sentiment_per_year_plots(ds, config)
    samples_per_year_stacked(config)
    comment_dist_total(config)
    comment_dist_within_year(ds, config)
    boxplot_comment_length(config)
    subreddit_plot_update(config, top_n_subreddits=5)
    #interactive_average_sentiment_per_subreddit(config, 30)
    hashtags_frequency_per_year(config)
    wordcloud_per_class(config)
    entity_displacy_plot(ds, config,
                         search_terms=['emission', 'energy', 'government', 'nuclear', 'oil',
                                       'science', 'war', 'temperature', 'electric car', 'extreme weather',
                                       'fossil fuel', 'greenhouse effect', 'natural gas',
                                       'renewable energy'])
    network_analysis(ds, config, entity_limit=10, save_img=True)
    entities_in_context_of_search_term(ds, config,
                                       search_terms=['energy', 'government', 'nuclear', 'wind', 'oil', 'fossil',
                                                     'solar', 'sea level', 'extreme weather', 'greenhouse effect',
                                                     'carbon tax'])
    # entity_heatmap(ds, config, save_img=True,
    #              entities_with_labels=[('Australia', 'GPE'),
    #                                   ('Canada', 'GPE'),
    #                                  ('China', 'GPE'),
    #                                 ('India', 'GPE'),
    #                                ('United Kingdom', 'GPE'),
    #                               ('USA', 'GPE')])
    entity_heatmap(ds, config, save_img=True,
                   entities_with_labels=[('George W. Bush', 'PERSON'),
                                         ('Barack Obama', 'PERSON'),
                                         ('Donald Trump', 'PERSON'),
                                         ('Joe Biden', 'PERSON')])
    entity_frequency_and_sentiment(ds, config, save_img=True,
                                   search_terms={'IPCC': 'ORG',
                                                 'Google': 'ORG',
                                                 'Fox News': 'ORG',
                                                 'Tesla': 'ORG',
                                                 'CNN': 'ORG',
                                                 'Apple': 'ORG',
                                                 'Microsoft': 'ORG',
                                                 'ExxonMobile': 'ORG',
                                                 'General Motors': 'ORG',
                                                 'NASA': 'ORG',
                                                 'Amazon': 'ORG'})
