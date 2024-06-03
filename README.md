# Bachelors Thesis. Nurzat Dzholchubekova, LMU, MaiNLP, June 3. 2024.

Repository for the materials associated with the Bachelors Thesis 
**Climate Change Insights through NLP** at the LMU Munich, June 2024.

# Content

## Thesis

- The Python related code used for generation of all outputs can be found in folder **X**
- The complete LaTeX project with PDF and TEX files can be found in folder **Y**

## Python

### Version

The minimum required Python version is 3.10. However, it is 
recommended to use the latest version (currently 3.12), as it was 
used during development.

### Dependencies

Prerequisite for the execution is the dataset that needs to be downloaded from 
[Kaggle](https://www.kaggle.com/datasets/pavellexyr/the-reddit-climate-change-dataset). 
The CSV file __the-reddit-climate-change-dataset-comments.csv__ with approximately 
4.11 GB needs to be downloaded and put into the folder **X**. 

Following Python modules need be installed to run the Python scripts:
- math
- os
- re
- time
- psutil
- pandas
- seaborn
- numpy
- matplotlib
- multiprocessing
- plotly
- pathlib
- tqdm
- scikit-learn
- spacy
- wordcloud
- nltk
- networkx
- collections

The _en_core_web_sm_ pipeline is used for spaCy and must be downloaded 
before the first run. It can also be replaced with any other desired 
pipeline.

```bash
download("en_core_web_sm")
```

### Files

- evaluation.py
- config.py
- dataset.py
- methodology.py
- word_counter.py
- vader.py
- ngram_loader.py

Only the scripts _evaluation.py_ and _config.py_ need to be updated 
to obtain different plots. Here, _evaluation.py_ is the only script 
to run.

## Brief Documentation

It is recommended to run the script in terminal to receive better
logging information. If you run the script using PyCharm, you could
achieve an emulation of the terminal by adding the run option
_Emulate terminal in output console_:

![img.png](PyCharm.png)

Initially, the _evaluation.py_ has all available plots enabled.
Some of them might take some time to load and calculate. Logging
information in terminal are provided as much as possible. Some
logging information might seem odd. This is caused by the limited 
functionality of _tqdm_ and multiprocessing tasks. 

All configuration regarding logging, output folders and calculation
parameter are done in **config.py**. 

Most of the configuration regarding plots is done in **evaluation.py**. 
An exception are the plots over time regarding n-grams. 

### Unigram and Bigram Plots

The unigram and bigram plots highly depend on the _WordCounterConfiguration_
in **config.py**. Using the _max_words_ parameter in combination with the 
parameter _type_ as _frequency_ will output the most frequent n-grams
(depending on parameter _ngram_range_). Depending on this, the value for 
variable _vocab_file_path_ is calculated, which will be used to plot
n-grams. 

One possible approach could be to calculate the most frequent n-grams as
described above. Afterwards, you could manually cherry pick relevant 
n-grams and instead configure the _WordCounterConfiguration_ to only 
consider a specific vocabulary by using the equally named parameter. 
Valid options for parameter _type_ are _frequency_ and _sentiment_. 
The retrieved output file can be used to configure the 
_word_counter_filename_ variable. Lastly, the parameter _ngrams_to_plot_
of function **topic_over_time_line** in **evaluation.py** can be 
used to pick and plot only selected ngrams from the provided file. 

## License

[MIT](https://choosealicense.com/licenses/mit/)