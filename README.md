# Statistical Machine Translation Using the IBM Models
Official repository for team #14 in *CS F469: Information Retrieval* [2019-20 Fall Semester]
at BITS Pilani, Pilani.  

Team #14:
  - [Hemanth V. Alluri](https://github.com/Hypro999) (2017A7PS1170P) [Maintainer of the datasets
  on [Google Drive](https://drive.google.com/drive/folders/1efH-6oDGVqrvrLyjAm2XcZqd9f-_SXGO?usp=sharing)]
  - [S. Hariharan](https://github.com/hariharan-srikrishnan) (2017A7PS0134P)
  - [Nevin Thomas](https://github.com/lesasi) (2017A7PS1175P)
  - [Sanchit Ahuja](https://github.com/sanchit-ahuja) (2017A3PS0216P) [Maintainer of the
  [GitHub mirror](https://github.com/sanchit-ahuja/ir-project.git)]

<hr>

### Introduction
The aim of this project is to implement statistical machine translation (SMT) using IBM
Models 1 and 2. In particular, we attempt to translate between Dutch and English. SMT was
taught during the course in lectures 18-22 and all of the relevant resources for gaining a
better understanding of the topic are included in /docs/reference. Also, the book titled
[Statistical Machine Translation](http://www.statmt.org/book/) by Philipp Kohen is a
recommended read.

A rubric outlining the project expectations, requirements and grading schema can be found
in /docs/rubric.

<hr>

### Requirements
The only real requirement is **Python 3**. The OS is more or less irrelevant here. The more
hardware you can provide the better.  

**Warning**: Avoid using any more than 10% of the original raw dataset. As you increase the
size of the dataset used both the RAM usage during training as well as disk usage by the
pickled trained model will increase greatly (especially with a lower "vocabulary reduction
threshold" - vocab reduction threshold is the number of time a word must occur before we
add it to the vocab - this helps with keeping the dimensions of training lower so that the
data will actually fit in memory).

<hr>

### Directory Structure
Note: `/` references this repository's root and not the file system root.

#### /datasets [on [Google Drive](https://drive.google.com/drive/folders/1efH-6oDGVqrvrLyjAm2XcZqd9f-_SXGO?usp=sharing)]
As the name implies, all of our datasets are stored here.  

The results report is also available here.
1. **/datasets/raw:** The raw parallel corpus between English and Dutch as provided by the
course instructor. Currently in a zipped format.
2. **/datasets/testing:** The test data to use after training is over.
3. **/datasets/training:** The preprocessed data from the raw parallel corpus labelled by
percentage of the original corpus used and the reduction threshold value.
4. **/datasets/model:** The results of our training are stored here based on the dataset used
for training, and number of iterations spent training.
5. **/datasets/translations:** Using the models in /datasets/models and the testing data in
/datasets/testing we generated a few sample translations and put them here.

#### /docs [on [Google Drive](https://drive.google.com/drive/folders/1efH-6oDGVqrvrLyjAm2XcZqd9f-_SXGO?usp=sharing)]
Just a few relevant documents for this course like powerpoint presentations on SMT and the
IBM models.

### /project
All of our code is stored here in the form of a python package.

#### /project/core
All of the main code for preprocessing, training, translating and then measuring accuracy
(testing) is stored here. `train.py` contains code for the IBM Model 1 and `train2.py` contains
code for IBM Model 2. `test` contains code for determining the cosine similarity and Pearson's
coefficient between 2 documents (e.g. actual translation vs machine translation).

#### /project/tools
A few useful command line tools.  
  `unpickle.py`: For unpickling pickled data.  
  `analyze.py`: Take a translation probabilities table (as a pickled file on disk) and a word
   and give the top 15 possible translation for it according to the table.

<hr>

### Tutorial and Usage Guide for IBM Model 1
**Note:** all of the CLIs support passing the `-h` or `--help` command line option to find out
exactly what options they provide. Please use this before running any command to know what all
of your options are.

#### Step One: Preprocessing the data
`preprocess.py` contains all of the preprocessing code and comes with a CLI to perform
custom preprocessing with ease. Alternatively, you could (and perhaps just should) use
the already-ready preprocessed files in /datasets/training. If you want to make your own
data files, then first make sure unzip the raw datasets in /datasets/raw.   
Example usage:   
`python -m project.core.preprocess datasets/raw/English.txt -v -t 4 -p 5 -o english_processed`  
This will produce a 5% dataset (with respect to the original dataset size) and a vocabulary
reduction threshold of 4. The files `english_processed.pkl` and `english_processed.reduced.pkl`.
The former does not use the vocab reduction threshold while the later does.

#### Step Two: Using the preprocessed data in training the IBM Model 1
`train.py` (formerly `engine.py`) contains the code for the IBM Model 1 Expectation Maximization
algorithm. This will take two preprocessed data files and then produce another datafile which
is essentially of the form {foreign word: {english word: probability}} (as a pickled Python
dictionary). More information can be gathered from the source file. Currently, the CLI only
supports using one of the default preprocessed training files but modifying just two lines
of code can be sufficient to allow you to use which ever datasets you like.
Example usage:  
`python -m project.core.train -p 3 -m 5 -c 0.0001 -w`  
This example will generate a file called `translation_probabilities_table.pkl` which will be
used for Dutch -> English translation. The datasets we will use are 3% of the original raw
dataset and have a vocab reduction threshold of 5 each (these are part of the default preprocessed
set). The `-c 0.0001` part means that we will set a convergence factor of 0.0001 and the `-m 5`
means perform 5 iterations at most and stop even if the data has not converged. The `-w` option
means that at the end of the epoach we will write back the results to disk so that if anything
goes wrong, we can recover and continue using the training data that was saved after each epoch
using the `-r` option (use `--help` for more information). The `-r` is also useful for doing
more epochs of training using a larger `-m` value.  The `-i` invert option (not used here) can be
used for generating a file for English -> Dutch translation.

#### Step Three: Using the trained model to translate test data
`translate.py` contains the code for using the trained model to translate from one language to
another.
Example usage:  
`python -m project.core.translate datasets/testing/testdata_dutch.txt -t translation_probabilities_table.pkl -v`  
The translator also comes with an "augment" option ("-a") to improve translation a bit,
especially with less trained models. Take the Dutch phrase "u bent", in English it's
equivalent to "you are". The word "bent" occurs fairly commonly next to "u", and without
sufficient training the model will think that "bent" translates to "you" instead of "are"
which is what it really is. The reason for this is roughly that the word "u" is more common
than "bent" and since "bent" often occurs near "u" (e.g. take "u hebt gelijk zoals u vaak
bent" -> "are you sure that you think that?"), without sufficient training, the model will map
both "u" and "bent" to "you". But we can still exploit the fact that the model will still realize
that the two occurrences of "u" more strongly correlate it to "you" (which also occurred twice)
than "bent" correlates to "u". So what augment does is that it will modify translation such that
if the translator notices two of the same words back to back, one will be swapped out with it's
second most likely word for a more accurate translation of short phrases. Using this flag is
recommended if the model has not received much training.

#### Step Four: Testing the results of translation
Apart from a manual verification to see if the translated sentence roughly means the same as the
actual sentence, we can use "Cosine Similarity" and "Pearson's Coefficient" to get a numerical
score for the quality of translation. `test.py` is used for this.  
Example usage:
`python -m project.core.test translation.txt datasets/testing/testdata_english.txt`

<hr>

### Difficulties Faced, and Shortcomings/Drawbacks
1. High degree of dimensions - the number of parameters are essentially |english_vocab| x
|dutch_vocab| which can easily grow with the size of the dataset considered. This means that
hardware constraints enter the picture and make it difficult to fit the translation probabilities
table and counts in memory simultaneously (since both of them are massive).
2. The IBM Model 1 does not consider alignments so the translation sounds awkward despite it still
capturing the essence of the sentence.
4. Synonyms are problematic for exact translation and lead to lower cosine similarity ratings.


### Updates:
To deal with the memory issue, we've replaced the dictionary of dictionaries translation table
with a list of lists which we refer to as a "translation matrix" which is much more efficient.
The code added now makes training take less RAM and less disk space. The `-x` option can be
used to prefer using translation matricies instead of using translation tables.  
Example usage:  
`python -m project.core.train -p 5 -m 10 -v -x`  
`python -m project.core.translate datasets\testing\testdata_dutch.txt -t translation_probabilities_matrix.pkl -v -x`

<hr>

### IBM Model 2

#### Summary
The IBM Model 2 has an additional model for alignment that is not present in Model 1. The translation
of a foreign input word in position i to an English word in position j is modelled by an alignment
probability distribution a(i|j,le, lf) where le and lf are the respective lengths of the English and foreign
words.

#### Step 1: Training IBM Model 2
`train2.py` contains the code for training IBM Model 2. This takes in the two pre-processed datasets(English
and foreign) as well as the file `translation_probabilities_table.pkl` generated from IBM Model 1 as inputs,
and produces(as output) two files: an updated `final_translation_prob.pkl` as well as a new
`final_alignment_prob.pkl` that contains the probabilities of all possible sentence alignments.
Example usage:
`python train2.py`

#### Step 2: Use trained model to get final aligned sentences
This takes in Dutch sentences and returns the corresponding English translation, with corrected
alignments.
This takes in the translation_probabilities_table and alignment_probability_table generated in the
previous step, applies the translate function of IBM model 1 in order to get corresponding one-to-one
mapping, and then applies alignment correction module to fix alignments.
Example usage:
`python translate2.py`
