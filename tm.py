import os
from flask import Flask, flash, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd
import re, nltk, spacy, gensim

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
import umap.umap_ as umap
import seaborn as sns

from sklearn.feature_extraction import text
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os



UPLOAD_FOLDER = r'C:\Users\noel.alexander\Documents\Fullstack\Topic Modelling\Uploads'
VIZ_FOLDER = r'C:\Users\noel.alexander\Documents\Fullstack\Topic Modelling\Viz'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIZ_FOLDER'] = VIZ_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return redirect(url_for('uploaded_file',
                                    #filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Applied Intelligence Studio Topic Modeller</h1>
    <h2>Upload A File</h2>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

from flask import send_from_directory

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def lemmatization(n,texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = n(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

def objective(space):
    print(space)
    global data_vectorized
    lda_model = LatentDirichletAllocation( n_components=int(space['n_topics']),    # number of topics
                                           learning_decay=space['learning_decay'], # control learning rate in the online learning method
                                           max_iter=10,                            # max learning iterations
                                           learning_method='online',               # use mini-batch of training data
                                           batch_size=128,                         # n docs in each learning iter
                                           n_jobs = -1,                            # use all available CPUs
                                         )
    
    lda_model.fit_transform(data_vectorized)
      
    score = lda_model.score(data_vectorized)
    print("SCORE:", score)
    return {'loss':-score, 'status': STATUS_OK } # minnimizing negative log-likelihood is equivalent to maximing log-likelihood

def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

# Show top n keywords for each topic
def show_topics(vectorizer, lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/analyze/<filename>')
def main(filename):
    global data_vectorized
    global lda_output
    global plot_df
    df = pd.read_csv(UPLOAD_FOLDER+'/'+filename) # CHANGE THIS 
    df = df.sample(frac=0.2, replace=False, random_state=1)
    N_NGRAM_RANGE = 2 # CHANGE HERE
    my_additional_stop_words = pd.read_csv(r'C:\Users\noel.alexander\Documents\Fullstack\Topic Modelling\Stopwords\custom_stopwords.csv').values.flatten().tolist() #CHANGE THIS
    stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)
    data = df.content.values.tolist()

    # Remove Emails
    data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub(r'\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    data_words = list(sent_to_words(data))
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only Noun, Adj, Verb, Adverb
    data_lemmatized = lemmatization(n=nlp,texts = data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    vectorizer = CountVectorizer(analyzer='word',       
                                min_df=0.05,                      # ignore terms that appear in less than 5% of the documents
                                stop_words=stop_words,            # remove stop words
                                lowercase=True,                   # convert all words to lowercase
                                token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                ngram_range=(1, N_NGRAM_RANGE)
                                )

    data_vectorized = vectorizer.fit_transform(data_lemmatized)
    space ={'n_topics': hp.quniform("n_topics", 6, 10, 1),           # search n_topics from 2-20
        'learning_decay': hp.uniform ('learning_decay',0.5,0.9), # search learning_decay from 0.5-0.9
       }

    trials = Trials()
    
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=25,
                trials=trials)
    
    LEARNING_DECAY = best['learning_decay'] #0.84529 #best['learning_decay']
    N_TOPICS = best['n_topics'] #9 #best['n_topics']
    print('starting lda')
    # Build LDA Model
    lda_model = LatentDirichletAllocation(n_components=int(N_TOPICS),   # number of topics
                                        learning_decay=LEARNING_DECAY, # control learning rate in the online learning method
                                        max_iter=10,                   # max learning iterations
                                        learning_method='online',      # use mini-batch of training data
                                        batch_size=128,                # n docs in each learning iter
                                        n_jobs = -1,                   # use all available CPUs
                                        )

    lda_output = lda_model.fit_transform(data_vectorized)
    lda_output = lda_model.transform(data_vectorized)

    # column names
    topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]

    # index names
    docnames = ["Doc" + str(i) for i in range(len(data))]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic
    # Apply Style
    df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
    df_topic_distribution.columns = ['Topic Num', 'Num Documents']
    df_topic_distribution['Percent of Total'] = round(df_topic_distribution['Num Documents'] / np.sum(df_topic_distribution['Num Documents'].values),2)
    topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=15)        

    # Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
    #pyLDAvis.enable_notebook()
    panel = pyLDAvis.sklearn.prepare(lda_model, data_vectorized, vectorizer, mds='tsne')
    topics_dic ={}
    for i in range(int(N_TOPICS)):
        topics_dic[i] = 'topic ' + str(i)
    plot_df = pd.DataFrame({'topics':labels})
    plot_df['topics'] = plot_df['topics'].map(topics_dic)
    labels = []
    for doc in lda_output:
        labels.append(np.argmax(doc))
    labels = np.array(labels)

    embedding = umap.UMAP(n_neighbors=100, min_dist=0.9).fit_transform(lda_output)

    plot_df['axis_1'] = embedding[:, 0]
    plot_df['axis_2'] = embedding[:, 1]

    html = pyLDAvis.prepared_data_to_html(panel)
    
    return html


@app.route('/visualize/<filename>')
def tensor(filename):
    global plot_df
    LOG_DIR = r'C:\Users\noel.alexander\Documents\Fullstack\Topic Modelling\logs'
    metadata = os.path.join(LOG_DIR, 'metadata.tsv')

    docs = tf.Variable(lda_output, name='docs')
    
    with open(metadata, 'w') as metadata_file:
        metadata_file.write('Index' + '\t' + 'Topic'  + '\n')
        for index, row in plot_df.iterrows():
            metadata_file.write(str(index) + '\t' + row['topics']  + '\n') 

    with tf.Session() as sess:
        saver = tf.train.Saver([docs])

        sess.run(docs.initializer)
        saver.save(sess, os.path.join(LOG_DIR, 'docs.ckpt'))

        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = docs.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = metadata
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
        os.system('tensorboard --logdir=logs')
        return '''
    <!doctype html>
    <body>
    <iframe src="http://localhost:6006/"></iframe>
    </body>
    '''

if __name__ == '__main__':
    data_vectorized = None
    lda_output = None
    plot_df = None
    app.debug = True
    app.run(host = '0.0.0.0',port=5000)