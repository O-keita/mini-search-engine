from flask import Flask, render_template
import nltk
from nltk.stem import WordNetLemmatizer
from  nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from form import SearchForm


app = Flask(__name__)

app.config['SECRET_KEY'] = '99733f54e5f2e4d129ba292187176e53'

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


file_paths = [
    {"path":"files/paper1.pdf", "author":"International Journal of Science and Research (IJSR)", "title":"Machine Learning Algorithms - A Review", "link":"https://www.researchgate.net/profile/Batta-Mahesh/publication/344717762_Machine_Learning_Algorithms_-A_Review/links/5f8b2365299bf1b53e2d243a/Machine-Learning-Algorithms-A-Review.pdf?eid=5082902844932096t"},
    {"path":"files/file1.pdf", "author":"E Alpaydin - 2021", "title":"Machine Learning", "link":"https://cs.pomona.edu/~dkauchak/classes/s16/cs30-s16/lectures/lecture12-NN-basics.pdf"},
    {"path":"files/file2.pdf", "author":"MI Jordan, TM Mitchell - Science, 2015", "title":"Machine learning: Trends, perspectives, and prospects", "link": "https://www.cs.cmu.edu/~tom/pubs/Science-ML-2015.pdf"},
    {"path":"files/file3.pdf", "author":"Prof. Jason Pacheco TA: Enfa Rose George TA: Saiful Islam Salim", "title":"CSC380: Principles of Data Science", "link":"http://www.pachecoj.com/courses/csc380_fall21/lectures/mlintro.pdf"},
    {"path":"files/file4.pdf", "author":"Qifang Bi, Katherine E. Goodman, Joshua Kaminsky, and Justin Lessler*", "title":"What is Machine Learning? A Primer for the Epidemiologist", "link":"https://sph.umsha.ac.ir/uploads/18/2023/May/22/kwz189.pdf"},
    {"path":"files/paper2.pdf", "author":"EB Hunt - 2014", "title":"Artificial intelligence in medicine", "link":"https://pmc.ncbi.nlm.nih.gov/articles/PMC1964229/pdf/15333167.pdf"},
    {"path":"files/paper3.pdf", "author":"W Ertel - 2024", "title":'Introduction to artificial intelligence', "link":"https://thuvienso.hoasen.edu.vn/bitstream/handle/123456789/10507/Contents.pdf?sequence=1&isAllowed=y"}
]




documents = []


def extract_text_frompdf(file_path):

    text = ""

    with fitz.open(file_path) as pdf:

        for page_num in range(pdf.page_count):

            page = pdf[page_num]

            text += page.get_text()

    return text


for file_path in file_paths:
    text = extract_text_frompdf(file_path['path'])

    documents.append({
        "text":text,
        "author":file_path['author'],
        "title": file_path['title'],
        "link": file_path['link']
    })



#preprocess

stop_words = list(stopwords.words("english"))
def preprocess(doc):

    tokens = word_tokenize(doc.lower())

    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]


    lematizer = WordNetLemmatizer()

    tokens = [lematizer.lemmatize(word) for word in tokens]


    return " ".join(tokens)


articles = [preprocess(doc['text']) for doc in documents]



#vectorization

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(articles)

#cosine similartiy




#search engine

def search(query):

    query = preprocess(query)

    query_tfidf = vectorizer.transform([query])

    sim_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    similar_indices = sim_scores.argsort()[::-1]

    return [(index , sim_scores[index]) for index in similar_indices if sim_scores[index] > 0 ]



@app.route("/", methods=['GET', 'POST'])
def home():


    form = SearchForm()

    query = "machine Learning"

    if form.validate_on_submit():


        query = form.query.data 

    result_text = []

    result = search(query)
    result_text = [
        {
            "title": documents[index]["title"],
            "author": documents[index]["author"],
            "link": documents[index]["link"],
            "filename": documents[index]["link"].split('/')[-1],
            "text": documents[index]["text"][:200],  # Show a snippet of text
            "score": score
        }
        for index, score in result
    ]
    return render_template('index.html', result=result_text, form=form)

if __name__  == "__main__":
    app.run(debug=True)