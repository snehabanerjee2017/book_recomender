import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-available.png",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator="\n") # separate titles so should not have any overlap. 
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(
    documents,
    OpenAIEmbeddings())


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
        ) -> pd.DataFrame:
    
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    books_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category!="All":
        books_recs = books_recs[books_recs["simple_categories"] == category][:final_top_k]
    else:
        books_recs = books_recs.head(final_top_k)

    if tone == "Happy":
        books_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        books_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Sad":
        books_recs.sort_values(by="sadness", ascending=False, inplace=True)
    elif tone == "Angry":
        books_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Fearful":
        books_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Disgusted":
        books_recs.sort_values(by="disgust", ascending=False, inplace=True)

    return books_recs


def recommend_books(
        query: str,
        category: str,
        tone: str ,
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_desc = " ".join(truncated_desc_split[:30]) + "..." if len(truncated_desc_split) > 30 else description

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        
        caption = f"{row['title']} by {authors_str} : {truncated_desc}"
        results.append((row["large_thumbnail"], caption))
        
    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Sad", "Angry", "Fearful", "Disgust"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book",
            placeholder="A story about forgiveness")
        category_dropdown=gr.Dropdown(
                choices=categories,
                label = "Select a category",
                value = "All",
            )
        tone_dropdown=gr.Dropdown(
                choices=tones,
                label = "Select a tone",
                value = "All",
            )
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Book Recommendations", columns=8, rows=2)

    submit_button.click(
        fn = recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown], 
        outputs=output
    )  

if __name__ == "__main__":
    dashboard.launch()


    

    

    
     
        
    