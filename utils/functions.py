import pandas as pd
import numpy as np
import spacy
from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt
import re
from pyvis.network import Network


def ner(book):
    """Function to perform Named Entity Recognition (NER) on a given text file
        This function uses the spaCy library to identify named entities in the text."""
    NER = spacy.load("en_core_web_sm")
    with open(book, 'r', encoding='utf-8') as f:
        book_text = f.read()
        book_doc = NER(book_text)
    return book_doc

def get_entities(book_doc):
    sent_entity_df = []
    for sent in book_doc.sents:
        entity_list = [ent.text for ent in sent.ents]
        sent_entity_df.append({
            'sentence': sent,
            "entities": entity_list
        })
    sent_entity_df = pd.DataFrame(sent_entity_df)
    return sent_entity_df



def filter_entities(ent_list, character_df):
    """
    Filter entities based on character names.
    """
    mapping = {
    "Dumbledore": "Albus Dumbledore",
    "Malfoy": "Draco Malfoy",
    "Weasley": "Ronald Weasley",
    "Snape": "Severus Snape",
}

    filtered_entities = []
    
    for ent in ent_list:
        matching_titles = []

        if ent in character_df['title'].values:
            matching_titles.append(ent)
            
        elif ent in character_df['character_firstname'].values:
            matching_titles = character_df[character_df['character_firstname'] == ent]['title'].tolist()
            
        elif ent in character_df['character_othername'].values:
            matches = character_df[character_df['character_othername'] == ent]['title'].tolist()
            matching_titles = mapping.get(ent, matches)
                
        else:
            continue
        if matching_titles:
            filtered_entities.append(matching_titles[0] if isinstance(matching_titles, list) else matching_titles)

    return filtered_entities


def get_filtered_entities(sent_entity_df, char_df):
    sent_entity_df['character_entities'] = sent_entity_df['entities'].apply(lambda x: filter_entities(x, char_df))
    #Filter out sentences with no character entities
    sent_entity_df_filtered = sent_entity_df[sent_entity_df['character_entities'].apply(lambda x: len(x) > 0)]
    return sent_entity_df_filtered


def extract_relationships(sent_entity_df_filtered):
    #Extract Relationships

    relationships = []
    for i in range(sent_entity_df_filtered.index[-1]):
        end_i = min(i+5, sent_entity_df_filtered.index[-1])
        char_list = sum((sent_entity_df_filtered.loc[i:end_i].character_entities), [])
        
        #remove duplicated characters that are next to each other 
        char_unique = [char_list[i] for i in range(len(char_list)) if i == 0 or char_list[i] != char_list[i-1]]
        if len(char_unique) < 2:
            continue
        for idx, a in enumerate(char_unique[:-1]):
            b = char_unique[idx+1]
            relationships.append({'source': a, 'target': b})
            
    relationship_df = pd.DataFrame(relationships)
    #sort the cases with a->b and b->a
    relationship_df = pd.DataFrame(np.sort(relationship_df.values, axis=1), columns = relationship_df.columns)
    #Aggregate relationships into a weight column 
    relationship_df['weight'] = 1
    relationship_df = relationship_df.groupby(["source", "target"], sort=False, as_index=False).sum()
    return relationship_df


def create_graph(relationship_df):
    G = nx.from_pandas_edgelist(relationship_df, 
                                source="source", 
                                target="target", 
                                edge_attr="weight")
    return G
