
import yaml
import os, sys, datetime
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
load_dotenv(dotenv_path='keys.env')
KEYS = {
    'openai': os.getenv('OPENAI_API_KEY')
}
os.environ['OPENAI_API_KEY'] = KEYS['openai']

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import SingleTurnSample 
from ragas.metrics import ResponseRelevancy, Faithfulness, AspectCritic, FactualCorrectness, BleuScore, RougeScore, SemanticSimilarity
from ragas.metrics._string import NonLLMStringSimilarity, DistanceMeasure
import seaborn as sns
import matplotlib.pyplot as plt

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
class Labels(object):
    def __init__(self, labels, list_ds_urls, list_metadata_fields):
        self.labels = labels
        self.list_ds_urls = list_ds_urls
        self.list_metadata_fields = list_metadata_fields

        self._findability_exists = False 
        self._findability_not_exists = False
        for url, md_vals in self.labels.items():
            assert isinstance(md_vals, dict), f"Metadata for {url} is not a dictionary."
            for md_field, md_val in md_vals.items():
                assert isinstance(md_field, str), f"Metadata field {md_field} for {url} is not a string."
                if isinstance(md_val, list):
                    assert len(md_val) == 1, f"Metadata field {md_field} for {url} has multiple values: {md_val}. Expected a single value."
                    assert type(md_val[0]) in [str, int, float], f"Metadata field {md_field} for {url} has an invalid type: {type(md_val[0])}. Expected str, int, or float."
                    self._findability_not_exists = True 
                    assert not self._findability_exists, 'Expected only one of findability or findability_not_exists to be True.'
                elif isinstance(md_val, dict):
                    assert len(md_val) == 2, f"Metadata field {md_field} for {url} has multiple values: {md_val}. Expected a single value."
                    assert all(k in ['text', 'findability'] for k in md_val.keys()), f"Metadata field {md_field} for {url} has unexpected keys: {md_val.keys()}."
                    self._findability_exists = True
                    assert not self._findability_not_exists, 'Expected only one of findability or findability_not_exists to be True.'
                else:
                    raise ValueError(f"Metadata field {md_field} for {url} has an unexpected type: {type(md_val)}. Expected list or dict.")
        if not self._findability_exists and not self._findability_not_exists:
            raise ValueError("Labels must contain either 'findability' or 'findability_not_exists' metadata fields.")
        if self._findability_exists:
            self.findability_exists = True
        else:
            self.findability_exists = False

        self.context_lists = {url: [] for url in self.list_ds_urls}
        for url in self.list_ds_urls:
            for m in list_metadata_fields:
                val = self.get_text(url, m)
                if val is not None:
                    if isinstance(val, str) or isinstance(val, int) or isinstance(val, float):
                        self.context_lists[url].append(f'The {m} of the dataset is: {val}')
                    else:
                        raise ValueError(f"Unexpected type for metadata field {m} in URL {url}: {type(val)}. Expected str or list.")
                
    def get(self, url, field):
        assert url in self.list_ds_urls, f"URL {url} not found in labels."
        assert field in self.list_metadata_fields, f"Field {field} not found in metadata fields." 

        if field in self.labels[url].keys():
            return self.labels[url][field]
        else:
            print(f"Field {field} not found in labels for URL {url}. Available fields: {self.list_metadata_fields}")
            print(f"Available fields for {url}: {self.labels[url].keys()}")
            return [None]
        
    def get_text(self, url, field):
        if self.findability_exists:
            field_annot = self.labels[url][field]['text'] if field in self.labels[url] else None
        else:
            field_annot = self.labels[url][field][0] if field in self.labels[url] else None
    
        if type(field_annot) is datetime.date:
            field_annot = field_annot.strftime('%Y-%m-%d')
        return field_annot

    def get_text_list(self, url, field):
        return [self.get_text(url, field)]

    def get_findability(self, url, field):
        if self.findability_exists:
            return self.labels[url][field]['findability'] if field in self.labels[url] else None
        else:
            return None

    def __getitem__(self, url):
        return self.labels[url]
    def __len__(self):
        return len(self.labels)
    def keys(self):
        return self.labels.keys()
    def values(self):
        return self.labels.values()
    def items(self):
        return self.labels.items()
    def __contains__(self, url):
        return url in self.labels
    def __repr__(self):
        return f"Labels({len(self.labels)} URLs, {len(self.list_metadata_fields)} metadata fields: {self.list_metadata_fields})"

def load_pred_and_annot(fp_pred, fp_annot, keep_annotated_fields_only=False, metadata_format: str = 'cedar'):
    labels_annot = load_yaml(fp_annot)
    labels_pred = load_yaml(fp_pred)
    labels_annot = specify_and_convert_metadata_fields(labels_annot, metadata_format)
    labels_pred = specify_and_convert_metadata_fields(labels_pred, metadata_format)
    assert set(labels_annot.keys()) == set(labels_pred.keys()), f"Keys in annotation and prediction files do not match: {set(labels_annot.keys()).symmetric_difference(set(labels_pred.keys()))}"
    metadata_fields_annot = list(list(labels_annot.values())[0].keys())

    if keep_annotated_fields_only:  # Only keep fields that are present in the annotations
        for url, vals in labels_pred.items():
            for key in list(vals.keys()):
                if key not in metadata_fields_annot:
                    del vals[key]

    for _, vals in labels_pred.items():
        tmp_md_fields_pred = set(list(vals.keys()))
        tmp_md_fields_annot = set(metadata_fields_annot).union(set(['Landing page']))
        assert tmp_md_fields_pred.difference(tmp_md_fields_annot) == set(), f"Prediction fields contain unexpected keys. {tmp_md_fields_pred.difference(tmp_md_fields_annot)}"
    list_ds_urls = list(labels_annot.keys())
    labels_annot = Labels(labels_annot, list_ds_urls, metadata_fields_annot)
    labels_pred = Labels(labels_pred, list_ds_urls, metadata_fields_annot)
    print(f"Loaded {len(labels_annot.labels)} annotations and predictions.")

    for l_name, l in zip(['predicted'], [labels_pred]):
        for url, vals in l.items():
            for meta_key, meta_vals in vals.items():
                if len(meta_vals) > 1:
                    print(f"Multiple values for {l_name}: {meta_key} in {url}: {meta_vals}")
                    
    return labels_annot, labels_pred, (list_ds_urls, metadata_fields_annot)

def specify_and_convert_metadata_fields(labels, metadata_format: str):
    assert metadata_format in ['cedar', 'croissant']
    if metadata_format == 'cedar':
        metadata_fields = ['Access rights',
                            'Data contact point',
                            'Data creator',
                            'Data publisher',
                            'Description',
                            'Distribution access URL',
                            'Distribution byte size',
                            'Distribution format',
                            'Keywords',
                            'License',
                            'Metadata date',
                            'Metadata language',
                            'Resource type',
                            'Responsible organization metadata',
                            'Spatial coverage',
                            'Spatial reference system',
                            'Spatial resolution',
                            'Temporal coverage',
                            'Temporal resolution',
                            'Title',
                            'Unique Identifier']
        metadata_fields_dict = {x: x for x in metadata_fields}
    elif metadata_format == 'croissant':
        metadata_fields_dict = {'creator': 'Data creator',
                                'publisher': 'Data publisher',
                                'dateModified': 'Date last modified',
                                'datePublished': 'Date published',
                                'description': 'Description',
                                'keywords': 'Keywords',
                                'license': 'License',
                                'inLanguage': 'Metadata language',
                                'sameAs': 'Same as',
                                'name': 'Title'}
        ## Also add value to value for annotated data that already uses these fields as keys.
        metadata_fields_dict = {**metadata_fields_dict, **{v: v for v in metadata_fields_dict.values() if v not in metadata_fields_dict.keys()}}
    else:
        raise ValueError(f"Unknown metadata format: {metadata_format}")
    
    new_labels = {}
    for url, vals in labels.items():
        new_vals = {}
        for meta_key, meta_val in vals.items():
            if meta_key in metadata_fields_dict:
                new_vals[metadata_fields_dict[meta_key]] = meta_val
        new_labels[url] = new_vals

    return new_labels