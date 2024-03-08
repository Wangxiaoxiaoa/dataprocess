from typing import Any
from abc import abstractmethod
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS,Chroma
from langchain_community.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer

from datatrove.io import DataFolderLike,get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.data import Media
from datatrove.pipeline.writers.disk_base import DiskWriter
# from datatrove.pipeline.dedup.utils import DiskWriter
from datatrove.data import Document,DocumentsPipeline
from datatrove.utils.typeshelper import StatHints

from loguru import logger
import contextlib
from dataclasses import field
from dataclasses import dataclass
import gc
import re
import jieba


class VectorDedup(PipelineStep):
    type = "🫂 - DEDUPS"
    name = "→ Vecctor-dedups"
    _requires_dependencies = ["transformers"]
    
    def __init__(
        self,
        embeddingmodel: str,
        cuda_index: int = 0,
        k_retriever: int = 5,
        exclusion_writer: DiskWriter = None,
        
    ):
        """Args:
        output_folder: output folder
        embeddingmodel: embedding model local path or huggingface model id
        use_doc_dedup: whether document deduplication
        use_sentence_dedup: whether sentence deduplication in same document
        use_tokens_dedup: whether tokens deduplication in same document
        cuda_index: use which cuda,start in 0
        """
        super().__init__()
        self.embeddingmodel = embeddingmodel
        self.cuda_index = cuda_index
        self.k_retriever = k_retriever
        self.exclusion_writer = exclusion_writer
        

    def load_embedingmodel(self):
        cuda_index = 'cuda:'+str(self.cuda_index)
        model_kwargs = {'device':cuda_index}
        encode_kwargs = {'batch_size':64,'normalize_embeddings':True,'show_progress_bar':False}
        embed_model = HuggingFaceEmbeddings(
            model_name=self.embeddingmodel,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return embed_model
     
    def generate_documentpipeline(self,docs:list,metadata:dict):
        for idx,sentence in enumerate(docs):
            data_single = Document(text=sentence,metadata=metadata,id=str(idx))
            yield data_single
                
    def collect(self,db):
        del db
        gc.collect()
    
    def lc2d(self,lcdoc:LangchainDocument,id:int) -> Document:
        doc = Document(text=lcdoc.page_content,metadata=lcdoc.metadata,id=str(id))
        return doc   
    
    def d2lc(self,doc:Document) -> LangchainDocument:
        doc =  LangchainDocument(page_content=doc.text,metadata=doc.metadata)        
        return doc  

    def build_retriever(self,doc:Document,embed_model:HuggingFaceEmbeddings,threshold:float):
        db = FAISS.from_documents(doc,embed_model,distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        threshold = threshold
        search_kwargs = {"score_threshold": threshold, "k": self.k_retriever}
        retriever = db.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
        return retriever,db
    
    def doc_dedup(self,doc: Document,embed_model:HuggingFaceEmbeddings):
        pass
    
    # @abstractmethod    
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        pass
    
class Doc_Dedup(VectorDedup):
    type = "※ - DEDUPS"
    name = "→ Doc-dedups"
    def __init__(  
        self,
        embeddingmodel: HuggingFaceEmbeddings,
        cuda_index: int = 0,
        k_retriever: int = 5,
        doc_dedup_threshold: float = 0.9
    ):
        super().__init__(embeddingmodel,cuda_index,k_retriever)
        self.doc_dedup_threshold = doc_dedup_threshold

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:       
        embed_model = self.load_embedingmodel()
        retriever,db = None,None
        for idx, doc in enumerate(data):
            self.stat_update(StatHints.total)
            if idx == 0:
                assert doc.text is not None ,f"{doc.text=} is None,The first document must have text!"
                first_doc = [self.d2lc(doc)]
                retriever,db = self.build_retriever(first_doc,embed_model,self.doc_dedup_threshold)
                self.stat_update(StatHints.forwarded)
            else:
                try:
                    related_passages = retriever.get_relevant_documents(doc.text)
                    if len(related_passages) == 0:
                        lcdoc = self.d2lc(doc)
                        retriever.add_documents([lcdoc])                        
                        self.stat_update(StatHints.forwarded)   
                        self.update_doc_stats(doc)                     
                        yield doc
                        self.update_doc_stats(doc)
                    else:
                        self.stat_update(StatHints.dropped)
                        continue
                except Exception as e:
                    print(f"the{idx=} document error,reason is:{e}") 
                    self.stat_update(StatHints.dropped)
                    continue   
        self.collect(db)
        
            
    
class Sentence_Dedup(VectorDedup):
    type = "※ - DEDUPS"
    name = "→ Sentence-dedups"
    
    def __init__(
        self,
        embeddingmodel: HuggingFaceEmbeddings,
        sentence_dedup_threshold: float = 0.8,
        cuda_index: int = 0,
        k_retriever: int = 5,
        
    ):
        super().__init__(embeddingmodel,cuda_index,k_retriever)
        self.sentence_dedup_threshold = sentence_dedup_threshold       
        

        
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        punctuation = ['。','？','！',';','……','.','?','!',';']
        pattern = '[' + re.escape(''.join(punctuation)) + ']'
        embed_model = self.load_embedingmodel()        
        for doc in data:
            retriever,db = None,None
            self.stat_update(StatHints.total)
            metadata = doc.metadata
            idx = doc.id
            sentences_list = re.split(pattern, doc.text)
            sentences_list = [s.strip() for s in sentences_list if s.strip()]
            single_data: DocumentsPipeline = self.generate_documentpipeline(sentences_list,metadata)
            drop_sentence = list()

            for id, sentence in enumerate(single_data):
                if id == 0:
                    first_sentence = [self.d2lc(sentence)]
                    retriever,db = self.build_retriever(first_sentence,embed_model,self.sentence_dedup_threshold)
                else:
                    try:
                        related_passages = retriever.get_relevant_documents(sentence.text)
                        if len(related_passages) == 0:
                            sentence = self.d2lc(sentence)
                            retriever.add_documents([sentence])                        
                        else:
                            drop_sentence.append(sentence.text)
                    except Exception as e:
                        print(f"the{idx=} sentence error,reason is:{e}") 
                        drop_sentence.append(sentence.text)
                        continue  
            self.collect(db)
            # logger.info(f"Sentence dedup total drop {len(drop_sentence)} sentences")
            # doc.text = ''.join([doc.text.replace(sentence,'') for sentence in drop_sentence])
            for sentence in drop_sentence:
                doc.text = doc.text.replace(sentence,'')
            if doc.text:
                # self.update_doc_stats(doc)
                self.stat_update(StatHints.forwarded)
                self.update_doc_stats(doc)
                yield doc
            else:
                self.stat_update(StatHints.dropped)
                continue          
                                         
               
class Tokens_Dedup(VectorDedup):
    type = "※ - DEDUPS"
    name = "→ Tokens-dedups"
    
    def __init__(
        self,
        embeddingmodel: HuggingFaceEmbeddings,
        tokens_dedup_threshold: float = 0.7 ,       
        cuda_index: int = 0,
        k_retriever: int = 5,      
        dedup_use_tokenizer: bool = True,
        tokens_dedup_model_name: str = None,  
    ):
        super().__init__(embeddingmodel,cuda_index,k_retriever)
        self.tokens_dedup_threshold = tokens_dedup_threshold
        self.dedup_use_tokenizer = dedup_use_tokenizer
        self.tokens_dedup_model_name = tokens_dedup_model_name       
        
        
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        punctuation = ['。','？','！',';','……','.','?','!',';']
        pattern = '[' + re.escape(''.join(punctuation)) + ']'
        embed_model = self.load_embedingmodel()
        for doc in data:
            self.stat_update(StatHints.total)
            metadata = doc.metadata
            sentences_list = re.split(pattern, doc.text)
            sentences_list = [s.strip() for s in sentences_list if s.strip()]
            result_sentences_list = list()
            single_data: DocumentsPipeline = self.generate_documentpipeline(sentences_list,metadata)
            for idx,sentence in enumerate(single_data):                
                retriever,db = None,None
                if self.dedup_use_tokenizer:
                    tokenizer = AutoTokenizer.from_pretrained(self.tokens_dedup_model_name,trust_remote_code=True)
                    tokens_list = tokenizer.tokenize(sentence.text)
                else:
                    tokens_list = jieba.lcut(sentence.text)
                single_sentence: DocumentsPipeline = self.generate_documentpipeline(tokens_list,metadata)
                drop_tokens = list()
                for id,token in enumerate(single_sentence):
                    if id == 0:
                        first_token = [self.d2lc(token)]
                        retriever,db = self.build_retriever(first_token,embed_model,self.tokens_dedup_threshold)
                    else:
                        try:
                            related_tokens = retriever.get_relevant_documents(token.text)
                            if len(related_tokens) == 0:
                                token = self.d2lc(token)
                                retriever.add_documents([token])                        
                            else:
                                drop_tokens.append(token.text)
                        except Exception as e:
                            print(f"the{id=} sentence error,reason is:{e}") 
                            drop_tokens.append(sentence.text)
                            continue                         
                self.collect(db)
                
                # sentence.text = ''.join([sentence.text.replace(token,'') for token in drop_tokens])
                for token in drop_tokens:
                    sentence.text = sentence.text.replace(token,'')
                result_sentences_list.append(sentence.text)
            doc.text = ''.join(result_sentences_list)
        
            if doc.text:
                # self.update_doc_stats(doc)
                self.stat_update(StatHints.forwarded)
                self.update_doc_stats(doc)
                yield doc
            else:
                self.stat_update(StatHints.dropped)
                continue  
                        
                    
                    
                    
                
                
                
            
            
            
