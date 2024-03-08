# filter
from datatrove.pipeline.filters import LambdaFilter
from datatrove.data import Document,DocumentsPipeline
from datatrove.utils.typeshelper import StatHints
from transformers import AutoTokenizer
from datatrove.data import Document
from langdetect import detect
from typing import Callable
import jionlp as jio
import jieba
from jieba import posseg
from jieba import analyse
import re
import os

'''
@dataclass
class Document:
    text: str
    id: str
    media: list[Media] = field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
'''
       
#隐私删除
def delete_privacy (text:str) -> str:
    #人名
    words_and_flags = posseg.cut(text)
    text = ''.join(words if flags != 'nr' else '' for words,flags in words_and_flags)
    #qq号
    text = jio.remove_qq(text)
    #身份证号
    text = jio.remove_id_card(text)
    #IP
    text = jio.remove_ip_address(text)
    text = jio.clean_text(text) #清洗文本，包括去除html标签、去除异常字符、去除冗余字符、去除括号补充内容、去除URL、去除E-mail、去除电话号码，将全角字母数字空格替换为半角。
    
    return text      

# 过滤敏感词（政治、广告类）
def filter_Sensitive (doc:Document,sensitivewords_path:str):
    return filter_empty_doc(doc,sensitivewords_path,n_delwords_Threshold=0,word_list=None)

#删除停用词
def filter_stopwords(doc:Document,stopwords_path,use_tokenizer,tokenizer):

    if use_tokenizer:
        with open(stopwords_path,'r',encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f.readlines())
        tokenizer.add_tokens(stopwords)
        word_list = tokenizer.token(doc.text)
        doc.text = ''.join([token for token in word_list if token not in stopwords])
        return doc,word_list
    else:
        try:
            analyse.set_stop_words(stopwords_path)
            word_list = jieba.lcut(doc.text)
            doc.text = ''.join(word_list)
            return doc,word_list
        except:
            with open(stopwords_path,'r',encoding='utf-8') as f:
                stopwords = set(line.strip() for line in f.readlines())
            doc.text = ' '.join([word for word in doc.text.split('\n') if word not in stopwords])
            return doc

#过滤无意义文档
def filter_empty_doc(doc:Document,delwords_path:str,n_delwords_Threshold:str,word_list:str=None):
    # if os.path.exists(delwords_path):
    with open(delwords_path,'r',encoding='utf-8') as f:
        delwords = set(line.strip() for line in f.readlines())
    if any(jump_word in doc.text for jump_word in delwords):
        doc.text = ''
    if word_list != None and len(word_list) <= n_delwords_Threshold:
        doc.text = ''           
    return doc 

def text_filter(doc:Document,stopwords_path:str,sensitivewords_path:str,delwords_path:str,n_delwords_Threshold:int,use_tokenizer:bool,tokenizer=None):           
    doc.text = delete_privacy(doc.text)
    result = filter_stopwords(doc,stopwords_path,use_tokenizer,tokenizer)
    if isinstance(result,tuple):
        doc,word_list = result   
        doc = filter_empty_doc(doc,delwords_path,n_delwords_Threshold,word_list)     
    else:
        doc = result
        doc = filter_empty_doc(doc,delwords_path,n_delwords_Threshold)  
    doc = filter_Sensitive(doc,sensitivewords_path)
    return doc if doc.text else None
        
class ULambdaFilter(LambdaFilter):
    def __init__(self,filter_func:Callable[[Document],bool],stopwords_path:str=None,sensitivewords_path:str = None,
                 delwords_path:str=None,n_delwords_Threshold:int=0,use_tokenizer:bool=False,model_name:str=None):
        super().__init__(filter_function=filter_func)
        self.stopwords_path = stopwords_path
        self.delwords_path = delwords_path
        self.n_delwords_Threshold = n_delwords_Threshold
        self.use_tokenizer = use_tokenizer
        self.model_name = model_name         
        self.sensitivewords_path = sensitivewords_path
        # self.filter_func = filter_func
        
    def get_filter_result(self,res):
        result, reason = res, None
        if isinstance(result, tuple):
            result, reason = res
        return result, reason
    
    def filter(self, doc: Document) :
        if self.use_tokenizer:
            try:
                if os.path.exists(self.model_name) or len(self.model_name.split('/')) == 2:
                    tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name,trust_remote_code=True)                   
            except Exception as e:
                self.use_tokenizer = False
                tokenizer = None
                print(f"{self.tokenizer_name=} is not ture model's tokenize json path or huggingface model id! now use jieba")              
        else:
            tokenizer = None
        return self.filter_function(doc,self.stopwords_path,self.sensitivewords_path,self.delwords_path,self.n_delwords_Threshold,self.use_tokenizer,tokenizer)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        # with self.exclusion_writer if self.exclusion_writer else contextlib.nullcontext() as writer:
        for doc in data:
            with self.stats.time_stats:
                self.stat_update(StatHints.total)
                doc = self.filter(doc)
                if doc:
                    self.stat_update(StatHints.forwarded)
                    self.update_doc_stats(doc)
                    yield doc
                else:
                    self.stat_update(StatHints.dropped)
                    continue
            
            
            # with self.track_time():
            #     filter_result, reason = self.get_filter_result(self.filter(doc))
            #     if filter_result:
            #         self.stat_update(StatHints.forwarded)
            #         self.update_doc_stats(doc)
            #     else:
            #         self.stat_update(StatHints.dropped)
            #         if reason:
            #             self.stat_update(f"dropped_{reason}")
            #         if self.exclusion_writer:
            #             if reason:
            #                 doc.metadata["filter_reason"] = reason
            #             writer.write(doc, rank)
            #         continue
            # yield doc
    