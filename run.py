from data_process.filter import ULambdaFilter,text_filter
from data_process.deduplication import Doc_Dedup,Sentence_Dedup,Tokens_Dedup
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.extractors import Trafilatura
from data_process.load import jsonl_adapter
from typing import Dict,Union
import argparse
from  argparse import Namespace


def process(args:Namespace):
    pipeline = [
        JsonlReader(data_folder=args.data_folder,text_key=args.text_key,id_key=args.id_key,
                    recursive=args.recursive,adapter=jsonl_adapter),
        ULambdaFilter(filter_func=text_filter,stopwords_path=args.stopwords_path,sensitivewords_path=args.sensitivewords_path,
                      delwords_path=args.delwords_path,n_delwords_Threshold=args.n_delwords_Threshold,
                      use_tokenizer=args.use_tokenizer,model_name=args.model_name),
        Doc_Dedup(embeddingmodel=args.embeddingmodel,doc_dedup_threshold=args.doc_dedup_threshold,
                    cuda_index=args.cuda_index,k_retriever=args.k_retriever),
        Sentence_Dedup(embeddingmodel=args.embeddingmodel,sentence_dedup_threshold=args.sentence_dedup_threshold,                    
                    cuda_index=args.cuda_index,k_retriever=args.k_retriever),
        Tokens_Dedup(embeddingmodel=args.embeddingmodel,tokens_dedup_threshold=args.tokens_dedup_threshold,                    
                    cuda_index=args.cuda_index,k_retriever=args.k_retriever,dedup_use_tokenizer=args.dedup_use_tokenizer,
                    tokens_dedup_model_name=args.tokens_dedup_model_name),
        JsonlWriter(output_folder=args.output_folder,output_filename=args.output_filename),

    ]
    executer = LocalPipelineExecutor(pipeline=pipeline, workers=1, tasks=1)
    print(executer.run())

def main():
    parser = argparse.ArgumentParser(description='jsonl data filter and deduplication')
    #load
    jsonlload = parser.add_argument_group(title='jsonlload',description='load jsonl folder path not file')    
    jsonlload.add_argument('--data_folder',type=str,default='data',help='jsonl data file path')
    jsonlload.add_argument('--text_key',type=str,default='text',help='jsonl text key')
    jsonlload.add_argument('--id_key',type=str,default='id',help='jsonl id')
    jsonlload.add_argument('--metadata',type=dict,help='jsonl default metadata')
    jsonlload.add_argument('--recursive',type=bool,default=True,help='Whether it is recursion')
    # filter
    filter_arg = parser.add_argument_group(title='filter',description='filter data')
    filter_arg.add_argument('--stopwords_path',type=str,default='config/stopwords.txt',help='stopwords path')
    filter_arg.add_argument('--sensitivewords_path',type=str,default='config/sensitivewords.txt',help='sensitivewords path')
    filter_arg.add_argument('--delwords_path',type=str,default='config/delwords.txt',help='delwords path')
    filter_arg.add_argument('--n_delwords_Threshold',type=int,default=200,help='document delwords threshold')
    filter_arg.add_argument('--use_tokenizer',type=bool,default=False,help='Whether use LLM token split text')
    filter_arg.add_argument('--model_name',type=str,default=None,help='if use_tokenizer is true,set model local path or huggingface model id')
    #deduplication
    dedup_arg = parser.add_argument_group(title='deduplication',description='deduplication data')
    dedup_arg.add_argument('--embeddingmodel',type=str,default='/data/sonald/xiao/常用/models/bce-embedding-base_v1',help='embedding model local path or huggingface embedding model id')
    dedup_arg.add_argument('--doc_dedup_threshold',type=int,default=0.8,help='document doc deduplication threshold')
    dedup_arg.add_argument('--sentence_dedup_threshold',type=int,default=0.6,help='sentence sentence deduplication threshold')
    dedup_arg.add_argument('--tokens_dedup_threshold',type=int,default=0.5,help='token token deduplication threshold')
    dedup_arg.add_argument('--dedup_use_tokenizer',type=bool,default=False,help='whether use llm tokenizer split token')
    dedup_arg.add_argument('--tokens_dedup_model_name',type=str,default=None,help='if dedup_use_tokenizer is true,set model local path or huggingface model id')
    dedup_arg.add_argument('--cuda_index',type=int,default=0,help='cuda index')
    dedup_arg.add_argument('--k_retriever',type=int,default=10,help='Vision retrieval returns the number of samples')
    #save
    save_arg = parser.add_argument_group(title='save',description='save data')
    save_arg.add_argument('--output_folder',type=str,default='output',help='save folder path')
    save_arg.add_argument('--output_filename',type=str,default='filter_and_dedup',help='save file name')
    
    args = parser.parse_args()
    process(args)
    
    
if __name__ == '__main__':
    main()