

def jsonl_adapter(data: dict, path: str, id_in_file: int | str):
    '''
    change jsonl keys
    
    data:jsonl sample
    path:jsonl path
    id_in_file: jsonl index
    '''   
    
    return {
        "text": data.pop("html",""),
        "id": id_in_file,
        "media": data.pop("media", ""),
        "metadata": {"title": data.pop("title","")}
    }
# def jsonl_adapter(data: dict, path: str, id_in_file: int | str):
#     return {
#         "text": data.pop("html",""),
#         "id": id_in_file,
#         "metadata": {"title": data.pop("title","")}
#     }