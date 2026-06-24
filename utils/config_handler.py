"""
配置文件加载器
"""

import yaml
from utils.path_tool import get_abs_path



#加载yaml文件的内容，将其转换为字典或列表，让代码可以直接使用这些文件
def load_model_config(config_path:str=get_abs_path("config/model.yml"),encoding:str="utf-8"):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)



def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)



def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)



def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_test_evaluate_config(config_path: str=get_abs_path("config/test_evaluate.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


model_conf = load_model_config()
prompts_conf = load_prompts_config()
rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
test_evaluate_conf = load_test_evaluate_config()


if __name__ == '__main__':
    print(model_conf)   #{'model': 'name:efficientnet-b3 num_classes:23', 'save_path': 'pretrained/'}
    print(model_conf["num_classes"])    #23




































