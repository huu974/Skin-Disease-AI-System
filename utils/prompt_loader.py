"""
从配置文件中读取不同场景的提示词文件路径，并加载对应的提示词内容
当前保留：RAG提示词加载
"""

from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger import logger


#加载RAG总结提示词
def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])

    except KeyError as e:
        logger.error(f'[load_rag_prompts]在yaml中未找到rag_summarize_prompt_path配置项')
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()

    except Exception as e:
        logger.error(f'[load_rag_prompts]解析RAG提示词出错,{str(e)}')
        raise e


if __name__ == '__main__':
    print(load_rag_prompts())












