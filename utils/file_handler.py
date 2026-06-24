"""
提供文件MD5计算，目录遍历，文档加载，支持PDF和TXT文件的处理，用于知识库构建和文件去重
"""

import os
import hashlib
from utils.logger import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


# 文件MD5计算，用于文件去重
def get_file_md5_hex(filepath):
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return

    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return

    #创建md5计算对象
    md5_obj = hashlib.md5()


    chunk_size = 4096   #4KB分片，避免内存占用过大
    try:
        with open(filepath,"rb") as f:
            chunk = f.read(chunk_size)
            while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
                #获取md5的16进制表示
                md5_hex = md5_obj.hexdigest()
                return md5_hex

    except Exception as e:
        logger.error(f'计算文件{filepath}的md5值时出错: {str(e)}')
        return None



#获取该目录下的所有文件类型路径(元组)
def listdir_with_allowed_type(path,allowed_types:tuple[str]):
    files = []

    if not os.path.isdir(path):
        logger(f'[listdir_with_allowed_type]]{path}不是文件夹')
        return allowed_types


    #遍历目录中的所有文件和子目录
    for f in os.listdir(path):
        #检查文件名是否以允许的类型结尾
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)


#加载PDF文件
def pdf_loader(file_path,password=None) -> list[Document]:
    return PyPDFLoader(file_path,password).load()



#加载TXT文件
def txt_loader(file_path) -> list[Document]:
    return TextLoader(file_path, encoding="utf-8").load()


















