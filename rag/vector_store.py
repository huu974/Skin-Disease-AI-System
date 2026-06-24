"""
管理chroma向量数据库，实现文档加载，文本分块，向量化存储和检索功能，支持MD5去重和多种文件格式
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger import logger
import os

class VectorStoreService:
    def __init__(self):
        #创建chroma向量数据库实例
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],  #表名
            embedding_function=embed_model,                  #嵌入模型
            persist_directory=chroma_conf["persist_directory"]  #数据库保存路径
        )

        #创建递归文本分割器，用于将长文档分割成短片段
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf['chunk_size'],       #每个片段最大字符数
            chunk_overlap=chroma_conf['chunk_overlap'], #相邻片段重叠字符数
            separators=chroma_conf['separators'],       #分割符
            length_function=len                         #计算文本长度的函数
        )


    #获取检索器
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf['k']})



    #加载文档到向量存储的方法
    def load_document(self):
        #检查md5值
        def check_md5_hex(md5_for_check):
            if not os.path.exists(get_abs_path(chroma_conf['md5_hex_store'])):
                # 创建md5文件
                open(get_abs_path(chroma_conf['md5_hex_store']), 'w', encoding='utf-8').close()
                return False                #没处理过

            with open(get_abs_path(chroma_conf["md5_hex_store"]),"r",encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True         #处理过

                return False  #没处理过




        #保存md5值
        def save_md5_hex(md5_for_check):
            with open(get_abs_path(chroma_conf["md5_hex_store"]),"a",encoding='utf-8') as f:
                f.write(md5_for_check + "\n")



        #获取文档
        def get_file_documents(read_path):
            if read_path.endswith('txt'):
                return txt_loader(read_path)


            elif read_path.endswith('pdf'):
                return pdf_loader(read_path)


            return []


        allowed_files_path = listdir_with_allowed_type(
            get_abs_path(chroma_conf['data_path']),
            tuple(chroma_conf['allow_knowledge_file_type'])
        )


        for path in allowed_files_path:
            #获取文件md5值
            md5_hex = get_file_md5_hex(path)


            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]文件{path}的md5值已存在，跳过")
                continue

            try:
                documents = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}分片后没有有效内容，跳过")
                    continue

                split_document = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效内容，跳过")
                    continue

                #将内容存入向量库
                self.vector_store.add_documents(split_document)



                #记录这个以处理好的md5值，避免重复处理
                save_md5_hex(md5_hex)


                logger.info(f"[加载知识库]{path}完成")


            except Exception as e:
                logger.error(f"[加载知识库]{path}失败")
                logger.error(e)


if __name__ == '__main__':
    service = VectorStoreService()
    service.load_document()
    retriver = service.get_retriever()


    res = retriver.invoke("光化性角化病和基底细胞癌")
    for r in res:
        print(r.page_content)
