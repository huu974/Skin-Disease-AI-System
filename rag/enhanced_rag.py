"""
增强版RAG服务（简化版，兼容现有环境）
功能：多源检索 -> 重排序 -> 多重过滤 -> 引用标注 -> LLM生成 -> 来源追溯
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from collections import defaultdict


class EnhancedRAGService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.base_retriever = self.vector_store.get_retriever()
        
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()
    
    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain



    # 1.多源检索
    def multi_source_retrieval(self, query: str) -> list:
        results = self.base_retriever.invoke(query)
        return results

    # 2.重排序（简化版：按文档长度排序
    def rerank_documents(self, query: str, documents: list) -> list:
        if not documents:
            return documents
        scored_docs = [(doc, len(doc.page_content)) for doc in documents]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]



    #3. 多重过滤
    def filter_documents(self, documents: list, query: str) -> list:
        if not documents:
            return documents
        
        filtered = []
        seen_content = set()
        
        for doc in documents:
            content = doc.page_content.strip()
            if not content or len(content) < 20:
                continue
            
            if content in seen_content:
                continue
            
            if len(filtered) >= 5:
                break
            
            seen_content.add(content)
            filtered.append(doc)
        
        return filtered



    # 4.引用标注
    def add_citations(self, documents: list) -> list:
        cited_docs = []
        for i, doc in enumerate(documents):
            new_content = f"【参考{i+1}】{doc.page_content}"
            metadata = doc.metadata.copy()
            metadata['citation'] = i + 1
            cited_docs.append(Document(
                page_content=new_content,
                metadata=metadata
            ))
        return cited_docs


    # 5.来源追溯
    def trace_sources(self, documents: list) -> dict:
        source_stats = defaultdict(list)
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            source_stats[source].append({
                'citation': doc.metadata.get('citation', 0),        #引用编号
                'preview': doc.page_content[:100] + '...'           #内容预览
            })
        return dict(source_stats)



    # 6.LLM生成
    def generate_response(self, query: str, context_docs: list) -> str:
        context = ""
        seen = set()
        for doc in context_docs:
            # 提取原始内容，移除【参考X】标记
            content = doc.page_content
            # 移除 【参考1】 这样的前缀
            import re
            content_cleaned = re.sub(r'^\[?【?参考\d+】?\]?\s*', '', content)
            
            if content_cleaned in seen:
                continue
            seen.add(content_cleaned)
            context += f"【参考资料{len(seen)}】:{content_cleaned}\n"
        
        try:
            response = self.chain.invoke({
                "input": query,
                "context": context
            })
            return response
        except Exception as e:
            return f"生成回答失败: {str(e)}"



    # 完整RAG流程
    def rag_retrieve(self, query: str) -> dict:
        #1.多元检索
        sources = self.multi_source_retrieval(query)
        #2.重排序
        sources = self.rerank_documents(query, sources)
        #3.多重过滤
        sources = self.filter_documents(sources, query)
        #4.引用标注
        sources = self.add_citations(sources)

        #LLM生成
        response = self.generate_response(query, sources)
        
        return {
            "answer": response,
            "documents": sources,
            #5.来源追溯
            "source_tracing": self.trace_sources(sources)
        }


if __name__ == "__main__":
    rag = EnhancedRAGService()
    result = rag.rag_retrieve("湿疹治疗")
    print("Answer:", result["answer"][:200] if result["answer"] else "No answer")
    print("Docs:", len(result["documents"]))
    print("Done!")
