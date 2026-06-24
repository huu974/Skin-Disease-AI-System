"""
图像诊断Agent
负责分析皮肤图像，进行疾病分类和检测
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from model.factory import chat_model
from agent.tools.agent_tools import skin_classify
from rag.enhanced_rag import EnhancedRAGService
from utils.logger import logger
import os
from PIL import Image


#图像诊断Agent
class ImageDiagnosisAgent:
    def __init__(self):
        self.rag = EnhancedRAGService()
        self.system_prompt = self._load_system_prompt()
        self._init_models()
    
    def _load_system_prompt(self) -> str:
        return """你是一名专业的皮肤科AI诊断助手，擅长分析皮肤图像。

你的职责：
1. 使用分类模型判断可能的皮肤疾病类型
2. 综合分析图像特征和检测结果
3. 提供专业的诊断建议

注意：
- 只提供辅助诊断参考，不做确定性诊断
- 建议用户进行专业检查确认
- 图像不清晰时建议重新拍摄
- 输出回答时不要使用###或其他markdown格式符号，不要使用列表格式，用自然段落描述"""



    #初始化模型（延迟加载）
    def _init_models(self):
        self.models_loaded = False



    def _ensure_models_loaded(self):
        if not self.models_loaded:
            logger.info("初始化图像诊断模型...")
            self.models_loaded = True



    #执行疾病分类
    def detect_and_classify(self, image_path: str) -> dict:
        self._ensure_models_loaded()
        logger.info(f"[图像诊断] 开始分析图像: {image_path}")
        
        try:
            classify_result = skin_classify.invoke({"image_path": image_path})
            logger.info(f"[分类] 分类完成: {classify_result}")
        except Exception as e:
            logger.error(f"分类失败: {e}")
            classify_result = f"分类失败: {str(e)}"
        
        return {
            "classification": classify_result
        }




    #综合分析：结合图像结果和用户症状
    def analyze_with_context(self, image_path: str, user_symptoms: str = None) -> dict:
        model_results = self.detect_and_classify(image_path)
        disease_name = self._extract_disease_name(model_results["classification"])
        
        query = disease_name
        if user_symptoms:
            query += " " + user_symptoms
        
        rag_result = self.rag.rag_retrieve(query)
        logger.info(f"[图像诊断] RAG检索query: {query}")
        logger.info(f"[图像诊断] RAG检索结果前200字: {str(rag_result.get('answer', ''))[:200]}")
        
        prompt = f"""请基于以下图像分析结果提供专业诊断意见：

图像检测结果：
- 疾病分类：{model_results['classification']}

用户症状描述：{user_symptoms or '未提供'}

参考知识：{rag_result['answer'][:500]}

请提供简洁的诊断意见，包括综合诊断、病情评估和建议。
注意：不要使用列表格式（如1. 2. 或-）或emoji，用连贯的自然段落描述。
"""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = chat_model.invoke(messages)
        
        return {
            "diagnosis": response.content,
            "model_results": model_results,
            "rag_result": rag_result,
            "disease_name": disease_name
        }



    #从分类结果中提取疾病名称
    def _extract_disease_name(self, classification_result: str) -> str:
        if "分类结果" in classification_result:
            parts = classification_result.split("，")
            if parts:
                name = parts[0].replace("分类结果：", "")
                return name
        return classification_result



    #完整的图像分析流程
    def analyze(self, image_path: str, user_symptoms: str = None) -> dict:
        if not os.path.exists(image_path):
            return {"error": f"图像文件不存在: {image_path}"}
        
        result = self.analyze_with_context(image_path, user_symptoms)
        
        return result

#图像预处理器
class ImagePreprocessor:
    @staticmethod
    def validate_image(image_path: str) -> dict:
        """验证图像是否有效"""
        result = {
            "valid": False,
            "message": "",
            "suggestions": []
        }
        
        if not os.path.exists(image_path):
            result["message"] = "文件不存在"
            return result
        
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            if width < 100 or height < 100:
                result["message"] = "图像尺寸过小"
                result["suggestions"].append("请提供更清晰的图像")
                return result
            
            if width > 4096 or height > 4096:
                result["message"] = "图像尺寸过大"
                result["suggestions"].append("图像可能被压缩，影响分析")
            
            if img.mode not in ['RGB', 'L']:
                result["message"] = "图像模式不支持"
                result["suggestions"].append("请转换图像格式")
                return result
            
            result["valid"] = True
            result["message"] = f"图像有效，尺寸: {width}x{height}"
            
        except Exception as e:
            result["message"] = f"无法读取图像: {str(e)}"
        
        return result
    
    @staticmethod
    def suggest_improvements(image_path: str) -> list[str]:
        """建议图像改进"""
        suggestions = []
        
        precheck = ImagePreprocessor.validate_image(image_path)
        if not precheck["valid"]:
            return precheck["suggestions"]
        
        suggestions.extend(precheck.get("suggestions", []))
        
        return suggestions


if __name__ == "__main__":
    agent = ImageDiagnosisAgent()
    print("图像诊断Agent已初始化")
    print("使用方法：agent.analyze('图像路径', '用户症状描述')")
