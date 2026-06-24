"""
治疗建议Agent
负责根据诊断结果生成治疗建议和护理方案
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from model.factory import chat_model
from rag.enhanced_rag import EnhancedRAGService
from utils.logger import logger
import json



#治疗建议Agent
class TreatmentAdviceAgent:
    def __init__(self):
        self.rag = EnhancedRAGService()
        self.system_prompt = self._load_system_prompt()
    
    def _load_system_prompt(self) -> str:
        return """你是一名专业的皮肤科医生助手，擅长提供治疗建议和护理方案。

你的职责：
1. 根据诊断结果，提供针对性的治疗建议
2. 提供日常护理建议和预防措施
3. 说明药物使用方法和注意事项
4. 给出就医建议和复查提醒

注意：
- 只提供参考建议，不替代医生诊断
- 严重情况建议立即就医
- 药物使用需遵医嘱
- 输出回答时不要使用###或其他markdown格式符号，直接用自然段落描述
- 保持回答简洁专业"""



    #生成治疗方案
    def generate_treatment_plan(self, diagnosis_info: dict, patient_info: dict = None) -> dict:
        disease = diagnosis_info.get("disease_name", "")
        symptoms = diagnosis_info.get("user_symptoms", "")
        
        query = disease
        if symptoms:
            query += " " + symptoms
        
        rag_result = self.rag.rag_retrieve(query)
        
        # 检查RAG结果是否有效
        rag_answer = rag_result.get("answer", "")
        if not rag_answer or "未找到相关知识" in rag_answer:
            # 没有检索到知识时，直接用通用治疗建议
            default_treatment = self._generate_default_treatment(disease, symptoms)
            return {
                "treatment_plan": default_treatment,
                "rag_result": {},
                "sources": {}
            }
        
        prompt = f"""请为以下情况提供治疗建议：

疾病诊断：{disease}

患者症状：{symptoms}

患者信息：{json.dumps(patient_info, ensure_ascii=False) if patient_info else '未提供'}

参考知识：{rag_answer[:800]}

请提供简洁的治疗建议，包括治疗方案、日常护理、饮食注意事项、预警信号和复查建议。
注意：不要使用任何列表格式（如1. 2. 或-）或emoji符号，用连贯的自然段落描述。
"""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = chat_model.invoke(messages)
        
        return {
            "treatment_plan": response.content,
            "rag_result": rag_result,
            "sources": rag_result.get("source_tracing", {})
        }
    
    
    def _generate_default_treatment(self, disease: str, symptoms: str) -> str:
        """生成默认治疗建议（当没有检索到知识时）"""
        base_info = f"根据您提供的诊断（{disease}），"
        
        if "血管炎" in disease:
            return base_info + """建议如下：治疗方面，医生可能会根据病情严重程度使用糖皮质激素（如泼尼松）来控制炎症，必要时可能联合使用免疫抑制剂。日常护理方面，请注意休息、避免感染、保持皮肤清洁、戒烟。有皮损的部位要避免搔抓，防止继发感染。如果出现新发皮疹、呼吸困难、血尿、视力变化或肢体麻木等异常情况，请立即就医。定期复查血常规、尿常规和炎症指标有助于监测病情变化。具体用药请遵医嘱。"""
        else:
            return base_info + "建议您尽快到正规医院皮肤科或相应专科就诊，进行详细检查和规范化治疗。日常注意休息、保持皮肤清洁、避免刺激。"



    #获取药物指南
    def get_medication_guide(self, disease: str) -> dict:
        query = f"{disease} 药物治疗"
        rag_result = self.rag.rag_retrieve(query)
        
        return {
            "guide": rag_result.get("answer", ""),
            "sources": rag_result.get("source_tracing", {})
        }



    #获取护理指南
    def get_nursing_guide(self, disease: str) -> dict:
        query = f"{disease} 护理"
        rag_result = self.rag.rag_retrieve(query)
        
        return {
            "guide": rag_result.get("answer", ""),
            "sources": rag_result.get("source_tracing", {})
        }



    #获取预防建议
    def get_prevention_tips(self, disease: str) -> dict:
        query = f"{disease} 预防"
        rag_result = self.rag.rag_retrieve(query)
        
        return {
            "tips": rag_result.get("answer", ""),
            "sources": rag_result.get("source_tracing", {})
        }



    #完整分析流程
    def analyze(self, diagnosis_info: dict, patient_info: dict = None) -> dict:
        treatment = self.generate_treatment_plan(diagnosis_info, patient_info)
        
        return treatment



    #验证治疗方案安全性
    def validate_treatment_safety(self, treatment_plan: str, patient_info: dict = None) -> dict:
        validation = {
            "safe": True,
            "warnings": [],
            "contraindications": []
        }
        
        if patient_info:
            age = patient_info.get("age")
            if age and (age < 18 or age > 65):
                validation["warnings"].append("特殊人群用药需特别注意")
            
            allergies = patient_info.get("allergies", [])
            if allergies:
                validation["warnings"].append(f"注意过敏史：{', '.join(allergies)}")
            
            pregnancy = patient_info.get("pregnant", False)
            if pregnancy:
                validation["contraindications"].append("孕妇用药需特别谨慎")
            
            conditions = patient_info.get("medical_conditions", [])
            if conditions:
                validation["warnings"].append(f"既往病史：{', '.join(conditions)}")
        
        dangerous_keywords = ["堕胎", "流产", "致癌", "致畸"]
        for kw in dangerous_keywords:
            if kw in treatment_plan:
                validation["warnings"].append(f"涉及关键词：{kw}，请咨询专业医生")
        
        return validation



#药物指南
class MedicineGuide:
    COMMON_MEDICINES = {
        "痤疮": {
            "外用": ["阿达帕林凝胶", "过氧化苯甲酰", "夫西地酸乳膏"],
            "口服": ["多西环素", "米诺环素", "异维A酸"]
        },
        "湿疹": {
            "外用": ["氢化可的松", "地奈德乳膏", "他克莫司软膏"],
            "口服": ["氯雷他定", "西替利嗪"]
        },
        "银屑病": {
            "外用": ["卡泊三醇软膏", "卤米松乳膏"],
            "口服": ["阿维A", "甲氨蝶呤", "生物制剂"]
        },
        "荨麻疹": {
            "外用": ["炉甘石洗剂"],
            "口服": ["氯雷他定", "西替利嗪", "左西替利嗪"]
        },
        "真菌感染": {
            "外用": ["酮康唑乳膏", "特比萘芬乳膏", "硝酸咪康唑"],
            "口服": ["伊曲康唑", "特比萘芬", "氟康唑"]
        },
        "疱疹": {
            "外用": ["阿昔洛韦乳膏", "喷昔洛韦乳膏"],
            "口服": ["阿昔洛韦", "伐昔洛韦", "泛昔洛韦"]
        }
    }


    #获取疾病的常用药物
    @classmethod
    def get_medicines(cls, disease: str) -> dict:
        for disease_name, medicines in cls.COMMON_MEDICINES.items():
            if disease_name in disease:
                return medicines
        return {}



    #获取药物安全须知
    @classmethod
    def get_safety_notice(cls, medicine: str) -> str:
        notices = {
            "异维A酸": "育龄期禁用，怀孕可能致畸",
            "阿维A": "育龄期禁用，怀孕可能致畸",
            "甲氨蝶呤": "需定期监测肝肾功能",
            "米诺环素": "可能导致牙齿黄染，8岁以下禁用",
            "多西环素": "可能导致牙齿黄染，8岁以下禁用",
            "他克莫司": "长期使用可能增加皮肤癌风险",
            "激素类": "长期大面积使用可能导致皮肤萎缩"
        }
        return notices.get(medicine, "请遵医嘱使用")



#紧急情况检查器
class EmergencyChecker:
    EMERGENCY_SIGNALS = [
        ("呼吸困难", "过敏性休克"),
        ("意识模糊", "严重过敏反应"),
        ("高热不退", "严重感染"),
        ("皮疹迅速扩散", "严重过敏反应"),
        ("水疱大面积出现", "严重药物反应"),
        ("皮肤溃烂坏死", "坏死性筋膜炎"),
        ("口腔黏膜受累", "严重皮肤反应"),
        ("全身弥漫性红斑", "中毒性表皮坏死松解症")
    ]


    #检查是否需要紧急就医
    @classmethod
    def check_emergency(cls, symptoms: str) -> dict:
        result = {
            "is_emergency": False,
            "signals_found": [],
            "recommendation": ""
        }
        
        symptoms_lower = symptoms.lower()
        
        for signal, condition in cls.EMERGENCY_SIGNALS:
            if signal in symptoms:
                result["signals_found"].append({
                    "signal": signal,
                    "possible_condition": condition
                })
        
        if result["signals_found"]:
            result["is_emergency"] = True
            result["recommendation"] = "发现紧急信号，请立即就医！"
        else:
            result["recommendation"] = "目前无明显紧急信号，但仍需密切关注症状变化"
        
        return result


if __name__ == "__main__":
    agent = TreatmentAdviceAgent()
    print("治疗建议Agent已初始化")
    print("使用方法：agent.analyze(diagnosis_info, patient_info)")
