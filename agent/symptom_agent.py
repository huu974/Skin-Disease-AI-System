"""
症状分析Agent
负责分析用户描述的症状，提供初步诊断建议
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from model.factory import chat_model
from rag.enhanced_rag import EnhancedRAGService
import json
import re



#症状分析Agent
class SymptomAnalyzerAgent:
    def __init__(self):
        self.rag = EnhancedRAGService()
        self.system_prompt = self._load_system_prompt()
        self.symptom_patterns = self._init_symptom_patterns()
    
    def _load_system_prompt(self) -> str:
        return """你是一名专业的皮肤科医生助手，擅长分析用户描述的皮肤症状。

你的职责：
1. 仔细分析用户描述的症状（部位、形态、感觉、时间等）
2. 从医学角度识别可能的皮肤问题类型
3. 提供初步的诊断建议和需要进一步确认的问题
4. 根据症状严重程度建议是否需要紧急就医

分析维度：
- 皮损形态：红斑、丘疹、水疱、脓疱、糜烂、溃疡、疤痕等
- 分布部位：面部、躯干、四肢、隐私部位等
- 主观症状：瘙痒、疼痛、灼热、麻木等
- 持续时间：急性（<2周）、亚急性、慢性
- 诱发因素：食物、药物、接触物、环境等

重要提示：
- 只提供初步分析，不做确定性诊断
- 建议用户进行专业检查确认
- 症状严重时建议立即就医
- 输出回答时不要使用###或其他markdown格式符号，直接用自然段落描述
- 保持回答简洁专业"""
    
    def _init_symptom_patterns(self) -> dict:
        return {
            "瘙痒": ["痒", "瘙痒", "发痒", "痒感", "蛰"],
            "疼痛": ["疼", "痛", "酸痛", "刺痛", "灼痛", "钝痛"],
            "红斑": ["红", "发红", "红斑", "红晕"],
            "丘疹": ["疙瘩", "痘痘", "丘疹", "小颗粒"],
            "水疱": ["水疱", "水泡", "透明疱", "脓疱"],
            "糜烂": ["糜烂", "渗出", "流水", "溃烂"],
            "脱屑": ["脱屑", "脱皮", "掉皮", "鳞屑"],
            "色素": ["色素", "发黑", "发白", "变色"],
            "肿胀": ["肿", "肿胀", "鼓起", "变大"]
        }



    #从用户文本中提取症状关键词
    def extract_symptoms(self, user_text: str) -> dict:
        symptoms = {
            "mentioned": [],
            "location": [],
            "duration": None,
            "severity": None
        }
        
        text = user_text.lower()
        
        for symptom, keywords in self.symptom_patterns.items():
            for keyword in keywords:
                if keyword in text:
                    symptoms["mentioned"].append(symptom)
                    break
        
        locations = ["脸", "面部", "额头", "眼睛", "鼻子", "嘴巴", "唇", "下巴",
                     "颈", "脖子", "胸", "胸部", "背", "背部", "肚子", "腹",
                     "手", "脚", "腿", "胳膊", "腋下", "腹股沟", "私处", "阴部"]
        for loc in locations:
            if loc in text:
                symptoms["location"].append(loc)
        
        duration_match = re.search(r'(\d+)(天|周|月|年)', text)
        if duration_match:
            symptoms["duration"] = f"{duration_match.group(1)}{duration_match.group(2)}"
        
        if any(w in text for w in ["很痒", "非常痒", "特别痒", "剧烈", "严重", "无法忍受"]):
            symptoms["severity"] = "严重"
        elif any(w in text for w in ["有点", "轻微", "不太", "一般"]):
            symptoms["severity"] = "轻微"
        
        return symptoms



    #分析用户症状
    def analyze(self, user_query: str) -> dict:
        extracted = self.extract_symptoms(user_query)
        
        # 检查提取到的信息是否足够
        mentioned = extracted.get("mentioned", [])
        location = extracted.get("location", [])
        duration = extracted.get("duration")
        
        # 如果信息太少，主动追问
        if not mentioned and not location:
            # 信息不足，生成追问提示
            prompt = f"""用户只提供了非常有限的信息："{user_query}"

你是一名皮肤科医生，需要获取更多症状信息才能进行分析。

请生成3个简短的问题来询问用户，获取关键信息：
1. 具体症状（痒？疼？红斑？脱皮？）
2. 具体部位（脸上？身上？手上？）
3. 持续时间（多久了？）

注意：
- 不要用列表格式，直接用自然段落
- 问题要简洁，一次问1-2个
- 语气友好，像医生问诊"""

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = chat_model.invoke(messages)
            
            return {
                "analysis": response.content,
                "extracted_symptoms": extracted,
                "rag_result": {"answer": "信息不足，无法检索知识库"},
                "need_image": True,
                "need_more_info": True
            }
        
        query_for_rag = user_query
        if extracted["mentioned"]:
            query_for_rag += " " + " ".join(extracted["mentioned"])
        
        rag_result = self.rag.rag_retrieve(query_for_rag)
        
        prompt = f"""请根据以下症状信息进行专业分析：

用户描述：{user_query}

提取的症状：
- 症状类型：{', '.join(extracted['mentioned']) if extracted['mentioned'] else '未明确'}
- 发病部位：{', '.join(extracted['location']) if extracted['location'] else '未明确'}
- 持续时间：{extracted['duration'] or '未明确'}
- 严重程度：{extracted['severity'] or '未明确'}

参考知识：{rag_result['answer'][:500]}

请提供简洁的症状分析，包括可能的皮肤问题类型、需要确认的问题和初步建议。
注意：不要使用列表格式（如1. 2. 或-）或emoji，用连贯的自然段落描述。
"""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = chat_model.invoke(messages)
        
        return {
            "analysis": response.content,
            "extracted_symptoms": extracted,
            "rag_result": rag_result,
            "need_image": len(extracted["mentioned"]) < 2 or not extracted["location"]
        }



    #单轮对话
    def chat(self, user_query: str) -> str:
        result = self.analyze(user_query)
        return result["analysis"]



    #判断需要进一步询问的问题
    def need_more_info(self, current_symptoms: dict) -> list[str]:
        questions = []
        
        if not current_symptoms.get("location"):
            questions.append("请问皮损具体在哪个部位？")
        
        if not current_symptoms.get("duration"):
            questions.append("这种情况出现多长时间了？")
        
        if not current_symptoms.get("severity"):
            questions.append("感觉严重吗？会影响日常生活吗？")
        
        return questions



#症状检查器：快速检查常见症状
class SymptomChecker:
    COMMON_SYMPTOMS = {
        "痤疮": ["痘痘", "粉刺", "毛孔粗大", "油脂多"],
        "湿疹": ["瘙痒", "红斑", "渗出", "反复"],
        "荨麻疹": ["风团", "忽起忽消", "剧烈瘙痒"],
        "银屑病": ["鳞屑", "红斑", "薄膜现象", "点状出血"],
        "真菌感染": ["脱屑", "瘙痒", "环状", "传染"],
        "疱疹": ["水疱", "疼痛", "簇集", "反复"],
        "色素疾病": ["色素沉着", "白斑", "颜色改变"],
        "皮肤肿瘤": ["痣", "增大", "破溃", "颜色不均"]
    }


    #快速检查可能的症状类型
    @classmethod
    def quick_check(cls, user_text: str) -> list[dict]:

        results = []
        text = user_text.lower()
        
        for disease, keywords in cls.COMMON_SYMPTOMS.items():
            match_count = sum(1 for kw in keywords if kw in text)
            if match_count > 0:
                results.append({
                    "disease": disease,
                    "match_count": match_count,
                    "keywords": [kw for kw in keywords if kw in text]
                })
        
        results.sort(key=lambda x: x["match_count"], reverse=True)
        return results


if __name__ == "__main__":
    agent = SymptomAnalyzerAgent()
    result = agent.analyze("脸上起了很多红疙瘩，很痒，帮忙看看是什么")
    print(result["analysis"])
