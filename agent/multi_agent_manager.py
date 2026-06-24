"""
多Agent协作管理器
协调症状分析Agent、图像诊断Agent、治疗建议Agent的工作
支持多轮对话、动态工具选择、自我反思和验证
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from model.factory import chat_model
from agent.symptom_agent import SymptomAnalyzerAgent
from agent.image_agent import ImageDiagnosisAgent
from agent.treatment_agent import TreatmentAdviceAgent, EmergencyChecker
from rag.enhanced_rag import EnhancedRAGService
from utils.logger import logger
import json
import os
import asyncio
from datetime import datetime
from enum import Enum

#诊断阶段
class DiagnosisStage(Enum):
    INIT = "init"
    SYMPTOM_ANALYSIS = "symptom_analysis"
    IMAGE_DIAGNOSIS = "image_diagnosis"
    TREATMENT_ADVICE = "treatment_advice"
    COMPLETED = "completed"

#任务类型
class TaskType(Enum):
    SYMPTOM_ONLY = "symptom_only"
    IMAGE_ONLY = "image_only"
    SYMPTOM_AND_IMAGE = "symptom_and_image"
    FOLLOW_UP = "follow_up"
    GREETING = "greeting"


#多Agent协作管理器
class MultiAgentManager:
    def __init__(self):
        self.symptom_agent = SymptomAnalyzerAgent()
        self.image_agent = ImageDiagnosisAgent()
        self.treatment_agent = TreatmentAdviceAgent()
        self.rag = EnhancedRAGService()
        
        self.current_stage = DiagnosisStage.INIT
        self.task_type = None
        
        self.conversation_history = []
        self.long_term_goals = []
        
        self.system_prompt = self._load_system_prompt()
        
        self.thinking_log = []
        
        # 图像分析缓存，避免重复分析相同图像
        self.image_analysis_cache = {}
        
        # 初始化context
        self.context = {
            "user_symptoms": None,
            "image_path": None,
            "diagnosis_result": None,
            "treatment_result": None,
            "patient_info": {},
            "conversation_history": []  # 保存对话历史
        }

    #加载系统提示词
    def _load_system_prompt(self) -> str:
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "skinderm_llm_decision_prompt.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        return """你是一个智能皮肤病诊断系统，通过多Agent协作完成诊断任务。

工作流程：
1. 分析用户症状描述
2. 如有图像，进行图像诊断
3. 综合分析结果给出治疗建议
4. 验证诊断结果的可靠性

你的目标是为用户提供专业、准确的皮肤病诊断建议。"""

    # 备用任务类型判断
    def determine_task_type(self, query: str, has_image: bool) -> TaskType:
        text = query.strip()
        has_history = self.context.get("diagnosis_result") is not None
        greeting_words = ["你好", "您好", "hi", "hello", "嗨", "hey", "在吗", "你是谁", "干嘛"]
        inquiry_words = ["这是什么", "帮我看看", "这是什么症状", "帮我诊断", "看看这是什么",
                        "这是什么病", "帮我分析", "这是什么情况", "帮我判断", "请问"]

        if text in greeting_words:
            return TaskType.GREETING

        if has_image:
            if not text or text in ["开始", "开始诊断"] or any(text.startswith(word) for word in inquiry_words):
                return TaskType.IMAGE_ONLY
            return TaskType.SYMPTOM_AND_IMAGE

        if has_history:
            return TaskType.FOLLOW_UP

        return TaskType.SYMPTOM_ONLY
    
    def chat(self, query: str) -> str:
        results = self.execute(query)
        return self.generate_response(results)
    
    #agent思考流式对话
    async def chat_with_thinking_stream(self, query: str, current_image_path: str | None = None):
        """
        带思考过程的流式对话 - LLM自主决策版
        LLM决定任务类型、调用哪些Agent、如何整合结果
        """
        self.thinking_log = []
        has_current_image = current_image_path is not None
        has_image = has_current_image or self.context.get("image_path") is not None
        
        # 1. 意图识别 - 告诉LLM有图像
        step = {
            "stage": "🔍 意图识别",
            "thought": f"接收到用户输入: '{query[:50]}...'",
            "reasoning": f"检测到图像: {'是' if has_image else '否'}，准备让LLM自主决策",
            "decision": "等待LLM决策",
            "status": "thinking"
        }
        self.thinking_log.append(step)
        yield {"type": "thinking", "data": [step], "task_type": "unknown"}
        
        await asyncio.sleep(2)
        
        # 2. LLM自主决策 - 让LLM决定任务类型和需要调用的Agent
        
        # 检查是否有之前的诊断结果（判断是否是追问）
        has_existing_diagnosis = self.context.get("diagnosis_result") is not None
        image_path = self.context.get("image_path")
        
        decision_prompt = f"""【任务】判断对话任务类型并决定需要调用的Agent

【当前会话状态】
- 本次请求是否上传了新图片：{'是' if has_current_image else '否'}
- 用户本次输入的症状描述："{query}"
- 会话历史中已存储的图片：{image_path if image_path else '无'}
- 是否有之前的诊断结果：{'是' if has_existing_diagnosis else '否'}
- 之前的诊断结果：{self.context.get('diagnosis_result', {}).get('disease_name', '') if has_existing_diagnosis else '无'}

【用户本次输入】
- 用户上传的图片：{'有' if has_current_image else '无'}
- 用户输入的文字："{query}"（注意：用户输入了文字，即使只是"这是什么"、"帮我看看"也属于有文字）

【任务类型定义】
1. greeting - 用户只是简单问候（你好、你是谁、hi、hello、在吗等）
2. image_only - 只有图片诊断，无症状文字描述
3. symptom_and_image - 同时有图片和症状描述
4. symptom_only - 只有症状文字描述，没有图片
5. follow_up - 追问/补充说明，使用之前的诊断结果继续回答

【Agent说明】
- SymptomAnalyzerAgent：分析症状文字
- ImageDiagnosisAgent：分析图片
- TreatmentAdviceAgent：生成治疗建议

【判断规则】（重要！严格遵守）
1. 如果用户只是问候 → greeting，不调用任何Agent
2. 如果本次上传了新图片 + 用户输入了文字描述（即使是"这是什么"、"帮我看看"等也算有文字）→ symptom_and_image
3. 如果本次上传了新图片，但没有任何文字输入 → image_only
4. 如果本次没有上传图片，但有之前的诊断结果，且用户只是在追问 → follow_up
5. 如果本次没有上传图片，也没有之前的诊断结果 → symptom_only

【关键区分】
- 追问特征：用户回复"比较疼痛"、"还有别的症状吗"、"需要手术吗"等
- 新图片特征：本次请求中有图片上传

请严格按以下格式输出：
任务类型: [greeting/image_only/symptom_and_image/symptom_only/follow_up]
需要调用的Agent: [Agent名称，用逗号分隔]
下一步行动: [简短描述]
"""
        
        try:
            messages = [HumanMessage(content=decision_prompt)]
            response = chat_model.invoke(messages)
            decision_text = str(response.content)
            
            # 解析LLM决策
            task_type_str = "unknown"
            agents_to_call = []
            
            # 优先检查 greeting 类型（简单问候词）
            query_lower = query.lower().strip()
            is_greeting = query_lower in ["你好", "你是谁", "hi", "hello", "嗨", "您好", "hey", "在吗", "干嘛", "开始", "开始诊断"] or query_lower.startswith("你好")
            
            if is_greeting or "greeting" in decision_text.lower():
                task_type_str = "greeting"
                agents_to_call = []
            elif "image_only" in decision_text.lower():
                task_type_str = "image_only"
                agents_to_call = ["ImageDiagnosisAgent", "TreatmentAdviceAgent"]
            elif "symptom_and_image" in decision_text.lower():
                task_type_str = "symptom_and_image"
                agents_to_call = ["SymptomAnalyzerAgent", "ImageDiagnosisAgent", "TreatmentAdviceAgent"]
            elif "symptom_only" in decision_text.lower():
                task_type_str = "symptom_only"
                agents_to_call = ["SymptomAnalyzerAgent", "TreatmentAdviceAgent"]
            else:
                task_type_str = "follow_up"
                agents_to_call = ["TreatmentAdviceAgent"]
            
            # greeting类型特殊处理
            if task_type_str == "greeting":
                # 使用RAG获取系统介绍
                rag_result = self.rag.rag_retrieve("皮诊智核 你是谁 系统介绍")
                system_info = rag_result.get("answer", "")[:500]
                
                # 获取参考资料
                documents = rag_result.get("documents", [])
                references = ""
                if documents:
                    for i, doc in enumerate(documents[:3], 1):
                        content = doc.page_content[:200] if hasattr(doc, 'page_content') else str(doc)[:200]
                        source = doc.metadata.get('source', '未知来源') if hasattr(doc, 'metadata') else '未知来源'
                        references += f"【参考资料{i}】({source}): {content}\n"
                
                # 判断是否有之前的诊断结果
                has_existing_diagnosis = self.context.get("diagnosis_result") is not None
                
                if has_existing_diagnosis:
                    # 有之前的诊断，结合诊断结果回复
                    diagnosis_info = self.context.get("diagnosis_result", {})
                    disease = diagnosis_info.get("disease_name", "")
                    greeting_prompt = f"""请以皮诊智核皮肤病智能诊断助手的身份回复用户。

用户问"你是谁"，请结合以下信息回复：

参考知识：
{system_info}

参考资料：
{references}

用户之前的诊断结果：{disease}

请生成友好的回复，介绍自己并结合之前的诊断结果。注意：不要使用任何列表格式或markdown符号，用自然段落描述。
"""
                else:
                    # 没有诊断，直接用RAG知识回复
                    greeting_prompt = f"""请以皮诊智核皮肤病智能诊断助手的身份回复用户。

用户问"你是谁"，请根据以下知识回复：

参考知识：
{system_info}

参考资料：
{references}

请生成友好的自我介绍。注意：不要使用任何列表格式或markdown符号，用自然段落描述。
"""
                
                messages = [HumanMessage(content=greeting_prompt)]
                response = chat_model.invoke(messages)
                greeting_response = response.content
                
                # 添加 LLM 决策步骤
                step = {
                    "stage": "💡 LLM决策",
                    "thought": f"识别到用户问候，任务类型: greeting",
                    "reasoning": "用户只是简单问候，无需调用诊断Agent，直接使用RAG知识回复",
                    "decision": "任务类型: greeting | 调用Agent: 无",
                    "tool": "无",
                    "status": "completed"
                }
                self.thinking_log.append(step)
                yield {"type": "thinking", "data": list(self.thinking_log), "task_type": "greeting"}
                
                yield {"type": "content", "data": greeting_response}
                yield {"type": "done"}
                return
            
            step = {
                "stage": "💡 LLM决策",
                "thought": decision_text[:200],
                "reasoning": "LLM自主分析了用户输入",
                "decision": f"任务类型: {task_type_str} | 调用Agent: {', '.join(agents_to_call)}",
                "tool": ", ".join(agents_to_call),
                "status": "completed"
            }
            self.thinking_log.append(step)
            yield {"type": "thinking", "data": list(self.thinking_log), "task_type": task_type_str}
            
            self.task_type = TaskType(task_type_str) if task_type_str != "unknown" else None
            
        except Exception as e:
            logger.error(f"LLM决策失败: {e}")
            # 备用决策
            task_type_str = self._decide_task_type_fallback(query, has_image).value
            agents = {
                "image_only": ["ImageDiagnosisAgent", "TreatmentAdviceAgent"],
                "symptom_only": ["SymptomAnalyzerAgent", "TreatmentAdviceAgent"],
                "symptom_and_image": ["SymptomAnalyzerAgent", "ImageDiagnosisAgent", "TreatmentAdviceAgent"],
                "follow_up": ["TreatmentAdviceAgent"]
            }
            agents_to_call = agents.get(task_type_str, ["TreatmentAdviceAgent"])
            step = {
                "stage": "💡 任务决策(备用)",
                "thought": f"LLM决策失败，使用备用方法: {task_type_str}",
                "decision": f"任务类型: {task_type_str} | 调用Agent: {', '.join(agents_to_call)}",
                "tool": ", ".join(agents_to_call),
                "status": "completed"
            }
            self.thinking_log.append(step)
            yield {"type": "thinking", "data": list(self.thinking_log), "task_type": task_type_str}
            self.task_type = TaskType(task_type_str)
        
        await asyncio.sleep(2)
        
        # 3. 根据决策调用相应的Agent
        results = await self._execute_by_decision(query, has_image)
        
        # 4. LLM整合结果生成最终回复
        step = {
            "stage": "📝 最终回复生成",
            "thought": "正在让LLM整合所有Agent结果生成回复...",
            "reasoning": "LLM根据用户问题和诊断结果自主决定回复内容",
            "status": "thinking"
        }
        self.thinking_log.append(step)
        yield {"type": "thinking", "data": list(self.thinking_log), "task_type": self.task_type.value if self.task_type else "unknown"}
        
        await asyncio.sleep(2)
        
        try:
            final_response = await self._llm_generate_response(query, results)
            yield {"type": "content", "data": final_response}
        except Exception as e:
            logger.error(f"LLM生成回复失败: {e}")
            yield {"type": "content", "data": self._generate_response_fallback(results)}
        
        yield {"type": "done"}

    # 根据决策执行相应的Agent
    async def _execute_by_decision(self, query: str, has_image: bool) -> dict:

        results = {}
        task_type = self.task_type
        
        # 症状分析（包含 follow_up，因为追问也可能需要分析症状）
        if task_type in [TaskType.SYMPTOM_ONLY, TaskType.SYMPTOM_AND_IMAGE, TaskType.FOLLOW_UP]:
            logger.info("🩺 调用症状分析Agent...")
            try:
                symptom_result = self.symptom_agent.analyze(query)
                results["symptom"] = symptom_result
                self.context["user_symptoms"] = query
                logger.info(f"🩺 症状分析完成: {symptom_result.get('extracted_symptoms', {}).get('mentioned', [])}")
            except Exception as e:
                logger.error(f"症状分析失败: {e}")
                results["symptom"] = {"analysis": "症状分析暂时不可用", "extracted_symptoms": {}}
        
        # 图像诊断
        if task_type in [TaskType.IMAGE_ONLY, TaskType.SYMPTOM_AND_IMAGE]:
            if self.context.get("image_path"):
                image_path = self.context["image_path"]
                # 检查缓存
                if image_path in self.image_analysis_cache:
                    logger.info("🖼️ 图像已分析，使用缓存结果")
                    results["image"] = self.image_analysis_cache[image_path]
                    self.context["diagnosis_result"] = self.image_analysis_cache[image_path]
                # 如果已经有诊断结果（刚才已经分析过），就不再重复分析
                elif self.context.get("diagnosis_result"):
                    logger.info("🖼️ 图像已诊断，跳过重复分析")
                    results["image"] = self.context["diagnosis_result"]
                else:
                    logger.info("🖼️ 调用图像诊断Agent...")
                    try:
                        user_symptoms = self.context.get("user_symptoms") or (query if query.strip() else None)
                        image_result = self.image_agent.analyze(image_path, user_symptoms)
                        results["image"] = image_result
                        self.context["diagnosis_result"] = image_result
                        # 缓存结果
                        self.image_analysis_cache[image_path] = image_result
                        logger.info(f"🖼️ 图像诊断完成: {image_result.get('disease_name', '未知')}")
                    except Exception as e:
                        logger.error(f"图像诊断失败: {e}")
                        results["image"] = {"diagnosis": "图像诊断暂时不可用", "disease_name": "未知"}
        
        # 治疗建议（包含 follow_up）
        if task_type in [TaskType.IMAGE_ONLY, TaskType.SYMPTOM_ONLY, TaskType.SYMPTOM_AND_IMAGE, TaskType.FOLLOW_UP]:
            logger.info("💊 调用治疗建议Agent...")
            try:
                # 优先使用之前的诊断结果
                diagnosis_info = results.get("image") or results.get("symptom", {})
                # 如果没有新的诊断结果，但有之前的诊断，使用之前的
                if not diagnosis_info and self.context.get("diagnosis_result"):
                    prev_diag = self.context.get("diagnosis_result", {})
                    diagnosis_info = {
                        "disease_name": prev_diag.get("disease_name", ""),
                        "model_results": prev_diag.get("model_results", {}),
                        "user_symptoms": query
                    }
                patient_info = self.context.get("patient_info", {})
                treatment_result = self.treatment_agent.analyze(diagnosis_info, patient_info)
                results["treatment"] = treatment_result
                logger.info("💊 治疗建议完成")
            except Exception as e:
                logger.error(f"治疗建议失败: {e}")
                results["treatment"] = {"treatment_plan": "暂时无法生成治疗建议"}
        
        return results          #AI辅助生成: 豆包，2026-04-02

    # 让LLM根据用户问题和诊断结果生成最终回复
    async def _llm_generate_response(self, query: str, results: dict) -> str:

        prompt = self._load_final_response_prompt()
        
        if not prompt:
            return self._generate_response_fallback(results)
        
        symptom_info = results.get("symptom", {}).get("analysis", "无")
        # 确保图像诊断结果正确传递，优先使用disease_name和model_results
        image_info = "无"
        if results.get("image"):
            image_result = results["image"]
            disease_name = image_result.get("disease_name", "未知")
            classification = image_result.get("model_results", {}).get("classification", "")
            diagnosis = image_result.get("diagnosis", "")
            # 组合图像诊断信息，确保结果一致性
            image_info = f"疾病名称：{disease_name}，{classification}，{diagnosis}" if diagnosis else f"疾病名称：{disease_name}，{classification}"
        
        treatment_info = results.get("treatment", {}).get("treatment_plan", "无")
        task_type = self.task_type.value if self.task_type else "unknown"
        
        # 增强对话历史的利用，保存更多历史记录
        history_text = ""
        if self.context.get("conversation_history"):
            # 保存更多历史，提高多轮对话连续性
            history = self.context["conversation_history"][-8:]  # 最近4轮对话
            for h in history:
                history_text += f"用户: {h.get('user', '')}\nAI: {h.get('assistant', '')}\n"
        
        # 获取之前的诊断结果作为上下文
        prev_diagnosis = ""
        if self.context.get("diagnosis_result"):
            prev = self.context["diagnosis_result"]
            disease = prev.get("disease_name", "")
            model_res = prev.get("model_results", {})
            classification = model_res.get("classification", "")
            if disease:
                prev_diagnosis = f"之前的诊断结果：{disease}（{classification}）"
        
        full_prompt = f"""用户问题：{query}
任务类型：{task_type}
{prev_diagnosis}
对话历史：
{history_text}

症状分析结果：{symptom_info}
图像诊断结果：{image_info}
治疗建议：：{treatment_info}

{prompt}"""
        
        try:
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=full_prompt)
            ]
            response = chat_model.invoke(messages)
            content = response.content
            if isinstance(content, list):
                content = str(content[0]) if content else ""
            
            # 保存对话历史，用于后续追问
            if self.context.get("conversation_history") is None:
                self.context["conversation_history"] = []
            self.context["conversation_history"].append({
                "user": query,
                "assistant": content
            })
            
            return content
        except Exception as e:
            logger.error(f"LLM生成回复失败: {e}")
            return self._generate_response_fallback(results)


    #加载llm决策提示词
    def _load_task_decision_prompt(self) -> str:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompt_path = os.path.join(base_dir, "prompts", "task_decision_prompt.txt")
        logger.info(f"尝试加载提示词: {prompt_path}")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        logger.warning(f"提示词文件不存在: {prompt_path}")
        return ""

    # 使用LLM判断任务类型
    def _decide_task_type(self, query: str, has_image: bool) -> TaskType:

        prompt = self._load_task_decision_prompt()
        
        if not prompt:
            logger.warning("未加载到任务决策提示词，使用备用方法")
            return self._decide_task_type_fallback(query, has_image)
        
        full_prompt = f"""用户问题：{query if query.strip() else '（无文字输入）'}

是否上传了图像：{'是' if has_image else '否'}

{prompt}"""
        
        logger.info(f"任务决策 - query='{query}', has_image={has_image}")
        
        try:
            messages = [
                HumanMessage(content=full_prompt)
            ]
            response = chat_model.invoke(messages)
            result = str(response.content).strip().lower()
            logger.info(f"LLM原始返回: {result}")
            
            if "greeting" in result:
                return TaskType.GREETING
            elif "image_only" in result:
                return TaskType.IMAGE_ONLY
            elif "symptom_and_image" in result:
                return TaskType.SYMPTOM_AND_IMAGE
            elif "symptom_only" in result:
                return TaskType.SYMPTOM_ONLY
            else:
                return TaskType.FOLLOW_UP
        except Exception as e:
            logger.error(f"LLM任务决策失败: {e}")
            return self._decide_task_type_fallback(query, has_image)



    """备用任务决策"""
    def _decide_task_type_fallback(self, query: str, has_image: bool) -> TaskType:

        query_has_content = query.strip() and query not in ["开始", "开始诊断"]
        has_history = self.context.get("diagnosis_result") is not None
        inquiry_words = ["这是什么", "帮我看看", "这是什么症状", "帮我诊断", "看看这是什么",
                        "这是什么病", "帮我分析", "这是什么情况", "帮我判断", "请问"]
        follow_up_words = ["还是", "那怎么办", "继续", "再看看", "还有", "补充", "之前"]
        is_inquiry = any(query.strip().startswith(word) for word in inquiry_words)
        is_follow_up = any(query.strip().startswith(word) for word in follow_up_words)

        if query.strip() in ["你好", "您好", "hi", "hello", "嗨", "hey", "在吗", "你是谁", "干嘛"]:
            return TaskType.GREETING

        if has_image and (not query_has_content or is_inquiry):
            return TaskType.IMAGE_ONLY
        elif has_image and query_has_content:
            return TaskType.SYMPTOM_AND_IMAGE
        elif (not has_image) and has_history and (is_follow_up or not query_has_content):
            return TaskType.FOLLOW_UP
        else:
            return TaskType.SYMPTOM_ONLY

    """任务决策步骤"""
    async def _task_decision_step(self, query: str, has_image: bool):

        await asyncio.sleep(5)
        
        # 2. 任务决策 - 使用LLM判断
        self.task_type = self._decide_task_type(query, has_image)
        
        step = {
            "stage": "💡 任务决策",
            "thought": "正在分析用户输入，判断任务类型...",
            "reasoning": "调用LLM进行任务类型决策",
            "decision": f"最终判定: {self.task_type.value if self.task_type else 'unknown'}",
            "status": "completed"
        }
        self.thinking_log.append(step)
        yield {"type": "thinking", "data": list(self.thinking_log)}
        
        await asyncio.sleep(5)
        
        results = {}
        
        # 3. 症状分析
        if self.task_type in [TaskType.SYMPTOM_ONLY, TaskType.SYMPTOM_AND_IMAGE]:
            step = {
                "stage": "🩺 症状分析",
                "thought": "正在调用症状分析Agent分析用户描述的症状特征...",
                "reasoning": "用户描述了症状，需要提取症状特征进行分析",
                "decision": "调用SymptomAnalyzerAgent",
                "tool": "SymptomAnalyzerAgent",
                "status": "calling"
            }
            self.thinking_log.append(step)
            yield {"type": "thinking", "data": list(self.thinking_log), "task_type": self.task_type.value if self.task_type else "unknown"}
            
            await asyncio.sleep(5)
            
            symptom_result = {"analysis": "症状分析暂时不可用，请稍后再试。", "extracted_symptoms": {}, "need_image": False}
            try:
                symptom_result = self.symptom_agent.analyze(query)
                results["symptom"] = symptom_result
                self.context["user_symptoms"] = query
            except Exception as e:
                logger.error(f"症状分析失败: {e}")
                results["symptom"] = symptom_result
            
            step = {
                "stage": "🩺 症状分析",
                "thought": "症状分析完成",
                "reasoning": f"提取到症状: {symptom_result.get('extracted_symptoms', {}).get('mentioned', [])}",
                "decision": f"分析结果: {symptom_result.get('analysis', '')[:80]}...",
                "tool": "SymptomAnalyzerAgent",
                "status": "completed"
            }
            self.thinking_log.append(step)
            yield {"type": "thinking", "data": list(self.thinking_log), "task_type": self.task_type.value if self.task_type else "unknown"}
            
            await asyncio.sleep(5)
            
            if symptom_result.get("need_image") and not self.context.get("image_path"):
                results["ask_for_image"] = True
                step = {
                    "stage": "⚠️ 信息评估",
                    "thought": "症状信息不足，需要更多图像来辅助诊断",
                    "reasoning": "症状分析结果建议需要图像辅助",
                    "decision": "提示用户上传图像",
                    "status": "waiting"
                }
                self.thinking_log.append(step)
                yield {"type": "thinking", "data": list(self.thinking_log), "task_type": self.task_type.value if self.task_type else "unknown"}
                await asyncio.sleep(5)
        
        # 4. 图像诊断
        if self.task_type in [TaskType.IMAGE_ONLY, TaskType.SYMPTOM_AND_IMAGE]:
            if self.context.get("image_path"):
                step = {
                    "stage": "🖼️ 图像诊断",
                    "thought": "正在调用图像诊断Agent分析用户上传的皮肤图像...",
                    "reasoning": f"图像路径: {os.path.basename(self.context['image_path'])}",
                    "decision": "调用ImageDiagnosisAgent",
                    "tool": "ImageDiagnosisAgent",
                    "status": "calling"
                }
                self.thinking_log.append(step)
                yield {"type": "thinking", "data": list(self.thinking_log), "task_type": self.task_type.value if self.task_type else "unknown"}
                
                await asyncio.sleep(5)
                
                try:
                    # IMAGE_ONLY 时 user_symptoms 还没设置，直接用 query
                    user_symptoms_for_image = self.context.get("user_symptoms") or (query if query.strip() else "")
                    image_result = self.image_agent.analyze(
                        self.context["image_path"],
                        user_symptoms_for_image if user_symptoms_for_image else None
                    )
                    results["image"] = image_result
                    self.context["diagnosis_result"] = image_result
                    
                    disease = image_result.get('disease_name', '未知')
                    confidence = image_result.get('model_results', {}).get('classification', '')
                except Exception as e:
                    logger.error(f"图像诊断失败: {e}")
                    results["image"] = {"diagnosis": "图像诊断暂时不可用，请稍后再试。", "disease_name": "未知", "model_results": {}}
                    confidence = ""
                    disease = "未知"
                
                step = {
                    "stage": "🖼️ 图像诊断",
                    "thought": f"图像诊断完成，AI识别结果: {disease}",
                    "reasoning": f"模型置信度: {confidence}",
                    "decision": "图像诊断已完成",
                    "tool": "ImageDiagnosisAgent",
                    "status": "completed"
                }
                self.thinking_log.append(step)
                yield {"type": "thinking", "data": list(self.thinking_log), "task_type": self.task_type.value if self.task_type else "unknown"}
                await asyncio.sleep(5)
        
        # 5. 治疗建议
        if self.task_type != TaskType.FOLLOW_UP:
            step = {
                "stage": "💊 治疗建议",
                "thought": "正在调用治疗建议Agent生成治疗方案和护理建议...",
                "reasoning": "综合诊断信息生成治疗方案",
                "decision": "调用TreatmentAdviceAgent",
                "tool": "TreatmentAdviceAgent",
                "status": "calling"
            }
            self.thinking_log.append(step)
            yield {"type": "thinking", "data": list(self.thinking_log), "task_type": self.task_type.value if self.task_type else "unknown"}
            
            await asyncio.sleep(5)
            
            try:
                diagnosis_info = {
                    "disease_name": results.get("image", {}).get("disease_name", ""),
                    "user_symptoms": self.context.get("user_symptoms", "") or ""
                }
                treatment_result = self.treatment_agent.analyze(diagnosis_info, self.context.get("patient_info") or {})
                results["treatment"] = treatment_result
                self.context["treatment_result"] = treatment_result
            except Exception as e:
                logger.error(f"治疗建议生成失败: {e}")
                results["treatment"] = {"treatment_plan": "治疗建议暂时不可用，请稍后再试。", "rag_result": {}, "sources": {}}
            
            step = {
                "stage": "💊 治疗建议",
                "thought": "治疗建议生成完成",
                "reasoning": "已生成治疗方案和护理建议",
                "decision": "治疗建议已生成",
                "tool": "TreatmentAdviceAgent",
                "status": "completed"
            }
            self.thinking_log.append(step)
            yield {"type": "thinking", "data": list(self.thinking_log), "task_type": self.task_type.value if self.task_type else "unknown"}
            await asyncio.sleep(5)
        
        # 6. 生成最终回复
        step = {
            "stage": "📝 回复生成",
            "thought": "正在整合所有Agent的结果，生成完整诊断报告...",
            "reasoning": "综合症状分析、图像诊断、治疗建议生成最终回复",
            "decision": "生成最终回复",
            "status": "thinking"
        }
        self.thinking_log.append(step)
        yield {"type": "thinking", "data": list(self.thinking_log), "task_type": self.task_type.value if self.task_type else "unknown"}
        await asyncio.sleep(3)
        
        try:
            response = self.generate_response(results)
        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            response = "抱歉，服务暂时不可用，请稍后再试。"
        
        step = {
            "stage": "📝 回复生成",
            "thought": "诊断报告生成完成",
            "reasoning": "已生成完整的诊断报告和建议",
            "decision": "准备向用户返回结果",
            "status": "completed"
        }
        self.thinking_log.append(step)
        
        # 构建工具调用
        tool_calls = self._build_tool_calls(results)
        yield {"type": "tools", "data": tool_calls}
        
        # 返回完整内容
        yield {"type": "content", "data": response}
        yield {"type": "done", "data": {}}

    # 执行多Agent协作诊断
    def execute(self, query: str) -> dict:

        has_image = self.context.get("image_path") is not None
        self.task_type = self.determine_task_type(query, has_image)

        results = {}

        if self.task_type in [TaskType.SYMPTOM_ONLY, TaskType.SYMPTOM_AND_IMAGE]:
            symptom_result = self.symptom_agent.analyze(query)
            results["symptom"] = symptom_result
            self.context["user_symptoms"] = query
            
            if symptom_result.get("need_image") and not self.context.get("image_path"):
                results["ask_for_image"] = True
        
        if self.task_type in [TaskType.IMAGE_ONLY, TaskType.SYMPTOM_AND_IMAGE]:
            if self.context.get("image_path"):
                image_result = self.image_agent.analyze(
                    self.context["image_path"],
                    self.context.get("user_symptoms") or ""
                )
                results["image"] = image_result
                self.context["diagnosis_result"] = image_result
        
        if self.task_type != TaskType.FOLLOW_UP:
            diagnosis_info = {
                "disease_name": results.get("image", {}).get("disease_name", ""),
                "user_symptoms": self.context.get("user_symptoms", "") or ""
            }
            treatment_result = self.treatment_agent.analyze(diagnosis_info, self.context.get("patient_info") or {})
            results["treatment"] = treatment_result
            self.context["treatment_result"] = treatment_result
        
        return results
    
    #加载回复提示词
    def _load_final_response_prompt(self) -> str:
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "final_response_prompt.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    # 生成最终响应 - 使用LLM根据用户问题和诊断结果生成个性化回复
    def generate_response(self, results: dict) -> str:

        
        task_type = self.task_type.value if self.task_type else "unknown"
        
        prompt = self._load_final_response_prompt()
        
        if not prompt:
            return self._generate_response_fallback(results)
        
        symptom_info = ""
        if "symptom" in results and results.get("symptom", {}).get("analysis"):
            symptom_info = results["symptom"]["analysis"]
        
        ask_for_image = "是" if results.get("ask_for_image") else "否"
        
        image_info = ""
        if "image" in results and results.get("image", {}).get("diagnosis"):
            image_info = results["image"]["diagnosis"]
        
        treatment_info = ""
        if "treatment" in results and results.get("treatment", {}).get("treatment_plan"):
            treatment_info = results["treatment"]["treatment_plan"]
        
        full_prompt = f"""用户问题：{self.context.get('user_symptoms', '无')}

任务类型：{task_type}

症状分析结果：
{symptom_info}

是否需要图像：{ask_for_image}

图像诊断结果：
{image_info}

治疗建议：
{treatment_info}

请根据以上信息生成最终回复。"""
        
        try:
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=full_prompt)
            ]
            response = chat_model.invoke(messages)
            content = response.content
            if isinstance(content, list):
                content = str(content[0]) if content else ""
            return content
        except Exception as e:
            logger.error(f"LLM生成回复失败: {e}")
            return self._generate_response_fallback(results)

    #将多Agent的分散结果拼接成统一的用户可读回复
    def _generate_response_fallback(self, results: dict) -> str:
        response_parts = []

        #模块化组装
        if "symptom" in results and results.get("symptom", {}).get("analysis"):
            response_parts.append("【症状分析】\n" + results["symptom"]["analysis"])
        
        if "ask_for_image" in results:
            response_parts.append("\n为了更准确诊断，能否提供皮肤图像？")
        
        if "image" in results and results.get("image", {}).get("diagnosis"):
            response_parts.append("\n【图像诊断】\n" + results["image"]["diagnosis"])
        
        if "treatment" in results and results.get("treatment", {}).get("treatment_plan"):
            response_parts.append("\n【治疗建议】\n" + results["treatment"]["treatment_plan"])

        #异常处理
        if not response_parts:
            response_parts.append("抱歉，暂时无法生成诊断结果，请稍后重试。")
        
        response = "\n".join(response_parts)
        
        response += "\n\n" + "="*40
        response += "\n免责声明：本诊断仅供参考，不能替代专业医生诊断。"
        response += "\n如有疑虑，请及时就医。"
        response += "\n" + "="*40
        
        return response

    # 构建工具调用记录
    def _build_tool_calls(self, results: dict) -> list:

        calls = []
        
        if "symptom" in results:
            calls.append({
                "tool": "SymptomAnalyzerAgent",
                "action": "analyze",
                "status": "success",
                "input": "用户症状描述",
                "output": "症状分析结果"
            })
        
        if "image" in results:
            calls.append({
                "tool": "ImageDiagnosisAgent",
                "action": "analyze",
                "status": "success",
                "input": "皮肤图像",
                "output": results["image"].get("disease_name", "未知")
            })
        
        if "treatment" in results:
            calls.append({
                "tool": "TreatmentAdviceAgent",
                "action": "analyze",
                "status": "success",
                "input": "诊断信息",
                "output": "治疗方案"
            })
        
        calls.append({
            "tool": "EnhancedRAGService",
            "action": "rag_retrieve",
            "status": "success",
            "input": "查询知识库",
            "output": "检索相关皮肤病资料"
        })
        
        return calls

    # 更新患者信息
    def update_patient_info(self, info: dict):

        self.context["patient_info"].update(info)

    # 清空上下文
    def clear_context(self):

        self.context = {
            "user_symptoms": None,
            "image_path": None,
            "diagnosis_result": None,
            "treatment_result": None,
            "patient_info": self.context.get("patient_info", {})
        }
