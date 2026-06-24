"""
定义皮肤疾病诊断智能体可调用的各种工具函数
使用@tool装饰器注册为可用工具
"""
from utils.logger import logger
from langchain_core.tools import tool
from rag.enhanced_rag import EnhancedRAGService
import torch
from utils.config_handler import model_conf
import os
from PIL import Image
import torchvision.transforms as transforms
import datetime



classifier_model = None
rag_service = None



#加载分类模型
def get_classifier_model():
    global classifier_model
    if classifier_model is None:
        from model.custom_skin_net import CustomSkinNet
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(project_root, "variables", "efficientnet_b3", "best_model.pth.tar")
        logger.info(f"[分类] 加载自定义模型: {model_path}")
        classifier_model = CustomSkinNet(
            num_classes=model_conf["num_classes"],
            width_coef=1.5,
            depth_coef=1.4,
            pretrained=False
        )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        classifier_model.load_state_dict(checkpoint["model_state_dict"])
        classifier_model.eval()
    return classifier_model



#加载RAG服务
def get_rag_service():
    global rag_service
    if rag_service is None:
        rag_service = EnhancedRAGService()
    return rag_service


# 皮肤病类别标签
SKIN_DISEASE_CLASSES= [
"痤疮和酒渣鼻","光化性角化病和基底细胞癌","特应性皮炎",
"大疱性疾病","蜂窝组织炎和细菌感染","湿疹",
"发疹和药物性皮炎","脱发","疱疹/HPV",
"色素性疾病","红斑狼疮","黑色素瘤和痣",
"甲真菌病","毒葛皮炎","银屑病和扁平苔藓",
"疥疮和莱姆病","脂溢性角化病和良性肿瘤","系统性疾病",
"真菌感染","荨麻疹","血管瘤",
"血管炎","疣和传染性软疣"
]


@tool(description="使用分类模型对皮肤图像进行疾病分类判断")
def skin_classify(image_path: str) -> str:
    """
    返回:分类结果，包含疾病类型和置信度
    """
    try:
        model = get_classifier_model()
        
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        disease = SKIN_DISEASE_CLASSES[predicted.item()]
        conf = confidence.item()
        
        return f"分类结果：{disease}，置信度：{conf:.2%}"
        
    except Exception as e:
        logger.error(f"[skin_classify]分类失败: {e}")
        return f"分类失败: {str(e)}"


@tool(description="根据疾病名称和用户症状从医学知识库检索相关建议")
def rag_query(query: str) -> str:
    """
    query: 查询字符串，包含疾病名称和/或用户症状描述
    返回:从知识库检索并总结后的相关医学建议
    """

    try:
        rag = get_rag_service()
        result = rag.rag_retrieve(query)

        documents = result.get("documents", [])
        answer = result.get("answer", "")

        if not documents:
            return "未找到相关医学知识，建议咨询专业医生"

        return answer if answer else "\n".join([doc.page_content for doc in documents])

    except Exception as e:
        logger.error(f"[rag_query]检索失败: {e}")
        return f"检索失败: {str(e)}"


@tool(description="生成完整的皮肤疾病诊断报告，包含分类、症状和建议信息")
def generate_report(classification_result: str, user_symptoms: str, rag_suggestions: str) -> str:
    """
    classification_result: 分类模型结果
    user_symptoms: 用户描述的症状
    rag_suggestions: RAG检索的建议
    返回:格式化的诊断报告
    """

    report = f"""
===============================================
        皮肤疾病诊断报告
===============================================

【基本信息】
- 诊断时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 检测方式：AI辅助诊断

【疾病分类结果】
{classification_result}

【用户症状】
{user_symptoms if user_symptoms else '用户未提供'}

【医学建议】
{rag_suggestions}

===============================================
              免责声明
===============================================
本报告由AI系统辅助生成，仅供参考学习，不能替代
专业皮肤科医生的面诊和诊断。如有身体不适或
疑虑，请及时前往正规医院皮肤科就诊。

报告编号：{hash(datetime.datetime.now()) % 100000:05d}
===============================================
    """
    return report
