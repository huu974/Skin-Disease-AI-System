"""
FastAPI 后端入口
皮肤病智能诊断系统 API
"""

from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
import sys
import tempfile
import asyncio
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.path_tool import get_abs_path
from agent.multi_agent_manager import MultiAgentManager

app = FastAPI(title="皮肤病智能诊断系统 API", version="1.0.0")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据模型

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    diagnosis_context: Optional[dict] = None
    images: Optional[list] = None

class UserProfile(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    skin_type: Optional[str] = None
    allergies: Optional[str] = None

# 模拟数据存储（生产环境用数据库）

# 用户存储 {username: {password, email, ...}}
users_db = {}

# 历史记录存储
diagnosis_history = []

# 当前登录用户
current_user = {"username": "", "token": ""}

# 用户信息存储
user_profiles = {
    "default": {
        "name": "访客用户",
        "email": "",
        "phone": "",
        "age": 25,
        "gender": "未设置",
        "skin_type": "普通",
        "allergies": "",
        "theme": "light",
        "language": "中文"
    }
}

# 对话历史
chat_sessions = {}

# 按session_id存储不同的Agent实例
agent_managers = {}

def get_agent_manager(session_id: str = "default"):
    global agent_managers
    if session_id not in agent_managers:
        agent_managers[session_id] = MultiAgentManager()
    return agent_managers[session_id]

def clear_agent_manager(session_id: str = None):
    #清空Agent管理器
    global agent_managers
    if session_id and session_id in agent_managers:
        del agent_managers[session_id]
    elif session_id is None:
        agent_managers = {}

# 类别名称
CLASS_NAMES = [
    "痤疮和酒渣鼻", "光化性角化病和基底细胞癌", "特应性皮炎",
    "大疱性疾病", "蜂窝组织炎和细菌感染", "湿疹",
    "发疹和药物性皮炎", "脱发", "疱疹/HPV",
    "色素性疾病", "红斑狼疮", "黑色素瘤和痣",
    "甲真菌病", "毒葛皮炎", "银屑病和扁平苔藓",
    "疥疮和莱姆病", "脂溢性角化病和良性肿瘤", "系统性疾病",
    "真菌感染", "荨麻疹", "血管瘤",
    "血管炎", "疣和传染性软疣"
]

# 加载模型
model = None

def load_classification_model():
    global model
    try:
        from model.custom_skin_net import CustomSkinNet
        model_path = get_abs_path("variables/efficientnet_b3/best_model.pth.tar")
        
        print(f"加载模型: {model_path}")
        
        model = CustomSkinNet(
            num_classes=23,
            width_coef=1.5,
            depth_coef=1.4,
            pretrained=False
        )
        
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        model = None

load_classification_model()

# 图像预处理（与 agent_tools.py 一致）
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ============ API 路由 ============

@app.get("/")
async def root():
    return {"message": "皮肤病智能诊断系统 API", "version": "1.0.0"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ============ 检测与分类 API ============

@app.post("/api/detect")
async def detect_and_classify(
    image: UploadFile = File(...),
    classification_model: Optional[str] = Form(default="efficientnet_b3")
):

    try:
        # 保存上传的图像
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image.filename.split('.')[-1]}") as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 使用模型进行分类
        if model is not None:
            try:
                img = Image.open(tmp_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    
                # 获取 Top-5
                top5_probs, top5_indices = torch.topk(probs, 5)
                top5_results = []
                for prob, idx in zip(top5_probs, top5_indices):
                    top5_results.append({
                        "class": CLASS_NAMES[idx.item()],
                        "probability": round(prob.item(), 4)
                    })
                
                primary_class = CLASS_NAMES[top5_indices[0].item()]
                primary_prob = top5_probs[0].item()
            except Exception as e:
                print(f"分类失败: {e}")
                top5_results = [{"class": "未知", "probability": 0.0}]
                primary_class = "未知"
                primary_prob = 0.0
        else:
            top5_results = [{"class": "模型未加载", "probability": 0.0}]
            primary_class = "模型未加载"
            primary_prob = 0.0
        
        result = {
            "cropped_regions": [tmp_path],
            "classification": {
                "model_used": classification_model,
                "top1": {
                    "class": primary_class,
                    "probability": round(primary_prob, 4)
                },
                "top5": top5_results
            },
            "timestamp": datetime.now().isoformat()
        }

        # 保存到历史记录
        record = {
            "id": len(diagnosis_history) + 1,
            "image_path": tmp_path,
            "image_name": image.filename,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        diagnosis_history.append(record)

        return JSONResponse(content=result)
    #AI辅助生成: 豆包，2026-04-04
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


#智能对话 API

@app.post("/api/chat/stream")
async def chat_stream(
    message: str = Form(...),
    session_id: str = Form(default="default"),
    images: List[UploadFile] = Form(default=None)
):
    """
    流式智能对话 API
    - 基于 RAG + Agent 的智能问诊
    - 支持文字+图片
    """
    print(f"[DEBUG] Received request - message: {message}, images: {images}")
    try:
        # session_id 已通过 Form 有默认值
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        # 保存图片路径
        image_paths = []
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        if images is not None:
            for img in images:
                if img and img.content_type and img.content_type.startswith('image/'):
                    import uuid
                    ext = img.filename.split('.')[-1] if '.' in img.filename else 'jpg'
                    img_path = os.path.join(upload_dir, f"{uuid.uuid4()}.{ext}")
                    content = await img.read()
                    with open(img_path, 'wb') as f:
                        f.write(content)
                    image_paths.append(img_path)
        
        chat_sessions[session_id].append({
            "role": "user",
            "content": message,
            "images": image_paths,
            "timestamp": datetime.now().isoformat()
        })

        # 调用真正的 Agent 进行处理（流式）
        agent = get_agent_manager(session_id)
        
        # 调试：打印关键信息
        print(f"[DEBUG] session_id: {session_id}")
        print(f"[DEBUG] chat_sessions keys: {list(chat_sessions.keys())}")
        print(f"[DEBUG] chat_sessions[{session_id}]: {chat_sessions.get(session_id, [])}")
        print(f"[DEBUG] session message count: {len(chat_sessions.get(session_id, []))}")
        print(f"[DEBUG] is_new_session: {session_id not in chat_sessions or len(chat_sessions.get(session_id, [])) == 0}")
        
        # 检查是否为第一次对话（会话刚刚创建，且只有1条消息）
        is_new_session = session_id not in chat_sessions or len(chat_sessions.get(session_id, [])) == 0
        
        if is_new_session:
            # 新会话，初始化Agent的context
            print("[DEBUG] 初始化Agent context（新会话）")
            agent.context = {
                "image_path": None,
                "diagnosis_result": None,
                "user_symptoms": None,
                "conversation_history": [],
                "patient_info": {}
            }
        elif not image_paths and agent.context.get("diagnosis_result"):
            # 有历史但没有新图片，超过6条消息后重置诊断
            history_count = len(chat_sessions.get(session_id, []))
            if history_count > 6:
                agent.context["diagnosis_result"] = None
        
        # 保存对话历史到agent的context中
        agent.context["conversation_history"] = [
            {"user": msg["content"], "assistant": msg.get("assistant_content", "")}
            for msg in chat_sessions[session_id][-10:]
            if msg["role"] == "user"
        ]
        
        # 根据是否有图片决定处理方式
        has_existing_diagnosis = agent.context.get("diagnosis_result") is not None
        
        # 如果没有新图片且没有之前的诊断结果，走完整Agent流程（让LLM智能回答）
        if not image_paths and not has_existing_diagnosis:
            full_response = []
            
            async def generate_with_agent():
                nonlocal full_response
                async for chunk in agent.chat_with_thinking_stream(message, current_image_path=image_paths[0] if image_paths else None):
                    if chunk["type"] == "thinking":
                        yield f"data: {json.dumps({'type': 'thinking', 'data': chunk['data']})}\n\n"
                    elif chunk["type"] == "tools":
                        yield f"data: {json.dumps({'type': 'tools', 'data': chunk['data']})}\n\n"
                    elif chunk["type"] == "content":
                        full_response.append(chunk['data'])
                        for char in chunk['data']:
                            yield f"data: {json.dumps({'type': 'content', 'data': char})}\n\n"
                            await asyncio.sleep(0.02)
                    elif chunk["type"] == "done":
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
            # 流式结束后更新对话历史
            if agent.context.get("conversation_history") and full_response:
                if len(agent.context["conversation_history"]) > 0:
                    agent.context["conversation_history"][-1]["assistant"] = "".join(full_response)
            
            return StreamingResponse(generate_with_agent(), media_type="text/event-stream")
        
        # 有图片时，使用完整Agent推理流程
        if image_paths:
            image_result = agent.image_agent.analyze(image_paths[0], message)
            agent.context["image_path"] = image_paths[0]
            agent.context["diagnosis_result"] = image_result
        
        full_response = []
        
        async def generate():
            nonlocal full_response
            # 完整Agent流程
            async for chunk in agent.chat_with_thinking_stream(message, current_image_path=image_paths[0] if image_paths else None):
                if chunk["type"] == "thinking":
                    yield f"data: {json.dumps({'type': 'thinking', 'data': chunk['data']})}\n\n"
                elif chunk["type"] == "tools":
                    yield f"data: {json.dumps({'type': 'tools', 'data': chunk['data']})}\n\n"
                elif chunk["type"] == "content":
                    full_response.append(chunk['data'])
                    for char in chunk['data']:
                        yield f"data: {json.dumps({'type': 'content', 'data': char})}\n\n"
                        await asyncio.sleep(0.02)
                elif chunk["type"] == "done":
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    
                    # 只有本实际上传了图片时才保存诊断结果到历史记录
                    if image_paths and agent.context.get("diagnosis_result"):
                        diagnosis_result = agent.context.get("diagnosis_result")
                        disease = diagnosis_result.get("disease_name", "")
                        model_results = diagnosis_result.get("model_results", {})
                        classification = model_results.get("classification", "")
                        
                        # 提取置信度
                        confidence_value = 0.0
                        if classification:
                            import re
                            match = re.search(r'置信度[：:]\s*([\d.]+)', classification)
                            if match:
                                confidence_value = float(match.group(1)) / 100
                            else:
                                match = re.search(r'([\d.]+)%', classification)
                                if match:
                                    confidence_value = float(match.group(1)) / 100
                        
                        if disease:
                            record = {
                                "id": len(diagnosis_history) + 1,
                                "image_name": diagnosis_result.get("image_name", "智能对话"),
                                "result": {
                                    "classification": {
                                        "top1": {
                                            "class": disease,
                                            "probability": confidence_value
                                        }
                                    }
                                },
                                "timestamp": datetime.now().isoformat(),
                                "source": "chat"
                            }
                            diagnosis_history.append(record)
            
            # 流式结束后更新对话历史
            if agent.context.get("conversation_history") and full_response:
                if len(agent.context["conversation_history"]) > 0:
                    agent.context["conversation_history"][-1]["assistant"] = "".join(full_response)
        
        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        import traceback
        print(f"对话处理错误: {e}")
        print(traceback.format_exc())
        
        async def generate_error():
            yield f"data: {json.dumps({'type': 'thinking', 'data': []})}\n\n"
            yield f"data: {json.dumps({'type': 'tools', 'data': []})}\n\n"
            error_msg = "抱歉，服务暂时不可用，请稍后重试。"
            yield f"data: {json.dumps({'type': 'content', 'data': error_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        return StreamingResponse(generate_error(), media_type="text/event-stream")


@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or "default"
        
        # 初始化会话
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        # 添加用户消息
        chat_sessions[session_id].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        
        # 调用真正的 Agent 进行处理
        agent = get_agent_manager()
        
        # 保存对话历史到agent的context中
        agent.context["conversation_history"] = [
            {"user": msg["content"], "assistant": ""}
            for msg in chat_sessions[session_id][-10:] if msg["role"] == "user"
        ]
        
        response_text = agent.chat(request.message)
        
        # 更新对话历史，添加AI的回复
        if agent.context.get("conversation_history"):
            if len(agent.context["conversation_history"]) > 0:
                agent.context["conversation_history"][-1]["assistant"] = response_text
        
        reasoning_steps = [
            {"step": "理解问题", "content": "分析用户描述的皮肤症状和相关需求"},
            {"step": "知识检索", "content": "从RAG知识库中检索相关的皮肤病信息"},
            {"step": "症状分析", "content": "使用症状分析Agent分析症状特点"},
            {"step": "诊断推理", "content": "结合检索结果进行诊断推理"},
            {"step": "建议生成", "content": "生成诊断结果和护理建议"}
        ]

        # 添加AI回复
        chat_sessions[session_id].append({
            "role": "assistant",
            "content": response_text,
            "steps": reasoning_steps,
            "timestamp": datetime.now().isoformat()
        })

        return {
            "reply": response_text,
            "steps": reasoning_steps,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对话失败: {str(e)}")


@app.get("/api/chat/history")
async def get_chat_history(session_id: str = "default"):
    #获取对话历史
    return {
        "session_id": session_id,
        "history": chat_sessions.get(session_id, [])
    }


@app.delete("/api/chat/history")
async def clear_chat_history(session_id: str = "default"):
    if session_id in chat_sessions:
        chat_sessions[session_id] = []
    # 删除对应的Agent实例
    clear_agent_manager(session_id)
    return {"message": "对话历史已清空"}


#历史记录 API

@app.get("/api/history")
async def get_history(page: int = 1, size: int = 10):
    start = (page - 1) * size
    end = start + size
    
    records = diagnosis_history[start:end]
    total = len(diagnosis_history)
    
    return {
        "page": page,
        "size": size,
        "total": total,
        "records": [
            {
                "id": r["id"],
                "image_name": r["image_name"],
                "result": r["result"],
                "timestamp": r["timestamp"]
            }
            for r in records
        ]
    }


@app.get("/api/history/{record_id}")
async def get_history_detail(record_id: int):
    """获取单条诊断详情"""
    for record in diagnosis_history:
        if record["id"] == record_id:
            return record
    raise HTTPException(status_code=404, detail="记录不存在")


@app.post("/api/detect/save")
async def save_diagnosis(
    image_name: str = Body(...),
    result: dict = Body(...)
):
    """保存诊断结果到历史记录"""
    global diagnosis_history
    
    record = {
        "id": len(diagnosis_history) + 1,
        "image_name": image_name,
        "result": result,
        "timestamp": datetime.now().isoformat(),
        "source": "diagnosis"
    }
    diagnosis_history.append(record)
    return {"message": "保存成功"}

@app.delete("/api/history/{record_id}")
async def delete_history(record_id: int):
    global diagnosis_history
    diagnosis_history = [r for r in diagnosis_history if r["id"] != record_id]
    return {"message": "删除成功"}


#用户中心 API

@app.get("/api/user/profile")
async def get_user_profile(user_id: str = "default"):
    #获取用户信息
    return user_profiles.get(user_id, user_profiles["default"])


@app.put("/api/user/profile")
async def update_user_profile(profile: UserProfile, user_id: str = "default"):
    if user_id in user_profiles:
        user_profiles[user_id].update(profile.dict(exclude_unset=True))
    else:
        user_profiles[user_id] = profile.dict()
    
    return {"message": "用户信息已更新", "profile": user_profiles[user_id]}


@app.post("/api/auth/register")
async def register(username: str = Form(...), password: str = Form(...), email: str = Form(...)):
    #用户注册
    if username in users_db:
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    users_db[username] = {
        "password": password,
        "email": email,
        "created_at": datetime.now().isoformat()
    }
    
    # 创建用户 profile
    user_profiles[username] = {
        "name": username,
        "email": email,
        "phone": "",
        "age": 25,
        "gender": "未设置",
        "skin_type": "普通",
        "allergies": "",
        "theme": "light",
        "language": "中文"
    }
    
    return {"message": "注册成功", "username": username}


@app.post("/api/auth/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if username not in users_db:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    
    if users_db[username]["password"] != password:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    
    token = f"token_{username}_{datetime.now().timestamp()}"
    current_user["username"] = username
    current_user["token"] = token
    
    return {
        "token": token,
        "username": username,
        "email": users_db[username]["email"]
    }


@app.post("/api/auth/logout")
async def logout():
    #用户登出
    current_user["username"] = ""
    current_user["token"] = ""
    return {"message": "已退出登录"}


@app.get("/api/auth/status")
async def get_auth_status():
    #获取登录状态
    return {
        "is_logged_in": bool(current_user["username"]),
        "username": current_user["username"]
    }


@app.get("/api/admin/users")
async def get_all_users():
    users_list = []
    for username, info in users_db.items():
        users_list.append({
            "username": username,
            "email": info.get("email", ""),
            "created_at": info.get("created_at", ""),
            "profile": user_profiles.get(username, {})
        })
    return {
        "total": len(users_list),
        "users": users_list
    }





#预防建议 API

@app.get("/api/advice")
async def get_advice(disease: Optional[str] = None):
    """获取预防建议"""
    advice_data = {
        "银屑病": {
            "预防": ["避免感染", "减少精神压力", "戒烟戒酒", "避免外伤"],
            "饮食": ["多吃蔬果", "补充Omega-3", "少吃牛羊肉"],
            "护理": ["保持皮肤湿润", "避免过度洗澡", "使用温和护肤品"]
        },
        "湿疹": {
            "预防": ["保持皮肤湿润", "避免接触过敏原", "穿棉质衣物", "适度洗澡"],
            "饮食": ["避免海鲜", "少吃辛辣", "补充维生素C"],
            "护理": ["使用保湿霜", "避免抓挠", "温水洗澡"]
        },
        "痤疮": {
            "预防": ["清洁面部", "不挤压痘痘", "规律作息", "卸妆彻底"],
            "饮食": ["少吃甜食", "避免油炸", "多喝水"],
            "护理": ["每天清洁", "使用控油产品", "避免用手触摸"]
        },
        "荨麻疹": {
            "预防": ["查找过敏原", "避免冷热刺激", "穿着宽松", "保持清洁"],
            "饮食": ["记录食物日记", "避免已知过敏食物"],
            "护理": ["冷敷缓解", "避免抓挠", "穿着宽松衣物"]
        }
    }
    
    if disease:
        return advice_data.get(disease, {"message": "未找到该疾病的建议"})
    return advice_data


#仪表盘统计 API

def get_db_stats():
    global diagnosis_history
    
    total_diagnoses = len(diagnosis_history)
    total_users = 0
    
    if total_diagnoses > 0:
        confidences = []
        for r in diagnosis_history:
            try:
                prob = r.get("result", {}).get("classification", {}).get("top1", {}).get("probability", 0)
                if prob > 0:
                    confidences.append(prob)
            except:
                pass
        if confidences:
            accuracy = round(sum(confidences) / len(confidences) * 100, 1)
        else:
            accuracy = 0
    else:
        accuracy = 0
    
    from datetime import datetime, timedelta
    week_ago = datetime.now() - timedelta(days=7)
    
    diagnoses_this_week = sum(1 for r in diagnosis_history 
                            if datetime.fromisoformat(r["timestamp"]) >= week_ago)
    
    diagnoses_last_week = total_diagnoses - diagnoses_this_week
    
    diagnoses_change = 0
    if diagnoses_last_week > 0:
        diagnoses_change = round((diagnoses_this_week / diagnoses_last_week) * 100, 1)
    elif diagnoses_this_week > 0:
        diagnoses_change = 100.0
    
    users_change = 0
    
    return {
        "total_diagnoses": total_diagnoses,
        "diagnoses_change": diagnoses_change,
        "disease_types": 23,
        "accuracy": accuracy,
        "dataset": "皮肤病变数据集",
        "total_users": total_users,
        "users_change": users_change
    }


@app.get("/api/system/dashboard/stats")
async def get_dashboard_stats():
    return get_db_stats()



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
