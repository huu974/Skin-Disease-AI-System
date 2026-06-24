import { useState, useRef, useEffect } from 'react'
import { Card, Input, Button, Typography, Space, Avatar, Collapse, Tag, Upload, Modal, List, Drawer, message } from 'antd'
import { SendOutlined, DeleteOutlined, RobotOutlined, UserOutlined, LoadingOutlined, ApiOutlined, BulbOutlined, PlusOutlined, PictureOutlined, HistoryOutlined, FileTextOutlined } from '@ant-design/icons'

const { Title, Text } = Typography
const { TextArea } = Input
const { Panel } = Collapse

const generateId = () => Date.now().toString(36) + Math.random().toString(36).substr(2)

// 获取所有对话历史
const getChatHistory = () => {
  const history = localStorage.getItem('chat_history')
  if (history) {
    try {
      return JSON.parse(history)
    } catch {
      return []
    }
  }
  return []
}

// 保存对话到历史记录
const saveToHistory = (messages, title) => {
  const history = getChatHistory()
  const newChat = {
    id: generateId(),
    title: title || messages[0]?.content?.slice(0, 30) || '新对话',
    messages: messages,
    timestamp: new Date().toISOString()
  }
  history.unshift(newChat)
  // 只保留最近20条对话
  const trimmed = history.slice(0, 20)
  localStorage.setItem('chat_history', JSON.stringify(trimmed))
  return trimmed
}

export default function Chat() {
  // 从 localStorage 读取当前对话
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem('chat_messages')
    if (saved) {
      try {
        return JSON.parse(saved)
      } catch {
        return [{ role: 'assistant', content: '🏥 您好！我是皮肤病智能诊断助手。本系统仅提供辅助建议，不能替代医生诊断。请描述您遇到的皮肤问题？', thinking: [], tools: [] }]
      }
    }
    return [{ role: 'assistant', content: '🏥 您好！我是皮肤病智能诊断助手。本系统仅提供辅助建议，不能替代医生诊断。请描述您遇到的皮肤问题？', thinking: [], tools: [] }]
  })
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [images, setImages] = useState([])
  const [previewImage, setPreviewImage] = useState(null)
  const [historyVisible, setHistoryVisible] = useState(false)
  const [chatHistory, setChatHistory] = useState([])
  const messagesEndRef = useRef(null)

  // 加载对话历史，刷新页面时保留对话内容
  useEffect(() => {
    // 如果localStorage中没有对话内容，初始化
    if (!localStorage.getItem('chat_messages')) {
      const initialMessage = [{ role: 'assistant', content: '🏥 您好！我是皮肤病智能诊断助手。本系统仅提供辅助建议，不能替代医生诊断。请描述您遇到的皮肤问题？', thinking: [], tools: [] }]
      localStorage.setItem('chat_messages', JSON.stringify(initialMessage))
      setMessages(initialMessage)
    }
    
    // 如果没有session_id，生成新的
    if (!localStorage.getItem('chat_session_id')) {
      const newSessionId = 'session_' + Date.now()
      localStorage.setItem('chat_session_id', newSessionId)
    }
    
    setChatHistory(getChatHistory())
  }, [])

  // 保存对话历史到 localStorage
  useEffect(() => {
    localStorage.setItem('chat_messages', JSON.stringify(messages))
  }, [messages])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleImageUpload = (file) => {
    const isImage = file.type.startsWith('image/')
    if (!isImage) {
      return false
    }
    if (images.length >= 4) {
      return false
    }
    setImages(prev => [...prev, file])
    return false
  }

  const removeImage = (index) => {
    setImages(prev => prev.filter((_, i) => i !== index))
  }

  const handleSend = async () => {
    if (!input.trim() && images.length === 0) return

    // 生成或获取当前会话ID
    let sessionId = localStorage.getItem('chat_session_id')
    if (!sessionId) {
      sessionId = 'session_' + Date.now()
      localStorage.setItem('chat_session_id', sessionId)
    }

    const formData = new FormData()
    formData.append('message', input)
    formData.append('session_id', sessionId)
    
    images.forEach(img => {
      formData.append('images', img)
    })

    const userMsg = { 
      role: 'user', 
      content: input || '[图片]', 
      images: images.map(img => URL.createObjectURL(img)),
      thinking: [], 
      tools: [],
      taskType: null
    }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setImages([])
    setLoading(true)

    // 添加空的AI消息，显示正在思考
    const userImageUrls = images.map(img => URL.createObjectURL(img))
    setMessages((prev) => [...prev, { 
      role: 'assistant', 
      content: '', 
      thinking: [], 
      tools: [],
      isThinking: true,
      hasImages: images.length > 0,
      taskType: null
    }])

    try {
      const response = await fetch('http://localhost:8000/api/chat/stream', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error('请求失败')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let fullContent = ''
      let thinking = []
      let tools = []

      // 不再重复添加消息，使用已有的正在思考的消息

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.type === 'thinking') {
                thinking = data.data
                const taskType = data.task_type || 'unknown'
                setMessages((prev) => {
                  const newMsgs = [...prev]
                  const lastMsg = newMsgs[newMsgs.length - 1]
                  if (lastMsg.role === 'assistant') {
                    lastMsg.thinking = thinking
                    lastMsg.taskType = taskType
                    lastMsg.isThinking = false // 开始收到思考过程
                    lastMsg.content = '' // 清空"正在思考中..."
                  }
                  return newMsgs
                })
              } else if (data.type === 'tools') {
                tools = data.data
                setMessages((prev) => {
                  const newMsgs = [...prev]
                  const lastMsg = newMsgs[newMsgs.length - 1]
                  if (lastMsg.role === 'assistant') {
                    lastMsg.tools = tools
                  }
                  return newMsgs
                })
              } else if (data.type === 'content') {
                fullContent += data.data
                setMessages((prev) => {
                  const newMsgs = [...prev]
                  const lastMsg = newMsgs[newMsgs.length - 1]
                  if (lastMsg.role === 'assistant') {
                    lastMsg.content = fullContent
                  }
                  return newMsgs
                })
              }
            } catch (e) {
              // ignore parse errors
            }
          }
        }
      }
    } catch (error) {
      setMessages((prev) => [...prev, { role: 'assistant', content: '抱歉，发生了错误，请稍后重试。', thinking: [], tools: [] }])
    }

    setLoading(false)
  }

  const handleClear = async () => {
    // 如果当前对话有内容，先保存到历史
    if (messages.length > 1) {
      const updatedHistory = saveToHistory(messages)
      setChatHistory(updatedHistory)
    }
    const newMessages = [{ role: 'assistant', content: '🏥 您好！我是皮肤病智能诊断助手。本系统仅提供辅助建议，不能替代医生诊断。请描述您遇到的皮肤问题？', thinking: [], tools: [] }]
    setMessages(newMessages)
    setImages([])
    localStorage.setItem('chat_messages', JSON.stringify(newMessages))
    
    // 先保存旧的session_id用于删除
    const oldSessionId = localStorage.getItem('chat_session_id') || 'default'
    
    // 生成新的session_id
    const newSessionId = 'session_' + Date.now()
    localStorage.setItem('chat_session_id', newSessionId)
    
    // 调用后端API清空旧session的对话历史
    try {
      await fetch(`http://localhost:8000/api/chat/history?session_id=${oldSessionId}`, {
        method: 'DELETE'
      })
      message.success('对话已清空，开始新会话')
    } catch (e) {
      console.error('清空对话历史失败', e)
    }
  }   //AI辅助生成: 豆包，2026-04-04

  const handleNewChat = () => {
    handleClear()
  }

  // 加载历史对话
  const loadChat = (chat) => {
    setMessages(chat.messages)
    localStorage.setItem('chat_messages', JSON.stringify(chat.messages))
    setHistoryVisible(false)
  }

  // 删除历史对话
  const deleteChat = (id, e) => {
    e.stopPropagation()
    const updated = chatHistory.filter(c => c.id !== id)
    setChatHistory(updated)
    localStorage.setItem('chat_history', JSON.stringify(updated))
  }

  return (
    <div>
      <div style={{ 
        background: '#fff7e6', 
        border: '1px solid #ffd591', 
        borderRadius: 8, 
        padding: '12px 16px',
        marginBottom: 16,
        display: 'flex',
        alignItems: 'center',
        gap: 8
      }}>
        <span style={{ fontSize: 18 }}>⚠️</span>
        <div>
          <Text strong style={{ color: '#d46b08' }}>免责声明</Text>
          <Text style={{ color: '#8c8c8c', fontSize: 12, display: 'block' }}>
            本系统仅提供辅助诊断建议，不能替代专业医生诊断。如有严重症状，请及时就医。
          </Text>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <div>
          <Title level={2} style={{ margin: 0 }}>
            智能对话
          </Title>
          <Text type="secondary">皮诊智核 · 皮肤病变智能诊断系统</Text>
        </div>
        <Space>
          <Button 
            icon={<HistoryOutlined />} 
            onClick={() => setHistoryVisible(true)}
          >
            历史记录
          </Button>
          <Button 
            type="primary" 
            icon={<PlusOutlined />} 
            onClick={handleNewChat}
            style={{
              background: 'linear-gradient(135deg, #165DFF 0%, #36CFC9 100%)',
              border: 'none',
            }}
          >
            新建对话
          </Button>
        </Space>
      </div>

      {/* 历史记录抽屉 */}
      <Drawer
        title="对话历史"
        placement="left"
        onClose={() => setHistoryVisible(false)}
        open={historyVisible}
        width={350}
      >
        {chatHistory.length === 0 ? (
          <Text type="secondary">暂无历史对话</Text>
        ) : (
          <List
            dataSource={chatHistory}
            renderItem={(item) => (
              <List.Item
                style={{ cursor: 'pointer', padding: '12px' }}
                onClick={() => loadChat(item)}
                actions={[
                  <Button 
                    key="delete" 
                    type="text" 
                    danger 
                    size="small"
                    onClick={(e) => deleteChat(item.id, e)}
                  >
                    删除
                  </Button>
                ]}
              >
                <List.Item.Meta
                  avatar={<FileTextOutlined style={{ fontSize: 24, color: '#165DFF' }} />}
                  title={item.title}
                  description={new Date(item.timestamp).toLocaleString('zh-CN')}
                />
              </List.Item>
            )}
          />
        )}
      </Drawer>

      <div style={{ marginTop: 16 }}>
        <Card
          bodyStyle={{
            padding: 0,
            height: 'calc(100vh - 300px)',
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <div style={{ flex: 1, padding: 16, overflowY: 'auto' }}>
            {messages.map((msg, index) => (
              <div
                key={index}
                style={{
                  display: 'flex',
                  justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                  marginBottom: 12,
                }}
              >
                <div
                  style={{
                    maxWidth: '85%',
                    display: 'flex',
                    flexDirection: 'row-reverse',
                    alignItems: 'flex-start',
                    gap: 8,
                  }}
                >
                  <Avatar
                    icon={msg.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
                    style={{
                      backgroundColor: msg.role === 'user' ? '#165DFF' : '#36CFC9',
                    }}
                  />
                  <div
                    style={{
                      background: msg.role === 'user' ? '#165DFF' : '#F8FAFC',
                      color: msg.role === 'user' ? 'white' : '#1E293B',
                      padding: '12px 16px',
                      borderRadius: msg.role === 'user' ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
                      border: msg.role === 'user' ? 'none' : '1px solid #E2E8F0',
                      width: '100%',
                    }}
                  >
                    <Text strong style={{ color: msg.role === 'user' ? 'white' : '#165DFF' }}>
                      {msg.role === 'user' ? '您' : 'AI医生'}
                    </Text>
                    
                    {/* 用户上传的图片 */}
                    {msg.images && msg.images.length > 0 && (
                      <div style={{ marginTop: 8, display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                        {msg.images.map((img, i) => (
                          <img 
                            key={i}
                            src={img} 
                            alt="上传图片"
                            style={{ 
                              width: 80, 
                              height: 80, 
                              objectFit: 'cover',
                              borderRadius: 8,
                              cursor: 'pointer'
                            }}
                            onClick={() => setPreviewImage(img)}
                          />
                        ))}
                      </div>
                    )}

                    {/* 正在思考提示 - 无图片时 */}
                    {msg.isThinking && !msg.hasImages && (
                      <div style={{ marginTop: 16, padding: 16, background: '#f0f5ff', borderRadius: 8, border: '1px solid #d6e4ff' }}>
                        <Space>
                          <LoadingOutlined spin style={{ color: '#1890ff' }} />
                          <Text strong style={{ color: '#1890ff' }}>AI 正在思考中...</Text>
                        </Space>
                        <div style={{ marginTop: 8 }}>
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            {msg.taskType === 'symptom_only' ? '正在分析您描述的情况...' : 
                             msg.taskType === 'follow_up' ? '正在思考您的回复...' :
                             '正在处理您的请求...'}
                          </Text>
                        </div>
                      </div>
                    )}

                    {/* 有图时显示推理过程 */}
                    {msg.isThinking && msg.hasImages && (
                      <div style={{ marginTop: 16, padding: 16, background: '#f0f5ff', borderRadius: 8, border: '1px solid #d6e4ff' }}>
                        <Space>
                          <LoadingOutlined spin style={{ color: '#1890ff' }} />
                          <Text strong style={{ color: '#1890ff' }}>AI 正在思考中...</Text>
                        </Space>
                        <div style={{ marginTop: 8 }}>
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            {msg.taskType === 'image_only' ? '正在分析您上传的图像...' : 
                             msg.taskType === 'symptom_and_image' ? '正在分析您的描述和上传的图像...' :
                             '正在处理您的请求...'}
                          </Text>
                        </div>
                      </div>
                    )}

                    {/* 推理过程 - 有thinking数据时显示（不限图片） */}
                    {msg.thinking && msg.thinking.length > 0 && (
                      <Collapse 
                        ghost 
                        style={{ marginTop: 8, background: '#f0f5ff', borderRadius: 8 }}
                        defaultActiveKey={msg.taskType && msg.taskType !== 'symptom_and_image' ? [] : ['thinking']}
                        items={[
                          {
                            key: 'thinking',
                            label: (
                              <Space>
                                <BulbOutlined style={{ color: '#faad14' }} />
                                <Text strong style={{ color: '#165DFF' }}>
                                  {msg.taskType === 'image_only' ? '🔍 图像诊断过程' : 
                                   msg.taskType === 'symptom_only' ? '🩺 症状分析过程' :
                                   msg.taskType === 'symptom_and_image' ? '🧠 综合诊断过程' :
                                   '💡 AI推理过程'}
                                </Text>
                                <Tag color="blue">{msg.thinking.length} 步</Tag>
                              </Space>
                            ),
                            children: (
                              <div>
                                {msg.thinking.map((step, i) => (
                                  <div 
                                    key={i} 
                                    style={{ 
                                      marginBottom: 12, 
                                      padding: 12, 
                                      background: '#fff', 
                                      borderRadius: 8,
                                      borderLeft: step.status === 'calling' ? '3px solid #1890ff' : 
                                                 step.status === 'completed' ? '3px solid #52c41a' :
                                                 step.status === 'waiting' ? '3px solid #faad14' : '3px solid #d9d9d9'
                                    }}
                                  >
                                    <Space>
                                      <Text strong style={{ color: '#1890ff' }}>{step.stage}</Text>
                                      <Tag color={step.status === 'completed' ? 'green' : 
                                                   step.status === 'calling' ? 'blue' : 
                                                   step.status === 'waiting' ? 'orange' : 'default'}>
                                        {step.status === 'calling' ? '调用中' : 
                                         step.status === 'completed' ? '完成' :
                                         step.status === 'waiting' ? '等待中' :
                                         step.status === 'thinking' ? '思考中' : step.status}
                                      </Tag>
                                      {step.tool && <Tag color="purple">{step.tool}</Tag>}
                                    </Space>
                                    <div style={{ marginTop: 8 }}>
                                      <Text style={{ display: 'block', fontWeight: 500 }}>
                                        💡 {step.thought}
                                      </Text>
                                      <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 4 }}>
                                        📝 推理: {step.reasoning}
                                      </Text>
                                      <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 2 }}>
                                        ✓ 决策: {step.decision}
                                      </Text>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            ),
                          },
                        ]}
                      />
                    )}

                    {/* 工具调用 */}
                    {msg.tools && msg.tools.length > 0 && (
                      <Collapse 
                        ghost 
                        style={{ marginTop: 8, background: '#f0f5ff', borderRadius: 8 }}
                        items={[
                          {
                            key: 'tools',
                            label: (
                              <Space>
                                <ApiOutlined style={{ color: '#165DFF' }} />
                                <Text strong style={{ color: '#165DFF' }}>🔧 工具调用</Text>
                                <Tag color="blue">{msg.tools.length} 个</Tag>
                              </Space>
                            ),
                            children: (
                              <div>
                                {msg.tools.map((tool, i) => (
                                  <div 
                                    key={i} 
                                    style={{ 
                                      marginBottom: 8, 
                                      padding: '8px 12px', 
                                      background: '#fff', 
                                      borderRadius: 6,
                                      border: '1px solid #e8e8e8'
                                    }}
                                  >
                                    <Space>
                                      <Tag color="green">{tool.tool}</Tag>
                                      <Text type="secondary">→</Text>
                                      <Tag>{tool.action}</Tag>
                                    </Space>
                                    <div style={{ marginTop: 4 }}>
                                      <Text type="secondary" style={{ fontSize: 12 }}>
                                        输入: {tool.input} | 输出: {tool.output}
                                      </Text>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            ),
                          },
                        ]}
                      />
                    )}

                    <div style={{ whiteSpace: 'pre-wrap', marginTop: 12 }}>
                      {loading && index === messages.length - 1 && !msg.isThinking ? (
                        <span style={{ color: '#999' }}>正在思考中... <LoadingOutlined /></span>
                      ) : (
                        msg.content
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </Card>

        {/* 图片预览 Modal */}
        <Modal
          open={!!previewImage}
          footer={null}
          onCancel={() => setPreviewImage(null)}
          width={800}
        >
          <img 
            src={previewImage} 
            alt="预览" 
            style={{ width: '100%', maxHeight: '70vh', objectFit: 'contain' }} 
          />
        </Modal>

        <Card bodyStyle={{ padding: 12 }}>
          {/* 已选图片展示 */}
          {images.length > 0 && (
            <div style={{ marginBottom: 8, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {images.map((img, i) => (
                <div key={i} style={{ position: 'relative' }}>
                  <img 
                    src={URL.createObjectURL(img)} 
                    alt="待上传"
                    style={{ 
                      width: 60, 
                      height: 60, 
                      objectFit: 'cover',
                      borderRadius: 8,
                      border: '2px solid #165DFF'
                    }}
                  />
                  <Button
                    type="circle"
                    size="small"
                    danger
                    icon={<DeleteOutlined />}
                    onClick={() => removeImage(i)}
                    style={{ 
                      position: 'absolute', 
                      top: -8, 
                      right: -8,
                      width: 20,
                      height: 20
                    }}
                  />
                </div>
              ))}
            </div>
          )}
          
          <Space.Compact style={{ width: '100%' }}>
            <Upload
              accept="image/*"
              showUploadList={false}
              beforeUpload={handleImageUpload}
              disabled={images.length >= 4}
            >
              <Button 
                icon={<PictureOutlined />}
                style={{ borderRadius: '8px 0 0 8px' }}
                disabled={images.length >= 4}
              >
                {images.length > 0 ? `${images.length}/4` : '图片'}
              </Button>
            </Upload>
            <TextArea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onPressEnter={(e) => {
                if (!e.shiftKey) {
                  e.preventDefault()
                  handleSend()
                }
              }}
              placeholder="请描述您的皮肤问题...（可同时上传图片）"
              autoSize={{ minRows: 1, maxRows: 4 }}
              style={{ flex: 1 }}
            />
            <Button
              type="primary"
              icon={<SendOutlined />}
              onClick={handleSend}
              loading={loading}
              style={{
                background: 'linear-gradient(135deg, #165DFF 0%, #36CFC9 100%)',
                border: 'none',
                borderRadius: '0 8px 8px 0',
              }}
            >
              发送
            </Button>
          </Space.Compact>
          <div style={{ marginTop: 8, display: 'flex', justifyContent: 'space-between' }}>
            <Button
              type="text"
              icon={<DeleteOutlined />}
              onClick={handleClear}
            >
              🗑️ 清空对话
            </Button>
            <Text type="secondary" style={{ fontSize: 12 }}>
              支持同时发送文字和最多4张图片
            </Text>
          </div>
        </Card>
      </div>
    </div>
  )
}
