import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Form, Input, Button, message, Card, Tabs, Divider } from 'antd'
import { UserOutlined, LockOutlined, MedicineBoxOutlined, TeamOutlined } from '@ant-design/icons'

export default function Login({ onLogin }) {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('login')
  const navigate = useNavigate()

  const handleLogin = async (values) => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `username=${encodeURIComponent(values.username)}&password=${encodeURIComponent(values.password)}`,
      })
      
      const data = await response.json()
      
      if (response.ok) {
        message.success('登录成功')
        localStorage.setItem('username', values.username)
        localStorage.setItem('token', data.token)
        onLogin()
        navigate('/')
      } else {
        message.error(data.detail || '用户名或密码错误')
      }
    } catch (error) {
      message.error('登录失败，请稍后重试')
    }
    setLoading(false)
  }

  const handleRegister = async (values) => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `username=${encodeURIComponent(values.username)}&password=${encodeURIComponent(values.password)}`,
      })
      
      const data = await response.json()
      
      if (response.ok) {
        message.success('注册成功，请登录')
        setActiveTab('login')
      } else {
        message.error(data.detail || '注册失败')
      }
    } catch (error) {
      message.error('注册失败，请稍后重试')
    }
    setLoading(false)
  }

  const handleGuestLogin = () => {
    message.success('游客登录成功')
    localStorage.setItem('username', '游客')
    localStorage.setItem('isGuest', 'true')
    onLogin()
    navigate('/')
  }

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #165DFF 0%, #36CFC9 100%)',
      }}
    >
      <Card
        style={{
          width: 400,
          borderRadius: 16,
          boxShadow: '0 8px 32px rgba(0,0,0,0.2)',
        }}
      >
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <div
            style={{
              width: 64,
              height: 64,
              borderRadius: 16,
              background: 'linear-gradient(135deg, #165DFF 0%, #36CFC9 100%)',
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: 16,
            }}
          >
            <MedicineBoxOutlined style={{ fontSize: 32, color: 'white' }} />
          </div>
          <h1 style={{ fontSize: 24, fontWeight: 600, color: '#1E293B', margin: 0 }}>
            智肤康
          </h1>
          <p style={{ color: '#64748B', marginTop: 8 }}>
            皮肤疾病AI全流程辅助诊疗系统
          </p>
        </div>

        <Button
          type="primary"
          block
          size="large"
          icon={<TeamOutlined />}
          onClick={handleGuestLogin}
          style={{
            height: 48,
            background: 'linear-gradient(135deg, #36CFC9 0%, #165DFF 100%)',
            border: 'none',
            fontWeight: 600,
            marginBottom: 16,
          }}
        >
          游客登录
        </Button>

        <Divider plain style={{ color: '#94A3B8' }}>或使用账号登录</Divider>

        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={[
            {
              key: 'login',
              label: '登录',
              children: (
                <Form
                  name="login"
                  onFinish={handleLogin}
                  size="large"
                >
                  <Form.Item
                    name="username"
                    rules={[{ required: true, message: '请输入用户名' }]}
                  >
                    <Input
                      prefix={<UserOutlined style={{ color: '#94A3B8' }} />}
                      placeholder="用户名"
                    />
                  </Form.Item>

                  <Form.Item
                    name="password"
                    rules={[{ required: true, message: '请输入密码' }]}
                  >
                    <Input.Password
                      prefix={<LockOutlined style={{ color: '#94A3B8' }} />}
                      placeholder="密码"
                    />
                  </Form.Item>

                  <Form.Item>
                    <Button
                      type="primary"
                      htmlType="submit"
                      loading={loading}
                      block
                      style={{
                        height: 44,
                        background: 'linear-gradient(135deg, #165DFF 0%, #36CFC9 100%)',
                        border: 'none',
                        fontWeight: 600,
                      }}
                    >
                      登录
                    </Button>
                  </Form.Item>
                </Form>
              ),
            },
            {
              key: 'register',
              label: '注册',
              children: (
                <Form
                  name="register"
                  onFinish={handleRegister}
                  size="large"
                >
                  <Form.Item
                    name="username"
                    rules={[
                      { required: true, message: '请输入账号' },
                      { min: 3, message: '账号至少3个字符' },
                    ]}
                  >
                    <Input
                      prefix={<UserOutlined style={{ color: '#94A3B8' }} />}
                      placeholder="账号"
                    />
                  </Form.Item>

                  <Form.Item
                    name="password"
                    rules={[
                      { required: true, message: '请输入密码' },
                      { min: 6, message: '密码至少6个字符' },
                    ]}
                  >
                    <Input.Password
                      prefix={<LockOutlined style={{ color: '#94A3B8' }} />}
                      placeholder="密码"
                    />
                  </Form.Item>

                  <Form.Item
                    name="confirmPassword"
                    dependencies={['password']}
                    rules={[
                      { required: true, message: '请确认密码' },
                      ({ getFieldValue }) => ({
                        validator(_, value) {
                          if (!value || getFieldValue('password') === value) {
                            return Promise.resolve()
                          }
                          return Promise.reject(new Error('两次输入的密码不一致'))
                        },
                      }),
                    ]}
                  >
                    <Input.Password
                      prefix={<LockOutlined style={{ color: '#94A3B8' }} />}
                      placeholder="确认密码"
                    />
                  </Form.Item>

                  <Form.Item>
                    <Button
                      type="primary"
                      htmlType="submit"
                      loading={loading}
                      block
                      style={{
                        height: 44,
                        background: 'linear-gradient(135deg, #165DFF 0%, #36CFC9 100%)',
                        border: 'none',
                        fontWeight: 600,
                      }}
                    >
                      注册
                    </Button>
                  </Form.Item>
                </Form>
              ),
            },
          ]}
        />
      </Card>
    </div>
  )
}
