import { useState, useEffect } from 'react'
import { Card, Form, Input, Button, Tabs, Row, Col, message, Statistic, Table, Tag, Select } from 'antd'
import { UserOutlined, SaveOutlined, DeleteOutlined, DatabaseOutlined, TeamOutlined, HistoryOutlined } from '@ant-design/icons'

export default function Profile() {
  const [form] = Form.useForm()
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(false)
  const [diagnosisCount, setDiagnosisCount] = useState(0)
  const [chatCount, setChatCount] = useState(0)
  const [diagnosisRecords, setDiagnosisRecords] = useState([])

  const currentUser = localStorage.getItem('username')
  const isGuest = localStorage.getItem('isGuest') === 'true'

  useEffect(() => {
    if (!isGuest) {
      fetchUsers()
      fetchStatistics()
    }
  }, [isGuest])

  const fetchUsers = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/admin/users')
      const data = await response.json()
      setUsers(data.users || [])
    } catch (error) {
      console.error('获取用户列表失败', error)
    }
    setLoading(false)
  }

  const fetchStatistics = async () => {
    try {
      const username = localStorage.getItem('username')
      const response = await fetch(`http://localhost:8000/api/user/statistics?username=${encodeURIComponent(username)}`)
      const data = await response.json()
      if (response.ok) {
        setDiagnosisCount(data.diagnosis_count || 0)
        setChatCount(data.chat_count || 0)
        setDiagnosisRecords(data.records || [])
      }
    } catch (error) {
      console.error('获取统计数据失败', error)
    }
  }

  const handleSave = (values) => {
    if (isGuest) {
      message.warning('游客模式下无法保存信息')
      return
    }
    message.success('信息已保存!')
  }

  const handleClearData = () => {
    if (isGuest) {
      message.warning('游客模式下无法操作')
      return
    }
    message.success('数据已清空!')
  }

  const recordColumns = [
    {
      title: '时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (text) => text ? new Date(text).toLocaleString() : '-',
    },
    {
      title: '疾病类型',
      dataIndex: 'disease',
      key: 'disease',
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (val) => `${(val * 100).toFixed(1)}%`,
    },
  ]

  const userColumns = [
    {
      title: '用户名',
      dataIndex: 'username',
      key: 'username',
      render: (text) => <Tag color="blue">{text}</Tag>,
    },
    {
      title: '注册时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (text) => text ? new Date(text).toLocaleString() : '-',
    },
  ]

  return (
    <div>
      <UserOutlined style={{ fontSize: 24 }} />
      <div style={{ marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>用户中心</h2>
        <p style={{ color: '#64748B', margin: '4px 0 0 0' }}>管理您的个人信息和诊断记录</p>
        {isGuest && <Tag color="orange">游客模式</Tag>}
      </div>

      <Card>
        <Tabs
          defaultActiveKey="1"
          items={[
            {
              key: '1',
              label: (
                <span>
                  <UserOutlined />
                  基本信息
                </span>
              ),
              children: (
                <Form
                  form={form}
                  layout="vertical"
                  onFinish={handleSave}
                  initialValues={{
                    username: isGuest ? '游客' : (currentUser || ''),
                  }}
                >
                  <Row gutter={16}>
                    <Col xs={24} md={12}>
                      <Form.Item
                        name="username"
                        label="账号"
                      >
                        <Input 
                          placeholder="请输入账号" 
                          disabled={true}
                        />
                      </Form.Item>
                    </Col>
                  </Row>
                  {!isGuest && (
                    <Button 
                      type="primary" 
                      htmlType="submit" 
                      icon={<SaveOutlined />}
                    >
                      💾 保存修改
                    </Button>
                  )}
                </Form>
              ),
            },
            {
              key: '2',
              label: (
                <span>
                  <DatabaseOutlined />
                  数据管理
                </span>
              ),
              children: (
                <div>
                  <Row gutter={16}>
                    <Col xs={24} sm={12}>
                      <Card>
                        <Statistic
                          title="诊断记录"
                          value={diagnosisCount}
                          prefix={<DatabaseOutlined />}
                          valueStyle={{ color: '#165DFF' }}
                        />
                      </Card>
                    </Col>
                    <Col xs={24} sm={12}>
                      <Card>
                        <Statistic
                          title="对话记录"
                          value={chatCount}
                          prefix={<HistoryOutlined />}
                          valueStyle={{ color: '#36CFC9' }}
                        />
                      </Card>
                    </Col>
                  </Row>
                  <div style={{ marginTop: 24 }}>
                    <h4>诊断记录</h4>
                    <Table
                      dataSource={diagnosisRecords}
                      columns={recordColumns}
                      rowKey="id"
                      pagination={{ pageSize: 5 }}
                      locale={{ emptyText: '暂无诊断记录' }}
                    />
                  </div>
                  <Button 
                    danger 
                    icon={<DeleteOutlined />} 
                    onClick={handleClearData}
                    disabled={isGuest}
                    style={{ marginTop: 16 }}
                  >
                    🗑️ 清空数据
                  </Button>
                </div>
              ),
            },
            ...(!isGuest ? [{
              key: '3',
              label: (
                <span>
                  <TeamOutlined />
                  用户管理
                </span>
              ),
              children: (
                <div>
                  <div style={{ marginBottom: 16 }}>
                    <Button 
                      type="primary" 
                      icon={<TeamOutlined />} 
                      onClick={fetchUsers}
                      loading={loading}
                    >
                      刷新列表
                    </Button>
                    <span style={{ marginLeft: 16 }}>
                      总计注册用户: <Tag color="blue">{users.length}</Tag>
                    </span>
                  </div>
                  <Table
                    dataSource={users}
                    columns={userColumns}
                    rowKey="username"
                    loading={loading}
                    pagination={{ pageSize: 10 }}
                  />
                </div>
              ),
            }] : []),
          ]}
        />
      </Card>
    </div>
  )
}
