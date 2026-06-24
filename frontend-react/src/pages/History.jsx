import { useState, useEffect } from 'react'
import { Card, Typography, Row, Col, Table, Tag, Button, Space, Empty, Modal, message, Statistic } from 'antd'
import { DeleteOutlined, ExportOutlined, MedicineBoxOutlined, LoadingOutlined, ExclamationCircleOutlined } from '@ant-design/icons'

const { Title, Text } = Typography
const { confirm } = Modal

export default function History() {
  const [records, setRecords] = useState([])
  const [loading, setLoading] = useState(false)
  const [pagination, setPagination] = useState({ current: 1, pageSize: 10, total: 0 })

  const fetchHistory = async (page = 1) => {
    setLoading(true)
    try {
      const response = await fetch(`http://localhost:8000/api/history?page=${page}&size=10`)
      const data = await response.json()
      setRecords(data.records || [])
      setPagination({
        current: data.page,
        pageSize: data.size,
        total: data.total
      })
    } catch (error) {
      console.error('获取历史记录失败:', error)
      message.error('获取历史记录失败')
    }
    setLoading(false)
  }

  useEffect(() => {
    fetchHistory()
  }, [])

  const handleDelete = (record) => {
    // 解析诊断结果
    let displayResult = ''
    if (typeof record.result === 'string') {
      displayResult = record.result
    } else if (record.result && record.result.classification) {
      const top1 = record.result.classification.top1 || {}
      const disease = top1.class || '未知'
      const prob = top1.probability ? (top1.probability * 100).toFixed(2) : '0'
      displayResult = `${disease} (${prob}%)`
    } else {
      displayResult = JSON.stringify(record.result)
    }
    
    confirm({
      title: '确定删除这条诊断记录?',
      icon: <ExclamationCircleOutlined />,
      content: `诊断疾病: ${displayResult} (${record.timestamp})`,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      async onOk() {
        try {
          const response = await fetch(`http://localhost:8000/api/history/${record.id}`, {
            method: 'DELETE'
          })
          if (response.ok) {
            message.success('删除成功')
            fetchHistory(pagination.current)
          } else {
            message.error('删除失败')
          }
        } catch (error) {
          message.error('删除失败')
        }
      },
    })
  }

  const handleDeleteAll = () => {
    confirm({
      title: '确定清空所有诊断记录?',
      icon: <ExclamationCircleOutlined />,
      content: '此操作不可恢复，请谨慎操作',
      okText: '清空',
      okType: 'danger',
      cancelText: '取消',
      async onOk() {
        try {
          // 逐条删除
          for (const record of records) {
            await fetch(`http://localhost:8000/api/history/${record.id}`, {
              method: 'DELETE'
            })
          }
          message.success('清空成功')
          fetchHistory(1)
        } catch (error) {
          message.error('清空失败')
        }
      },
    })
  }

  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 60,
    },
    {
      title: '图像',
      dataIndex: 'image_name',
      key: 'image_name',
      render: (text) => <MedicineBoxOutlined style={{ fontSize: 24, color: '#165DFF' }} />,
      width: 80,
    },
    {
      title: '诊断结果',
      dataIndex: 'result',
      key: 'result',
      render: (text, record) => {
        let diseaseName = '未知'
        // 兼容两种格式：1. classification对象（诊断分析页）, 2. 字符串（智能对话）
        if (record.result?.classification?.top1?.class) {
          diseaseName = record.result.classification.top1.class
        } else if (record.result?.disease_name) {
          diseaseName = record.result.disease_name
        } else if (typeof text === 'string') {
          // 智能对话格式：提取疾病名称
          const match = text.match(/^([^()]+)/)
          diseaseName = match ? match[0].trim() : text
        }
        return <Text strong>{diseaseName}</Text>
      },
    },
    {
      title: '诊断时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space>
          <Button 
            type="text" 
            danger 
            icon={<DeleteOutlined />}
            onClick={() => handleDelete(record)}
          >
            删除
          </Button>
        </Space>
      ),
      width: 100,
    },
  ]

  const avgConfidence = records.length > 0
    ? (records.reduce((sum, r) => {
        // 兼容两种格式：1. classification对象（诊断分析页）, 2. 字符串（智能对话）
        const conf = r.result?.classification?.top1?.probability || 
                     r.result?.model_results?.confidence ||
                     (typeof r.result === 'string' ? 0.5 : 0)
        return sum + conf
      }, 0) / records.length * 100).toFixed(1)
    : 0

  const latestTime = records.length > 0 ? records[0]?.timestamp : '暂无'

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <Title level={2} style={{ margin: 0 }}>
            诊断历史
          </Title>
          <Text type="secondary">查看和管理您的历史诊断记录</Text>
        </div>
        {records.length > 0 && (
          <Button danger onClick={handleDeleteAll}>
            清空记录
          </Button>
        )}
      </div>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="总诊断次数"
              value={pagination.total}
              prefix={<MedicineBoxOutlined />}
              valueStyle={{ color: '#165DFF' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="最近诊断"
              value={latestTime}
              valueStyle={{ fontSize: 16 }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="平均置信度"
              value={avgConfidence}
              suffix="%"
              valueStyle={{ color: '#36CFC9' }}
            />
          </Card>
        </Col>
      </Row>

      <Card style={{ marginTop: 24 }}>
        {pagination.total > 0 ? (
          <Table
            columns={columns}
            dataSource={records}
            rowKey="id"
            loading={loading}
            pagination={{
              current: pagination.current,
              pageSize: pagination.pageSize,
              total: pagination.total,
              onChange: (page) => fetchHistory(page),
              showSizeChanger: false,
            }}
          />
        ) : (
          <Empty 
            description="暂无诊断记录" 
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          >
            <Button type="primary" href="/diagnosis">
              去诊断
            </Button>
          </Empty>
        )}
      </Card>
    </div>
  )
}
