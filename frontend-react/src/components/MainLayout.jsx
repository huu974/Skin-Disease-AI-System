import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { Layout, Menu, Button, Avatar, Dropdown, Space } from 'antd'
import {
  HomeOutlined,
  MedicineBoxOutlined,
  MessageOutlined,
  SafetyOutlined,
  HistoryOutlined,
  UserOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  LogoutOutlined,
  SettingOutlined,
} from '@ant-design/icons'

const { Sider, Content, Footer } = Layout

const menuItems = [
  { key: '/', icon: <HomeOutlined />, label: '首页' },
  { key: '/diagnosis', icon: <MedicineBoxOutlined />, label: '诊断分析' },
  { key: '/chat', icon: <MessageOutlined />, label: '智能对话' },
  { key: '/prevention', icon: <SafetyOutlined />, label: '预防建议' },
  { key: '/history', icon: <HistoryOutlined />, label: '历史记录' },
  { key: '/profile', icon: <UserOutlined />, label: '用户中心' },
]

export default function MainLayout({ children, onLogout }) {
  const [collapsed, setCollapsed] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()

  const userMenuItems = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: '个人中心',
      onClick: () => navigate('/profile'),
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: '系统设置',
    },
    { type: 'divider' },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: '退出登录',
      danger: true,
      onClick: onLogout,
    },
  ]

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        width={240}
        collapsedWidth={80}
        style={{
          background: 'white',
          borderRight: '1px solid #E2E8F0',
          position: 'fixed',
          height: '100vh',
          left: 0,
          top: 0,
          zIndex: 100,
        }}
      >
        <div
          style={{
            height: 64,
            display: 'flex',
            alignItems: 'center',
            justifyContent: collapsed ? 'center' : 'flex-start',
            padding: collapsed ? '0' : '0 24px',
            borderBottom: '1px solid #E2E8F0',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}
          >
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: 8,
                background: 'linear-gradient(135deg, #165DFF 0%, #36CFC9 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontWeight: 'bold',
                fontSize: 18,
              }}
            >
              肤
            </div>
            {!collapsed && (
              <span
                style={{
                  fontSize: 18,
                  fontWeight: 600,
                  color: '#165DFF',
                }}
              >
                智肤康
              </span>
            )}
          </div>
        </div>

        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          style={{
            border: 'none',
            padding: '12px 8px',
          }}
        />

        <div
          style={{
            position: 'absolute',
            bottom: 70,
            left: 0,
            right: 0,
            padding: '0 16px',
          }}
        >
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
            style={{
              width: '100%',
              justifyContent: 'center',
              color: '#64748B',
            }}
          >
            {!collapsed && '收起'}
          </Button>
        </div>
      </Sider>

      <Layout
        style={{
          marginLeft: collapsed ? 80 : 240,
          transition: 'margin-left 0.2s',
          background: '#F8FAFC',
        }}
      >
        <Content
          style={{
            padding: 24,
            minHeight: 'calc(100vh - 64px)',
            animation: 'fadeIn 0.3s ease-out',
          }}
        >
          {children}
        </Content>

        <Footer
          style={{
            background: 'white',
            borderTop: '1px solid #E2E8F0',
            padding: '12px 24px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <div style={{ color: '#64748B', fontSize: 13 }}>
            智肤康 · 皮肤疾病AI全流程辅助诊疗系统 v1.0.0
          </div>
          <Space>
            <Dropdown
              menu={{ items: userMenuItems }}
              placement="topRight"
              trigger={['click']}
            >
              <Space style={{ cursor: 'pointer' }}>
                <Avatar
                  style={{
                    backgroundColor: '#165DFF',
                    cursor: 'pointer',
                  }}
                  icon={<UserOutlined />}
                />
                <span style={{ color: '#1E293B' }}>用户</span>
              </Space>
            </Dropdown>
          </Space>
        </Footer>
      </Layout>
    </Layout>
  )
}
