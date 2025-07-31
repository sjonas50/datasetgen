import React from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import { Layout as AntLayout, Menu } from 'antd';
import {
  DatabaseOutlined,
  ThunderboltOutlined,
  ApiOutlined,
  DashboardOutlined,
  LineChartOutlined,
  SettingOutlined,
  LogoutOutlined,
} from '@ant-design/icons';

const { Header, Sider, Content } = AntLayout;

interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const router = useRouter();
  const selectedKey = router.pathname;

  const menuItems = [
    {
      key: '/datasets',
      icon: <DatabaseOutlined />,
      label: <Link href="/datasets">Datasets</Link>,
    },
    {
      key: '/pipelines',
      icon: <ThunderboltOutlined />,
      label: <Link href="/pipelines">Pipelines</Link>,
    },
    {
      key: '/pipeline-builder',
      icon: <ApiOutlined />,
      label: <Link href="/pipeline-builder">Pipeline Builder</Link>,
    },
    {
      key: '/monitoring',
      icon: <LineChartOutlined />,
      label: <Link href="/monitoring">Monitoring</Link>,
    },
    {
      key: '/connectors',
      icon: <DashboardOutlined />,
      label: <Link href="/connectors">Connectors</Link>,
    },
    {
      key: '/settings',
      icon: <SettingOutlined />,
      label: <Link href="/settings">Settings</Link>,
    },
  ];

  const handleLogout = () => {
    localStorage.removeItem('token');
    router.push('/login');
  };

  return (
    <AntLayout style={{ minHeight: '100vh' }}>
      <Sider
        theme="dark"
        breakpoint="lg"
        collapsedWidth="0"
      >
        <div style={{
          height: 64,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#fff',
          fontSize: 18,
          fontWeight: 600,
        }}>
          DatasetGen
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[selectedKey]}
          items={menuItems}
        />
        <div style={{
          position: 'absolute',
          bottom: 0,
          width: '100%',
          padding: '16px',
        }}>
          <Menu
            theme="dark"
            mode="inline"
            items={[
              {
                key: 'logout',
                icon: <LogoutOutlined />,
                label: 'Logout',
                onClick: handleLogout,
              },
            ]}
          />
        </div>
      </Sider>
      <AntLayout>
        <Content style={{ background: '#f0f2f5' }}>
          {children}
        </Content>
      </AntLayout>
    </AntLayout>
  );
}