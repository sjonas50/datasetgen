import React from 'react';
import { Typography } from 'antd';

const { Title, Text } = Typography;

interface PageHeaderProps {
  title: string;
  subtitle?: string;
}

export default function PageHeader({ title, subtitle }: PageHeaderProps) {
  return (
    <div style={{ 
      background: '#fff', 
      padding: '24px',
      borderBottom: '1px solid #f0f0f0',
      marginBottom: 24
    }}>
      <Title level={2} style={{ margin: 0 }}>{title}</Title>
      {subtitle && (
        <Text type="secondary" style={{ fontSize: 16 }}>{subtitle}</Text>
      )}
    </div>
  );
}