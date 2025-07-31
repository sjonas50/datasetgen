import { useState } from 'react';
import { Card, Button, Row, Col, Typography, Space, Statistic } from 'antd';
import {
  DatabaseOutlined,
  ApiOutlined,
  RocketOutlined,
  SettingOutlined,
  PlusOutlined,
  LineChartOutlined,
  CloudUploadOutlined,
} from '@ant-design/icons';
import Link from 'next/link';
import Layout from '@/components/Layout';

const { Title, Paragraph } = Typography;

export default function Home() {
  const features = [
    {
      icon: <DatabaseOutlined style={{ fontSize: '32px', color: '#2563eb' }} />,
      title: 'Multi-Source Data',
      description: 'Connect to databases, APIs, files, and streaming sources',
    },
    {
      icon: <ApiOutlined style={{ fontSize: '32px', color: '#2563eb' }} />,
      title: 'LLM-Powered Processing',
      description: 'Leverage GPT-4, Gemini, and Claude for intelligent data transformation',
    },
    {
      icon: <LineChartOutlined style={{ fontSize: '32px', color: '#2563eb' }} />,
      title: 'Visual Pipeline Builder',
      description: 'Create complex data pipelines with drag-and-drop simplicity',
    },
    {
      icon: <CloudUploadOutlined style={{ fontSize: '32px', color: '#2563eb' }} />,
      title: 'Automated Quality',
      description: 'AI-driven data validation and quality assessment',
    },
  ];

  const stats = [
    { title: 'Datasets Created', value: 0 },
    { title: 'Active Pipelines', value: 0 },
    { title: 'Data Processed', value: '0 GB' },
    { title: 'Time Saved', value: '0 hrs' },
  ];

  return (
    <Layout>
      <div style={{ padding: '50px' }}>
        {/* Hero Section */}
        <div style={{ textAlign: 'center', marginBottom: '60px' }}>
          <Title level={1}>
            Transform Raw Data into ML-Ready Datasets
          </Title>
          <Paragraph style={{ fontSize: '18px', color: '#64748b', maxWidth: '800px', margin: '0 auto' }}>
            An AI-first platform that automates dataset creation through intelligent processing, 
            multi-modal fusion, and enterprise-grade security.
          </Paragraph>
          <Space size="large" style={{ marginTop: '30px' }}>
            <Link href="/datasets">
              <Button type="primary" size="large" icon={<PlusOutlined />}>
                Create Dataset
              </Button>
            </Link>
            <Link href="/pipelines">
              <Button size="large" icon={<RocketOutlined />}>
                Build Pipeline
              </Button>
            </Link>
          </Space>
        </div>

        {/* Stats Section */}
        <Row gutter={[16, 16]} style={{ marginBottom: '60px' }}>
          {stats.map((stat, index) => (
            <Col xs={24} sm={12} md={6} key={index}>
              <Card>
                <Statistic title={stat.title} value={stat.value} />
              </Card>
            </Col>
          ))}
        </Row>

        {/* Features Section */}
        <Title level={2} style={{ textAlign: 'center', marginBottom: '40px' }}>
          Platform Features
        </Title>
        <Row gutter={[24, 24]}>
          {features.map((feature, index) => (
            <Col xs={24} sm={12} md={6} key={index}>
              <Card 
                hoverable
                style={{ height: '100%', textAlign: 'center' }}
              >
                <div style={{ marginBottom: '20px' }}>{feature.icon}</div>
                <Title level={4}>{feature.title}</Title>
                <Paragraph style={{ color: '#64748b' }}>
                  {feature.description}
                </Paragraph>
              </Card>
            </Col>
          ))}
        </Row>

        {/* Getting Started Section */}
        <Card style={{ marginTop: '60px', background: '#f8fafc' }}>
          <Title level={3}>Getting Started</Title>
          <Row gutter={[24, 24]}>
            <Col xs={24} md={8}>
              <Space direction="vertical">
                <Title level={5}>1. Connect Your Data</Title>
                <Paragraph>
                  Choose from multiple connectors to import data from databases, 
                  APIs, or upload files directly.
                </Paragraph>
              </Space>
            </Col>
            <Col xs={24} md={8}>
              <Space direction="vertical">
                <Title level={5}>2. Design Your Pipeline</Title>
                <Paragraph>
                  Use our visual builder to create data transformation pipelines 
                  with LLM-powered processing steps.
                </Paragraph>
              </Space>
            </Col>
            <Col xs={24} md={8}>
              <Space direction="vertical">
                <Title level={5}>3. Generate Datasets</Title>
                <Paragraph>
                  Execute pipelines to create high-quality, ML-ready datasets 
                  with automated validation.
                </Paragraph>
              </Space>
            </Col>
          </Row>
        </Card>
      </div>
    </Layout>
  );
}