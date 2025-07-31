import React, { useState } from 'react';
import { Card, Row, Col, Input, Tag, Typography, Tabs } from 'antd';
import {
  DatabaseOutlined,
  FunctionOutlined,
  RobotOutlined,
  FileImageOutlined,
  CheckCircleOutlined,
  FilePdfOutlined,
  MergeCellsOutlined,
  ScanOutlined,
  SafetyCertificateOutlined,
} from '@ant-design/icons';
import styles from './StepLibrary.module.css';

const { Search } = Input;
const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface StepDefinition {
  type: string;
  label: string;
  description: string;
  category: string;
  icon: React.ReactNode;
  defaultConfig?: any;
}

const stepDefinitions: StepDefinition[] = [
  // Data Sources
  {
    type: 'connector',
    label: 'CSV Connector',
    description: 'Load data from CSV files with intelligent schema detection',
    category: 'sources',
    icon: <DatabaseOutlined style={{ color: '#52c41a' }} />,
    defaultConfig: {
      connector_type: 'csv',
      llm_enhanced: true,
    },
  },
  {
    type: 'connector',
    label: 'JSON Connector',
    description: 'Import JSON data with nested structure support',
    category: 'sources',
    icon: <DatabaseOutlined style={{ color: '#52c41a' }} />,
    defaultConfig: {
      connector_type: 'json',
      llm_enhanced: true,
    },
  },
  
  // Transformations
  {
    type: 'transform',
    label: 'Filter',
    description: 'Filter rows based on conditions',
    category: 'transform',
    icon: <FunctionOutlined style={{ color: '#1890ff' }} />,
    defaultConfig: {
      type: 'filter',
    },
  },
  {
    type: 'transform',
    label: 'Aggregate',
    description: 'Group and aggregate data',
    category: 'transform',
    icon: <FunctionOutlined style={{ color: '#1890ff' }} />,
    defaultConfig: {
      type: 'aggregate',
    },
  },
  {
    type: 'transform',
    label: 'Custom Transform',
    description: 'Apply custom transformation logic',
    category: 'transform',
    icon: <FunctionOutlined style={{ color: '#1890ff' }} />,
    defaultConfig: {
      type: 'custom',
    },
  },
  
  // LLM Processing
  {
    type: 'llm_process',
    label: 'LLM Enrichment',
    description: 'Enrich data using LLM insights',
    category: 'llm',
    icon: <RobotOutlined style={{ color: '#722ed1' }} />,
    defaultConfig: {
      type: 'enrich',
      llm_provider: 'claude',
      llm_model: 'claude-sonnet-4-20250514',
    },
  },
  {
    type: 'llm_process',
    label: 'Generate Code',
    description: 'Generate transformation code with LLM',
    category: 'llm',
    icon: <RobotOutlined style={{ color: '#722ed1' }} />,
    defaultConfig: {
      type: 'generate_code',
      use_extended_thinking: true,
    },
  },
  {
    type: 'llm_process',
    label: 'Quality Assessment',
    description: 'Assess data quality using LLM',
    category: 'llm',
    icon: <RobotOutlined style={{ color: '#722ed1' }} />,
    defaultConfig: {
      type: 'validate',
    },
  },
  
  // Multi-Modal Processing
  {
    type: 'multimodal',
    label: 'Document Parser',
    description: 'Parse multiple document formats (PDF, images, etc.)',
    category: 'multimodal',
    icon: <FileImageOutlined style={{ color: '#fa8c16' }} />,
    defaultConfig: {
      type: 'document_parse',
      fusion_strategy: 'hybrid',
    },
  },
  {
    type: 'multimodal',
    label: 'PDF Table Extractor',
    description: 'Extract tables from PDFs using Claude Vision',
    category: 'multimodal',
    icon: <FilePdfOutlined style={{ color: '#fa8c16' }} />,
    defaultConfig: {
      type: 'pdf_table_extract',
      confidence_threshold: 0.85,
    },
  },
  {
    type: 'multimodal',
    label: 'Image Analyzer',
    description: 'Extract data from charts, dashboards, and images',
    category: 'multimodal',
    icon: <ScanOutlined style={{ color: '#fa8c16' }} />,
    defaultConfig: {
      type: 'image_data_extract',
      analysis_type: 'auto',
    },
  },
  {
    type: 'multimodal',
    label: 'Data Fusion',
    description: 'Intelligently fuse multi-modal data',
    category: 'multimodal',
    icon: <MergeCellsOutlined style={{ color: '#fa8c16' }} />,
    defaultConfig: {
      type: 'multimodal_fusion',
      fusion_strategy: 'hierarchical',
    },
  },
  
  // Validation
  {
    type: 'validate',
    label: 'Data Validator',
    description: 'Validate data against rules',
    category: 'validation',
    icon: <CheckCircleOutlined style={{ color: '#52c41a' }} />,
    defaultConfig: {
      fail_on_error: false,
    },
  },
  {
    type: 'validate',
    label: 'PII Detector',
    description: 'Detect personally identifiable information',
    category: 'validation',
    icon: <SafetyCertificateOutlined style={{ color: '#52c41a' }} />,
    defaultConfig: {
      type: 'pii_detection',
      mask_pii: true,
    },
  },
];

interface StepLibraryProps {
  onAddStep: (type: string, config: any) => void;
}

const StepLibrary: React.FC<StepLibraryProps> = ({ onAddStep }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');

  const filteredSteps = stepDefinitions.filter((step) => {
    const matchesSearch = step.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         step.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || step.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const categories = [
    { key: 'all', label: 'All Steps' },
    { key: 'sources', label: 'Data Sources' },
    { key: 'transform', label: 'Transformations' },
    { key: 'llm', label: 'LLM Processing' },
    { key: 'multimodal', label: 'Multi-Modal' },
    { key: 'validation', label: 'Validation' },
  ];

  return (
    <div className={styles.stepLibrary}>
      <Search
        placeholder="Search steps..."
        onChange={(e) => setSearchTerm(e.target.value)}
        style={{ marginBottom: 16 }}
      />
      
      <Tabs activeKey={selectedCategory} onChange={setSelectedCategory}>
        {categories.map((cat) => (
          <TabPane tab={cat.label} key={cat.key}>
            <Row gutter={[16, 16]}>
              {filteredSteps
                .filter((step) => cat.key === 'all' || step.category === cat.key)
                .map((step) => (
                  <Col span={12} key={`${step.type}_${step.label}`}>
                    <Card
                      hoverable
                      className={styles.stepCard}
                      onClick={() => onAddStep(step.type, step)}
                    >
                      <div className={styles.stepCardContent}>
                        <div className={styles.stepIcon}>{step.icon}</div>
                        <div className={styles.stepInfo}>
                          <Title level={5} style={{ margin: 0 }}>
                            {step.label}
                          </Title>
                          <Text type="secondary" className={styles.stepDescription}>
                            {step.description}
                          </Text>
                        </div>
                      </div>
                    </Card>
                  </Col>
                ))
              }
            </Row>
          </TabPane>
        ))}
      </Tabs>
    </div>
  );
};

export default StepLibrary;