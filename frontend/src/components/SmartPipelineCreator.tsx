import React, { useState } from 'react';
import { Card, Button, Steps, Typography, Space, Alert, Spin, Result, Tag, Progress, Divider, Modal } from 'antd';
import {
  RocketOutlined,
  CheckCircleOutlined,
  LoadingOutlined,
  FileSearchOutlined,
  SafetyCertificateOutlined,
  ApiOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';
import api from '@/services/api';
import PipelineExecutionStatus from './PipelineExecutionStatus';
import ExecutionMonitor from './ExecutionMonitor';

const { Title, Text, Paragraph } = Typography;
const { Step } = Steps;

interface SmartPipelineCreatorProps {
  dataset: any;
  onComplete?: (pipeline: any) => void;
  onCancel?: () => void;
}

export default function SmartPipelineCreator({ dataset, onComplete, onCancel }: SmartPipelineCreatorProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<any>(null);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pipelineId, setPipelineId] = useState<string | null>(null);
  const [executionId, setExecutionId] = useState<string | null>(null);
  const [showExecutionStatus, setShowExecutionStatus] = useState(false);

  const analyzeDataset = async () => {
    setAnalyzing(true);
    setError(null);
    
    try {
      const response = await api.post(`/api/v1/datasets/${dataset.id}/analyze-pipeline`);
      setAnalysis(response.data);
      setCurrentStep(1);
      
      // If a pipeline was auto-created, save the ID
      if (response.data.created_pipeline_id) {
        setPipelineId(response.data.created_pipeline_id);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to analyze dataset');
    } finally {
      setAnalyzing(false);
    }
  };

  const executePipeline = async () => {
    if (!pipelineId) return;
    
    setCreating(true);
    setError(null);
    
    try {
      const response = await api.post(`/api/v1/pipelines/${pipelineId}/execute`);
      setExecutionId(response.data.id);
      setShowExecutionStatus(true);
      // Don't advance to step 2 immediately - let user see execution status
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to execute pipeline');
    } finally {
      setCreating(false);
    }
  };

  const getStepIcon = (step: any) => {
    const iconMap: { [key: string]: React.ReactNode } = {
      'quality_validation': <CheckCircleOutlined style={{ fontSize: 20 }} />,
      'pii_detection': <SafetyCertificateOutlined style={{ fontSize: 20 }} />,
      'data_cleaning': <FileSearchOutlined style={{ fontSize: 20 }} />,
      'data_transformation': <ApiOutlined style={{ fontSize: 20 }} />,
    };
    return iconMap[step.type] || <ThunderboltOutlined style={{ fontSize: 20 }} />;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return '#52c41a';
    if (confidence >= 0.6) return '#faad14';
    return '#ff4d4f';
  };

  return (
    <div style={{ maxWidth: 800, margin: '0 auto' }}>
      <Steps current={currentStep} style={{ marginBottom: 32 }}>
        <Step title="Analyze Data" description="AI analyzes your files" />
        <Step title="Review & Create" description="Review suggested pipeline" />
        <Step title="Process Data" description="Transform your data" />
      </Steps>

      {currentStep === 0 && (
        <Card>
          <div style={{ textAlign: 'center', padding: '40px 20px' }}>
            <RocketOutlined style={{ fontSize: 64, color: '#1890ff', marginBottom: 24 }} />
            <Title level={3}>Let AI Create Your Pipeline</Title>
            <Paragraph style={{ fontSize: 16, marginBottom: 32 }}>
              Our AI will analyze your dataset "{dataset.name}" and automatically create
              the best data processing pipeline for your needs.
            </Paragraph>
            
            <Space direction="vertical" size="large" style={{ width: '100%', maxWidth: 400 }}>
              <Alert
                message="What will happen:"
                description={
                  <ul style={{ textAlign: 'left', marginTop: 8 }}>
                    <li>AI will examine your uploaded files</li>
                    <li>Detect data types and potential issues</li>
                    <li>Check for sensitive information (PII)</li>
                    <li>Create a custom processing pipeline</li>
                    <li>Generate training datasets (Q&A, Classification, NER, etc.)</li>
                  </ul>
                }
                type="info"
                showIcon
              />
              
              <Button
                type="primary"
                size="large"
                icon={analyzing ? <LoadingOutlined /> : <RocketOutlined />}
                onClick={analyzeDataset}
                loading={analyzing}
                style={{ minWidth: 200 }}
              >
                {analyzing ? 'Analyzing...' : 'Start AI Analysis'}
              </Button>
              
              {error && (
                <Alert message={error} type="error" showIcon closable />
              )}
            </Space>
          </div>
        </Card>
      )}

      {currentStep === 1 && analysis && (
        <Card>
          <Title level={3}>AI Analysis Complete!</Title>
          
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <div>
              <Text strong>Data Type Detected: </Text>
              <Tag color="blue" style={{ fontSize: 14 }}>
                {analysis.analysis.data_type?.toUpperCase()}
              </Tag>
            </div>

            <div>
              <Text strong>Confidence Score: </Text>
              <Progress
                percent={Math.round((analysis.confidence_score || 0) * 100)}
                strokeColor={getConfidenceColor(analysis.confidence_score || 0)}
                style={{ maxWidth: 300 }}
              />
            </div>

            {analysis.analysis.contains_pii && (
              <Alert
                message="Sensitive Data Detected"
                description={`PII types found: ${analysis.analysis.pii_types?.join(', ') || 'Various'}`}
                type="warning"
                showIcon
              />
            )}

            <div>
              <Title level={4}>Suggested Pipeline Steps:</Title>
              {analysis.suggested_pipeline.steps.map((step: any, idx: number) => (
                <Card
                  key={idx}
                  size="small"
                  style={{ marginBottom: 12 }}
                  bordered={false}
                  bodyStyle={{ background: '#f5f5f5' }}
                >
                  <Space>
                    {getStepIcon(step)}
                    <div>
                      <Text strong>{step.name}</Text>
                      <br />
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {analysis.analysis.recommended_steps?.[idx]?.reason || 'Process data'}
                      </Text>
                    </div>
                  </Space>
                </Card>
              ))}
            </div>

            {analysis.analysis.insights && (
              <Alert
                message="AI Insights"
                description={analysis.analysis.insights}
                type="info"
                showIcon
              />
            )}

            <Divider />

            <Space size="large">
              <Button
                type="primary"
                size="large"
                icon={creating ? <LoadingOutlined /> : <ThunderboltOutlined />}
                onClick={executePipeline}
                loading={creating}
                disabled={!pipelineId}
              >
                {creating ? 'Processing...' : 'Run Pipeline Now'}
              </Button>
              
              <Button size="large" onClick={onCancel}>
                Customize Pipeline
              </Button>
            </Space>

            {error && (
              <Alert message={error} type="error" showIcon closable />
            )}
          </Space>
        </Card>
      )}

      <Modal
        title="Pipeline Execution Progress"
        open={showExecutionStatus}
        width={800}
        footer={null}
        onCancel={() => {
          setShowExecutionStatus(false);
          setCurrentStep(2);
          onComplete?.(analysis.suggested_pipeline);
        }}
      >
        {showExecutionStatus && executionId && pipelineId && (
          <ExecutionMonitor
            executionId={executionId}
            pipelineId={pipelineId}
            onComplete={(execution) => {
              if (execution.status === 'completed') {
                setCurrentStep(2);
                onComplete?.(analysis.suggested_pipeline);
              }
            }}
          />
        )}
      </Modal>

      {currentStep === 2 && (
        <Card>
          <Result
            status="success"
            title="Pipeline Executed Successfully!"
            subTitle="Your data has been processed and is ready for use."
            extra={[
              <Button type="primary" key="view" onClick={() => onComplete?.(analysis.suggested_pipeline)}>
                View Results
              </Button>,
              <Button key="new" onClick={onCancel}>
                Process Another Dataset
              </Button>,
            ]}
          />
        </Card>
      )}
    </div>
  );
}