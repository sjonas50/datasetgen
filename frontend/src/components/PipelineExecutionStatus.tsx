import React, { useState, useEffect } from 'react';
import { Modal, Steps, Progress, Alert, Spin, Typography, Space, Button, Card, Descriptions, Tag } from 'antd';
import {
  CheckCircleOutlined,
  ClockCircleOutlined,
  LoadingOutlined,
  CloseCircleOutlined,
  DownloadOutlined,
  EyeOutlined
} from '@ant-design/icons';
import api from '@/services/api';

const { Title, Text } = Typography;
const { Step } = Steps;

interface PipelineExecutionStatusProps {
  executionId: string;
  pipelineId: string;
  visible: boolean;
  onClose: () => void;
}

export default function PipelineExecutionStatus({ 
  executionId, 
  pipelineId, 
  visible, 
  onClose 
}: PipelineExecutionStatusProps) {
  const [execution, setExecution] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [polling, setPolling] = useState(true);

  useEffect(() => {
    if (visible && executionId) {
      loadExecutionStatus();
      const interval = setInterval(() => {
        if (polling) {
          loadExecutionStatus();
        }
      }, 2000); // Poll every 2 seconds

      return () => clearInterval(interval);
    }
  }, [visible, executionId, polling]);

  const loadExecutionStatus = async () => {
    try {
      const response = await api.get(`/api/v1/pipelines/${pipelineId}/executions`);
      const executions = response.data;
      const currentExecution = executions.find((e: any) => e.id === executionId);
      
      if (currentExecution) {
        setExecution(currentExecution);
        
        // Stop polling if execution is complete
        if (currentExecution.status === 'completed' || currentExecution.status === 'failed') {
          setPolling(false);
        }
      }
    } catch (error) {
      console.error('Failed to load execution status:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <ClockCircleOutlined />;
      case 'running':
        return <LoadingOutlined />;
      case 'completed':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'failed':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return null;
    }
  };

  const getStepStatus = (stepResult: any) => {
    if (!stepResult) return 'wait';
    if (stepResult.error) return 'error';
    return 'finish';
  };

  const renderResults = () => {
    if (!execution?.results) return null;

    return (
      <Card title="Processing Results" style={{ marginTop: 24 }}>
        <Space direction="vertical" style={{ width: '100%' }}>
          {Object.entries(execution.results).map(([stepName, result]: [string, any]) => (
            <Card key={stepName} size="small" type="inner" title={stepName}>
              {result.error ? (
                <Alert message="Error" description={result.error} type="error" />
              ) : (
                <Descriptions column={1} size="small">
                  {result.overall_score && (
                    <Descriptions.Item label="Quality Score">
                      <Progress 
                        percent={Math.round(result.overall_score)} 
                        status={result.overall_score >= 80 ? 'success' : result.overall_score >= 60 ? 'normal' : 'exception'}
                      />
                    </Descriptions.Item>
                  )}
                  {result.issues && (
                    <Descriptions.Item label="Issues Found">
                      {result.issues.map((issue: any, idx: number) => (
                        <Tag key={idx} color={issue.severity === 'high' ? 'red' : 'orange'}>
                          {issue.type}: {issue.count}
                        </Tag>
                      ))}
                    </Descriptions.Item>
                  )}
                  {result.pii_found !== undefined && (
                    <Descriptions.Item label="PII Detection">
                      <Tag color={result.pii_found ? 'red' : 'green'}>
                        {result.pii_found ? 'PII Detected' : 'No PII Found'}
                      </Tag>
                    </Descriptions.Item>
                  )}
                  {result.rows_processed && (
                    <Descriptions.Item label="Rows Processed">
                      {result.rows_processed}
                    </Descriptions.Item>
                  )}
                </Descriptions>
              )}
            </Card>
          ))}
        </Space>
      </Card>
    );
  };

  return (
    <Modal
      title="Pipeline Execution Status"
      open={visible}
      onCancel={onClose}
      width={800}
      footer={[
        <Button key="close" onClick={onClose}>
          Close
        </Button>,
        execution?.status === 'completed' && (
          <Button 
            key="download" 
            type="primary" 
            icon={<DownloadOutlined />}
            onClick={() => {
              // TODO: Implement download functionality
              window.open(`/api/v1/pipelines/${pipelineId}/executions/${executionId}/download`);
            }}
          >
            Download Results
          </Button>
        )
      ]}
    >
      {loading ? (
        <div style={{ textAlign: 'center', padding: 50 }}>
          <Spin size="large" />
          <Text style={{ display: 'block', marginTop: 16 }}>Loading execution status...</Text>
        </div>
      ) : execution ? (
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <Card>
            <Descriptions column={2}>
              <Descriptions.Item label="Status">
                <Space>
                  {getStatusIcon(execution.status)}
                  <Tag color={
                    execution.status === 'completed' ? 'success' :
                    execution.status === 'running' ? 'processing' :
                    execution.status === 'failed' ? 'error' :
                    'default'
                  }>
                    {execution.status.toUpperCase()}
                  </Tag>
                </Space>
              </Descriptions.Item>
              <Descriptions.Item label="Started">
                {new Date(execution.started_at).toLocaleString()}
              </Descriptions.Item>
              {execution.completed_at && (
                <Descriptions.Item label="Completed">
                  {new Date(execution.completed_at).toLocaleString()}
                </Descriptions.Item>
              )}
              {execution.error && (
                <Descriptions.Item label="Error" span={2}>
                  <Alert message={execution.error} type="error" />
                </Descriptions.Item>
              )}
            </Descriptions>
          </Card>

          {execution.status === 'running' && (
            <Progress 
              percent={execution.progress || 0} 
              status="active"
              strokeColor={{
                '0%': '#108ee9',
                '100%': '#87d068',
              }}
            />
          )}

          {renderResults()}
        </Space>
      ) : (
        <Alert
          message="Execution Not Found"
          description="The execution status could not be loaded."
          type="warning"
        />
      )}
    </Modal>
  );
}