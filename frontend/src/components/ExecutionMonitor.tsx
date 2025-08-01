import React, { useState, useEffect } from 'react';
import { Card, Progress, Timeline, Tag, Space, Typography, Button, List, Alert, Spin } from 'antd';
import { 
  CheckCircleOutlined, 
  ClockCircleOutlined, 
  ExclamationCircleOutlined,
  LoadingOutlined,
  RocketOutlined,
  FileTextOutlined,
  EyeOutlined
} from '@ant-design/icons';
import api from '@/services/api';

const { Title, Text, Paragraph } = Typography;

interface ExecutionStatus {
  id: string;
  pipeline_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  started_at: string;
  completed_at?: string;
  error?: string;
  progress: number;
  current_step?: string;
  results?: Record<string, any>;
  metrics?: {
    input_records: number;
    output_records: number;
    duration_seconds: number;
    steps_completed: number;
    output_path?: string;
  };
}

interface ExecutionMonitorProps {
  pipelineId: string;
  executionId: string;
  onComplete?: (execution: ExecutionStatus) => void;
}

const ExecutionMonitor: React.FC<ExecutionMonitorProps> = ({ pipelineId, executionId, onComplete }) => {
  const [execution, setExecution] = useState<ExecutionStatus | null>(null);
  const [polling, setPolling] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    const fetchExecutionStatus = async () => {
      try {
        const response = await api.get(`/api/v1/pipelines/${pipelineId}/executions`);
        const executions = response.data;
        const currentExecution = executions.find((e: ExecutionStatus) => e.id === executionId);
        
        if (currentExecution) {
          setExecution(currentExecution);
          
          // Stop polling if execution is complete
          if (currentExecution.status === 'completed' || currentExecution.status === 'failed') {
            setPolling(false);
            if (onComplete) {
              onComplete(currentExecution);
            }
          }
        }
      } catch (err: any) {
        console.error('Error fetching execution status:', err);
        setError(err.message || 'Failed to fetch execution status');
      }
    };

    // Initial fetch
    fetchExecutionStatus();

    // Set up polling
    if (polling) {
      intervalId = setInterval(fetchExecutionStatus, 2000); // Poll every 2 seconds
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [pipelineId, executionId, polling, onComplete]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <ClockCircleOutlined style={{ color: '#faad14' }} />;
      case 'running':
        return <LoadingOutlined style={{ color: '#1890ff' }} />;
      case 'completed':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'failed':
        return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return <ClockCircleOutlined />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'warning';
      case 'running': return 'processing';
      case 'completed': return 'success';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const formatDuration = (seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  const renderResults = () => {
    if (!execution?.results) return null;

    const resultEntries = Object.entries(execution.results);
    
    return (
      <Card title="Execution Results" style={{ marginTop: 16 }}>
        <Timeline>
          {resultEntries.map(([stepName, result]: [string, any]) => (
            <Timeline.Item
              key={stepName}
              color={result.status === 'success' ? 'green' : result.error ? 'red' : 'blue'}
              dot={result.status === 'success' ? <CheckCircleOutlined /> : result.error ? <ExclamationCircleOutlined /> : <ClockCircleOutlined />}
            >
              <Title level={5}>{stepName.replace(/_/g, ' ').replace(/step \d+ /, '')}</Title>
              {result.status && (
                <Tag color={result.status === 'success' ? 'success' : 'error'}>
                  {result.status}
                </Tag>
              )}
              {result.files_processed !== undefined && (
                <Text type="secondary"> • {result.files_processed} files processed</Text>
              )}
              {result.successful_extractions !== undefined && (
                <Text type="secondary"> • {result.successful_extractions} successful extractions</Text>
              )}
              {result.error && (
                <Alert
                  message="Error"
                  description={result.error}
                  type="error"
                  showIcon
                  style={{ marginTop: 8 }}
                />
              )}
              {result.rows_analyzed && (
                <Paragraph>
                  <Text type="secondary">Analyzed {result.rows_analyzed} rows</Text>
                  {result.average_word_count > 0 && (
                    <Text type="secondary"> • Average word count: {Math.round(result.average_word_count)}</Text>
                  )}
                </Paragraph>
              )}
            </Timeline.Item>
          ))}
        </Timeline>
      </Card>
    );
  };

  if (error) {
    return (
      <Alert
        message="Error"
        description={error}
        type="error"
        showIcon
      />
    );
  }

  if (!execution) {
    return (
      <Card>
        <Spin size="large" />
      </Card>
    );
  }

  return (
    <div>
      <Card>
        <Space direction="vertical" style={{ width: '100%' }}>
          <Space align="center">
            <RocketOutlined style={{ fontSize: 24, color: '#1890ff' }} />
            <Title level={4} style={{ margin: 0 }}>Pipeline Execution Monitor</Title>
          </Space>

          <Space>
            {getStatusIcon(execution.status)}
            <Tag color={getStatusColor(execution.status)}>
              {execution.status.toUpperCase()}
            </Tag>
            {execution.current_step && execution.status === 'running' && (
              <Text type="secondary">Current step: {execution.current_step}</Text>
            )}
          </Space>

          {execution.status === 'running' && (
            <Progress 
              percent={execution.progress} 
              status="active"
              strokeColor={{
                '0%': '#108ee9',
                '100%': '#87d068',
              }}
            />
          )}

          {execution.metrics && (
            <Card type="inner" title="Execution Metrics">
              <Space direction="vertical">
                <Text>
                  <FileTextOutlined /> Input Records: <strong>{execution.metrics.input_records}</strong>
                </Text>
                <Text>
                  <FileTextOutlined /> Output Records: <strong>{execution.metrics.output_records}</strong>
                </Text>
                <Text>
                  <ClockCircleOutlined /> Duration: <strong>{formatDuration(execution.metrics.duration_seconds)}</strong>
                </Text>
                <Text>
                  <CheckCircleOutlined /> Steps Completed: <strong>{execution.metrics.steps_completed}</strong>
                </Text>
                {execution.metrics.output_path && (
                  <Space>
                    <Text>Output saved to: {execution.metrics.output_path}</Text>
                    <Button 
                      size="small" 
                      icon={<EyeOutlined />}
                      onClick={() => window.open(`/api/v1/pipelines/${pipelineId}/executions/${executionId}/results`, '_blank')}
                    >
                      View Results
                    </Button>
                  </Space>
                )}
              </Space>
            </Card>
          )}

          {execution.error && (
            <Alert
              message="Execution Failed"
              description={execution.error}
              type="error"
              showIcon
            />
          )}
        </Space>
      </Card>

      {renderResults()}
    </div>
  );
};

export default ExecutionMonitor;