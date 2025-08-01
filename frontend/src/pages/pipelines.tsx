import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { Layout, Card, Table, Button, Space, Tag, Modal, message, Tooltip, Empty, Spin, Tabs, Descriptions, Badge, Drawer } from 'antd';
import {
  PlusOutlined,
  PlayCircleOutlined,
  EditOutlined,
  DeleteOutlined,
  DatabaseOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ReloadOutlined,
  EyeOutlined,
  FileTextOutlined,
} from '@ant-design/icons';
import PageHeader from '@/components/PageHeader';
import PipelineBuilder from '@/components/PipelineBuilder';
import ExecutionMonitor from '@/components/ExecutionMonitor';
import ResultsViewer from '@/components/ResultsViewer';
import api from '@/services/api';
// import { formatDistanceToNow } from 'date-fns';

// const { TabPane } = Tabs; // Deprecated

// Simple date formatting helper
function formatDistanceToNow(date: Date): string {
  const now = new Date();
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);
  
  if (diffInSeconds < 60) return `${diffInSeconds} seconds ago`;
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minutes ago`;
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`;
  if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)} days ago`;
  return date.toLocaleDateString();
}

export default function PipelinesPage() {
  const router = useRouter();
  const [pipelines, setPipelines] = useState<any[]>([]);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedDataset, setSelectedDataset] = useState<any>(null);
  const [builderVisible, setBuilderVisible] = useState(false);
  const [selectedPipeline, setSelectedPipeline] = useState<any>(null);
  const [executionDetails, setExecutionDetails] = useState<any>(null);
  const [executionModal, setExecutionModal] = useState(false);
  const [activeTab, setActiveTab] = useState('pipelines');
  const [monitoringExecution, setMonitoringExecution] = useState<{pipelineId: string, executionId: string} | null>(null);
  const [viewingResults, setViewingResults] = useState<{pipelineId: string, executionId: string} | null>(null);
  const [resultsDrawerVisible, setResultsDrawerVisible] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [pipelinesRes, datasetsRes] = await Promise.all([
        api.get('/api/v1/pipelines'),
        api.get('/api/v1/datasets'),
      ]);
      setPipelines(pipelinesRes.data);
      setDatasets(datasetsRes.data);
    } catch (error) {
      message.error('Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const handleCreatePipeline = () => {
    setActiveTab('builder');
    setSelectedPipeline(null);
  };

  const handleEditPipeline = (pipeline: any) => {
    const dataset = datasets.find(d => d.id === pipeline.dataset_id);
    setSelectedDataset(dataset);
    setSelectedPipeline(pipeline);
    setActiveTab('builder');
  };

  const handleExecutePipeline = async (pipelineId: string) => {
    try {
      const response = await api.post(`/api/v1/pipelines/${pipelineId}/execute`);
      message.success('Pipeline execution started');
      
      // Show execution monitor
      setMonitoringExecution({
        pipelineId: pipelineId,
        executionId: response.data.id
      });
      setExecutionModal(true);
      
      loadData();
    } catch (error) {
      message.error('Failed to execute pipeline');
    }
  };

  const handleDeletePipeline = async (pipelineId: string) => {
    Modal.confirm({
      title: 'Delete Pipeline',
      content: 'Are you sure you want to delete this pipeline? This action cannot be undone.',
      okText: 'Delete',
      okType: 'danger',
      onOk: async () => {
        try {
          await api.delete(`/api/v1/pipelines/${pipelineId}`);
          message.success('Pipeline deleted');
          loadData();
        } catch (error) {
          message.error('Failed to delete pipeline');
        }
      },
    });
  };

  const handleViewExecutions = async (pipeline: any) => {
    try {
      const response = await api.get(`/api/v1/pipelines/${pipeline.id}/executions`);
      setExecutionDetails({
        pipeline,
        executions: response.data,
      });
      setExecutionModal(true);
    } catch (error) {
      message.error('Failed to load execution history');
    }
  };

  const handlePipelineSave = () => {
    setActiveTab('pipelines');
    loadData();
  };

  const getStatusBadge = (status: string) => {
    const statusConfig = {
      active: { color: 'success', text: 'Active' },
      draft: { color: 'default', text: 'Draft' },
      failed: { color: 'error', text: 'Failed' },
      running: { color: 'processing', text: 'Running' },
    };
    const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.draft;
    return <Badge status={config.color as any} text={config.text} />;
  };

  const pipelineColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: any) => (
        <Space>
          <span style={{ fontWeight: 500 }}>{text}</span>
          {record.description && (
            <Tooltip title={record.description}>
              <EyeOutlined style={{ color: '#999' }} />
            </Tooltip>
          )}
        </Space>
      ),
    },
    {
      title: 'Dataset',
      dataIndex: 'dataset_id',
      key: 'dataset_id',
      render: (datasetId: string) => {
        const dataset = datasets.find(d => d.id === datasetId);
        return (
          <Space>
            <DatabaseOutlined />
            <span>{dataset?.name || 'Unknown'}</span>
          </Space>
        );
      },
    },
    {
      title: 'Steps',
      dataIndex: 'steps',
      key: 'steps',
      render: (steps: any[]) => (
        <Tag>{steps.length} steps</Tag>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => getStatusBadge(status),
    },
    {
      title: 'Last Run',
      dataIndex: 'last_execution_at',
      key: 'last_execution_at',
      render: (date: string) => (
        date ? (
          <Tooltip title={new Date(date).toLocaleString()}>
            <Space>
              <ClockCircleOutlined />
              {formatDistanceToNow(new Date(date))}
            </Space>
          </Tooltip>
        ) : (
          <span style={{ color: '#999' }}>Never run</span>
        )
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: any) => (
        <Space>
          <Tooltip title="Execute Pipeline">
            <Button
              type="primary"
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => handleExecutePipeline(record.id)}
            />
          </Tooltip>
          <Tooltip title="View Executions">
            <Button
              size="small"
              icon={<ClockCircleOutlined />}
              onClick={() => handleViewExecutions(record)}
            />
          </Tooltip>
          <Tooltip title="Edit Pipeline">
            <Button
              size="small"
              icon={<EditOutlined />}
              onClick={() => handleEditPipeline(record)}
            />
          </Tooltip>
          <Tooltip title="Delete Pipeline">
            <Button
              danger
              size="small"
              icon={<DeleteOutlined />}
              onClick={() => handleDeletePipeline(record.id)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const datasetColumns = [
    {
      title: 'Dataset',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: any) => (
        <Space>
          <DatabaseOutlined />
          <span style={{ fontWeight: 500 }}>{text}</span>
          <Tag>{record.row_count} rows</Tag>
          <Tag>{record.column_count} columns</Tag>
        </Space>
      ),
    },
    {
      title: 'Quality Score',
      dataIndex: 'quality_score',
      key: 'quality_score',
      render: (score: number) => (
        score ? (
          <Tag color={score >= 80 ? 'success' : score >= 60 ? 'warning' : 'error'}>
            {score.toFixed(0)}%
          </Tag>
        ) : (
          <Tag>Not analyzed</Tag>
        )
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: any) => (
        <Button
          type="primary"
          size="small"
          onClick={() => {
            setSelectedDataset(record);
            setSelectedPipeline(null);
            setActiveTab('builder');
          }}
        >
          Create Pipeline
        </Button>
      ),
    },
  ];

  return (
    <Layout>
      <PageHeader 
        title="Pipeline Management"
        subtitle="Create and manage data processing pipelines"
      />
      
      <Layout.Content style={{ padding: '24px' }}>
        <Tabs 
          activeKey={activeTab} 
          onChange={setActiveTab}
          tabBarExtraContent={
            activeTab === 'pipelines' && (
              <Button 
                type="primary" 
                icon={<PlusOutlined />}
                onClick={handleCreatePipeline}
              >
                Create Pipeline
              </Button>
            )
          }
          items={[
            {
              key: 'pipelines',
              label: 'Pipelines',
              children: loading ? (
                <div style={{ textAlign: 'center', padding: '50px' }}>
                  <Spin size="large" />
                </div>
              ) : pipelines.length === 0 ? (
                <Empty
                  description="No pipelines created yet"
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                >
                  <Button type="primary" onClick={handleCreatePipeline}>
                    Create Your First Pipeline
                  </Button>
                </Empty>
              ) : (
                <Table
                  dataSource={pipelines}
                  columns={pipelineColumns}
                  rowKey="id"
                  pagination={false}
                />
              )
            },
            {
              key: 'builder',
              label: 'Create Pipeline',
              children: !selectedDataset ? (
                <Card>
                  <h3>Select a Dataset</h3>
                  <p>Choose a dataset to create a pipeline for:</p>
                  <Table
                    dataSource={datasets}
                    columns={datasetColumns}
                    rowKey="id"
                    pagination={false}
                  />
                </Card>
              ) : (
                <Card style={{ height: 'calc(100vh - 200px)' }}>
                  <PipelineBuilder
                    datasetId={selectedDataset.id}
                    datasetName={selectedDataset.name}
                    datasetInfo={{
                      rows: selectedDataset.row_count || 0,
                      columns: selectedDataset.column_count || 0,
                    }}
                    existingPipeline={selectedPipeline}
                    onSave={handlePipelineSave}
                  />
                </Card>
              )
            }
          ]}
        />
        
        <Modal
          title={`Execution History: ${executionDetails?.pipeline.name}`}
          open={executionModal}
          onCancel={() => setExecutionModal(false)}
          width={800}
          footer={[
            <Button key="refresh" icon={<ReloadOutlined />} onClick={() => handleViewExecutions(executionDetails?.pipeline)}>
              Refresh
            </Button>,
            <Button key="close" onClick={() => setExecutionModal(false)}>
              Close
            </Button>,
          ]}
        >
          {executionDetails?.executions.length === 0 ? (
            <Empty description="No executions yet" />
          ) : (
            <Table
              dataSource={executionDetails?.executions}
              rowKey="id"
              pagination={false}
              columns={[
                {
                  title: 'Execution ID',
                  dataIndex: 'id',
                  key: 'id',
                  render: (id: string) => <code>{id.slice(0, 8)}</code>,
                },
                {
                  title: 'Status',
                  dataIndex: 'status',
                  key: 'status',
                  render: (status: string) => {
                    const icons = {
                      completed: <CheckCircleOutlined style={{ color: '#52c41a' }} />,
                      failed: <CloseCircleOutlined style={{ color: '#f5222d' }} />,
                      running: <ClockCircleOutlined style={{ color: '#1890ff' }} />,
                      pending: <ClockCircleOutlined style={{ color: '#d9d9d9' }} />,
                    };
                    return (
                      <Space>
                        {icons[status as keyof typeof icons]}
                        <span>{status.charAt(0).toUpperCase() + status.slice(1)}</span>
                      </Space>
                    );
                  },
                },
                {
                  title: 'Started',
                  dataIndex: 'started_at',
                  key: 'started_at',
                  render: (date: string) => new Date(date).toLocaleString(),
                },
                {
                  title: 'Duration',
                  key: 'duration',
                  render: (_: any, record: any) => {
                    if (!record.completed_at) return '-';
                    const duration = new Date(record.completed_at).getTime() - new Date(record.started_at).getTime();
                    return `${Math.round(duration / 1000)}s`;
                  },
                },
                {
                  title: 'Records',
                  key: 'records',
                  render: (_: any, record: any) => {
                    if (!record.metrics) return '-';
                    return `${record.metrics.input_records} → ${record.metrics.output_records}`;
                  },
                },
                {
                  title: 'Error',
                  dataIndex: 'error',
                  key: 'error',
                  render: (error: string) => error ? (
                    <Tooltip title={error}>
                      <span style={{ color: '#f5222d', cursor: 'pointer' }}>View error</span>
                    </Tooltip>
                  ) : '-',
                },
                {
                  title: 'Actions',
                  key: 'actions',
                  render: (_: any, record: any) => (
                    <Space>
                      {record.status === 'running' && (
                        <Button
                          size="small"
                          icon={<EyeOutlined />}
                          onClick={() => {
                            setMonitoringExecution({
                              pipelineId: record.pipeline_id,
                              executionId: record.id
                            });
                            setExecutionModal(false);
                          }}
                        >
                          Monitor
                        </Button>
                      )}
                      {record.status === 'completed' && (
                        <Button
                          size="small"
                          icon={<FileTextOutlined />}
                          onClick={() => {
                            setViewingResults({
                              pipelineId: record.pipeline_id,
                              executionId: record.id
                            });
                            setResultsDrawerVisible(true);
                            setExecutionModal(false);
                          }}
                        >
                          View Results
                        </Button>
                      )}
                    </Space>
                  ),
                },
              ]}
              expandable={{
                expandedRowRender: (record: any) => (
                  <Descriptions size="small" column={2}>
                    <Descriptions.Item label="Pipeline ID">{record.pipeline_id}</Descriptions.Item>
                    <Descriptions.Item label="Execution ID">{record.id}</Descriptions.Item>
                    {record.metrics && (
                      <>
                        <Descriptions.Item label="Steps Completed">
                          {record.metrics.steps_completed}
                        </Descriptions.Item>
                        <Descriptions.Item label="Duration">
                          {record.metrics.duration_seconds}s
                        </Descriptions.Item>
                      </>
                    )}
                    {record.error && (
                      <Descriptions.Item label="Error" span={2}>
                        <pre style={{ color: '#f5222d', margin: 0 }}>{record.error}</pre>
                      </Descriptions.Item>
                    )}
                  </Descriptions>
                ),
              }}
            />
          )}
        </Modal>

        {/* Execution Monitor Modal */}
        {monitoringExecution && (
          <Modal
            title="Pipeline Execution Progress"
            open={!!monitoringExecution}
            width={800}
            footer={null}
            onCancel={() => setMonitoringExecution(null)}
          >
            <ExecutionMonitor
              pipelineId={monitoringExecution.pipelineId}
              executionId={monitoringExecution.executionId}
              onComplete={(execution) => {
                if (execution.status === 'completed') {
                  message.success('Pipeline execution completed successfully!');
                  loadData();
                  
                  // Optionally show results
                  setViewingResults({
                    pipelineId: monitoringExecution.pipelineId,
                    executionId: monitoringExecution.executionId
                  });
                  setResultsDrawerVisible(true);
                  setMonitoringExecution(null);
                } else if (execution.status === 'failed') {
                  message.error('Pipeline execution failed');
                }
              }}
            />
          </Modal>
        )}

        {/* Results Viewer Drawer */}
        <Drawer
          title="Pipeline Execution Results"
          placement="right"
          width="80%"
          onClose={() => {
            setResultsDrawerVisible(false);
            setViewingResults(null);
          }}
          open={resultsDrawerVisible}
        >
          {viewingResults && (
            <ResultsViewer
              pipelineId={viewingResults.pipelineId}
              executionId={viewingResults.executionId}
              onClose={() => {
                setResultsDrawerVisible(false);
                setViewingResults(null);
              }}
            />
          )}
        </Drawer>
      </Layout.Content>
    </Layout>
  );
}