import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Button, 
  Space, 
  Typography, 
  Table, 
  Tag, 
  Tabs, 
  Alert, 
  Spin, 
  Statistic, 
  Row, 
  Col,
  Collapse,
  Divider,
  message,
  Drawer,
  Badge
} from 'antd';
import { 
  DownloadOutlined, 
  FileTextOutlined, 
  BarChartOutlined,
  EyeOutlined,
  ExportOutlined,
  CopyOutlined,
  CheckCircleOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import api from '@/services/api';

const { Title, Text, Paragraph } = Typography;
const { Panel } = Collapse;

interface ResultsViewerProps {
  pipelineId: string;
  executionId: string;
  onClose?: () => void;
}

interface ExecutionResults {
  status: string;
  output_file?: string;
  file_size?: number;
  rows?: number;
  columns?: string[];
  execution_status?: string;
  step_results?: Record<string, any>;
  metrics?: Record<string, any>;
  statistics?: {
    total_rows: number;
    total_columns: number;
    file_size_kb: number;
  };
  document_extraction?: {
    total_documents: number;
    preview_count: number;
    documents: Array<{
      index: number;
      source_file: string;
      file_type: string;
      content_preview?: string;
      content_length?: number;
      word_count?: number;
      metadata?: Record<string, any>;
      ai_insights?: Record<string, any>;
    }>;
    has_enhanced_content: boolean;
    has_ai_insights: boolean;
  };
  preview?: Array<Record<string, any>>;
}

const ResultsViewer: React.FC<ResultsViewerProps> = ({ pipelineId, executionId, onClose }) => {
  const [results, setResults] = useState<ExecutionResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDocument, setSelectedDocument] = useState<any>(null);
  const [drawerVisible, setDrawerVisible] = useState(false);

  useEffect(() => {
    fetchResults();
  }, [pipelineId, executionId]);

  const fetchResults = async () => {
    try {
      setLoading(true);
      const response = await api.get(`/api/v1/pipelines/${pipelineId}/executions/${executionId}/results`);
      setResults(response.data.results);
      setError(null);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch results');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (format: string = 'csv') => {
    try {
      const response = await api.get(
        `/api/v1/pipelines/${pipelineId}/executions/${executionId}/download?format=${format}`,
        { responseType: 'blob' }
      );
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `results_${executionId}.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      
      message.success(`Results downloaded as ${format.toUpperCase()}`);
    } catch (err) {
      message.error('Failed to download results');
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      message.success('Copied to clipboard');
    });
  };

  const renderDocumentExtraction = () => {
    if (!results?.document_extraction) return null;

    const { documents, total_documents, has_enhanced_content, has_ai_insights } = results.document_extraction;

    const columns = [
      {
        title: 'Source File',
        dataIndex: 'source_file',
        key: 'source_file',
        ellipsis: true,
        render: (text: string) => (
          <Text strong>{text.split('/').pop()}</Text>
        )
      },
      {
        title: 'Type',
        dataIndex: 'file_type',
        key: 'file_type',
        width: 100,
        render: (type: string) => (
          <Tag color={type === 'pdf' ? 'blue' : type === 'docx' ? 'green' : 'orange'}>
            {type.toUpperCase()}
          </Tag>
        )
      },
      {
        title: 'Content Stats',
        key: 'stats',
        width: 200,
        render: (_: any, record: any) => (
          <Space direction="vertical" size="small">
            {record.word_count && <Text type="secondary">Words: {record.word_count}</Text>}
            {record.content_length && <Text type="secondary">Chars: {record.content_length}</Text>}
          </Space>
        )
      },
      {
        title: 'Actions',
        key: 'actions',
        width: 150,
        render: (_: any, record: any) => (
          <Space>
            <Button 
              size="small" 
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedDocument(record);
                setDrawerVisible(true);
              }}
            >
              View
            </Button>
            <Button 
              size="small" 
              icon={<CopyOutlined />}
              onClick={() => copyToClipboard(record.content_preview || '')}
            >
              Copy
            </Button>
          </Space>
        )
      }
    ];

    return (
      <>
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={6}>
            <Card>
              <Statistic 
                title="Total Documents" 
                value={total_documents} 
                prefix={<FileTextOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic 
                title="Enhanced Content" 
                value={has_enhanced_content ? 'Yes' : 'No'} 
                valueStyle={{ color: has_enhanced_content ? '#3f8600' : '#999' }}
                prefix={has_enhanced_content ? <CheckCircleOutlined /> : <InfoCircleOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic 
                title="AI Insights" 
                value={has_ai_insights ? 'Available' : 'Not Available'} 
                valueStyle={{ color: has_ai_insights ? '#3f8600' : '#999' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic 
                title="File Size" 
                value={results.statistics?.file_size_kb || 0} 
                suffix="KB"
              />
            </Card>
          </Col>
        </Row>

        <Table
          dataSource={documents}
          columns={columns}
          rowKey="index"
          pagination={{
            pageSize: 10,
            showTotal: (total) => `Total ${total} documents`
          }}
        />

        <Drawer
          title="Document Content"
          placement="right"
          width={800}
          onClose={() => setDrawerVisible(false)}
          open={drawerVisible}
        >
          {selectedDocument && (
            <Space direction="vertical" style={{ width: '100%' }} size="large">
              <Card title="Document Information" size="small">
                <Paragraph>
                  <Text strong>Source: </Text>{selectedDocument.source_file}
                </Paragraph>
                <Paragraph>
                  <Text strong>Type: </Text>
                  <Tag>{selectedDocument.file_type}</Tag>
                </Paragraph>
                {selectedDocument.word_count && (
                  <Paragraph>
                    <Text strong>Word Count: </Text>{selectedDocument.word_count}
                  </Paragraph>
                )}
              </Card>

              {selectedDocument.metadata && Object.keys(selectedDocument.metadata).length > 0 && (
                <Card title="Metadata" size="small">
                  {Object.entries(selectedDocument.metadata).map(([key, value]) => (
                    <Paragraph key={key}>
                      <Text strong>{key}: </Text>
                      <Text>{String(value)}</Text>
                    </Paragraph>
                  ))}
                </Card>
              )}

              <Card title="Content Preview" size="small">
                <Paragraph 
                  copyable
                  style={{ 
                    whiteSpace: 'pre-wrap', 
                    maxHeight: 400, 
                    overflow: 'auto',
                    backgroundColor: '#f5f5f5',
                    padding: 12,
                    borderRadius: 4
                  }}
                >
                  {selectedDocument.content_preview}
                </Paragraph>
              </Card>

              {selectedDocument.ai_insights && (
                <Card title="AI Insights" size="small">
                  <pre style={{ 
                    maxHeight: 300, 
                    overflow: 'auto',
                    backgroundColor: '#f5f5f5',
                    padding: 12,
                    borderRadius: 4
                  }}>
                    {JSON.stringify(selectedDocument.ai_insights, null, 2)}
                  </pre>
                </Card>
              )}
            </Space>
          )}
        </Drawer>
      </>
    );
  };

  const renderStandardData = () => {
    if (!results?.preview) return null;

    const columns = results.columns?.map(col => ({
      title: col,
      dataIndex: col,
      key: col,
      ellipsis: true,
      render: (text: any) => {
        if (typeof text === 'object') {
          return <pre>{JSON.stringify(text, null, 2)}</pre>;
        }
        return text;
      }
    })) || [];

    return (
      <Table
        dataSource={results.preview}
        columns={columns}
        rowKey={(record, index) => index?.toString() || '0'}
        scroll={{ x: true }}
        pagination={{
          pageSize: 10,
          showTotal: (total) => `Showing preview of ${total} rows (Total: ${results.rows})`
        }}
      />
    );
  };

  if (loading) {
    return (
      <Card>
        <Spin size="large" />
      </Card>
    );
  }

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

  if (!results) {
    return (
      <Alert
        message="No Results"
        description="No results found for this execution"
        type="info"
        showIcon
      />
    );
  }

  const items = [
    {
      key: 'data',
      label: (
        <span>
          <FileTextOutlined /> Extracted Data
        </span>
      ),
      children: results.document_extraction ? renderDocumentExtraction() : renderStandardData()
    },
    {
      key: 'metrics',
      label: (
        <span>
          <BarChartOutlined /> Execution Metrics
        </span>
      ),
      children: (
        <Space direction="vertical" style={{ width: '100%' }}>
          {results.metrics && (
            <Card title="Pipeline Metrics">
              <Row gutter={16}>
                <Col span={6}>
                  <Statistic 
                    title="Input Records" 
                    value={results.metrics.input_records} 
                  />
                </Col>
                <Col span={6}>
                  <Statistic 
                    title="Output Records" 
                    value={results.metrics.output_records} 
                  />
                </Col>
                <Col span={6}>
                  <Statistic 
                    title="Steps Completed" 
                    value={results.metrics.steps_completed} 
                  />
                </Col>
                <Col span={6}>
                  <Statistic 
                    title="Duration" 
                    value={results.metrics.duration_seconds || 0} 
                    suffix="s"
                  />
                </Col>
              </Row>
            </Card>
          )}

          {results.step_results && (
            <Card title="Step Results">
              <Collapse>
                {Object.entries(results.step_results).map(([stepName, stepResult]: [string, any]) => (
                  <Panel 
                    header={
                      <Space>
                        <Badge 
                          status={stepResult.status === 'success' ? 'success' : 'error'} 
                          text={stepName.replace(/_/g, ' ')}
                        />
                      </Space>
                    }
                    key={stepName}
                  >
                    <pre style={{ 
                      maxHeight: 300, 
                      overflow: 'auto',
                      backgroundColor: '#f5f5f5',
                      padding: 12,
                      borderRadius: 4
                    }}>
                      {JSON.stringify(stepResult, null, 2)}
                    </pre>
                  </Panel>
                ))}
              </Collapse>
            </Card>
          )}
        </Space>
      )
    }
  ];

  return (
    <Card
      title={
        <Space>
          <FileTextOutlined style={{ fontSize: 24 }} />
          <Title level={4} style={{ margin: 0 }}>Pipeline Execution Results</Title>
        </Space>
      }
      extra={
        <Space>
          <Button icon={<DownloadOutlined />} onClick={() => handleDownload('csv')}>
            Download CSV
          </Button>
          <Button icon={<ExportOutlined />} onClick={() => handleDownload('json')}>
            Export JSON
          </Button>
          {onClose && (
            <Button onClick={onClose}>Close</Button>
          )}
        </Space>
      }
    >
      <Tabs items={items} />
    </Card>
  );
};

export default ResultsViewer;