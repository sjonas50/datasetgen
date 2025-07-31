import { useState, useEffect } from 'react';
import { Layout, Card, Button, Table, Tag, Space, Descriptions, Spin, message, Row, Col, Statistic } from 'antd';
import { ArrowLeftOutlined, PlayCircleOutlined, EditOutlined, ReloadOutlined } from '@ant-design/icons';
import { useRouter } from 'next/router';
import Link from 'next/link';
import { pipelineService } from '@/services/api';
import { PipelineConfig, PipelineExecution } from '@/types/pipeline';

const { Header, Content } = Layout;

export default function PipelineDetailsPage() {
  const router = useRouter();
  const { id } = router.query;
  const [pipeline, setPipeline] = useState<PipelineConfig | null>(null);
  const [executions, setExecutions] = useState<PipelineExecution[]>([]);
  const [loading, setLoading] = useState(true);
  const [executionsLoading, setExecutionsLoading] = useState(false);

  useEffect(() => {
    if (id && typeof id === 'string') {
      fetchPipeline(id);
      fetchExecutions(id);
    }
  }, [id]);

  const fetchPipeline = async (pipelineId: string) => {
    try {
      const response = await pipelineService.get(pipelineId);
      setPipeline(response.data);
    } catch (error) {
      message.error('Failed to load pipeline');
      router.push('/pipelines');
    } finally {
      setLoading(false);
    }
  };

  const fetchExecutions = async (pipelineId: string) => {
    setExecutionsLoading(true);
    try {
      const response = await pipelineService.getExecutions(pipelineId);
      setExecutions(response.data);
    } catch (error) {
      message.error('Failed to load execution history');
    } finally {
      setExecutionsLoading(false);
    }
  };

  const handleExecute = async () => {
    if (!id || typeof id !== 'string') return;

    try {
      await pipelineService.execute(id, {});
      message.success('Pipeline execution started');
      // Refresh executions after a delay
      setTimeout(() => fetchExecutions(id), 2000);
    } catch (error) {
      message.error('Failed to execute pipeline');
    }
  };

  const executionColumns = [
    {
      title: 'Execution ID',
      dataIndex: 'id',
      key: 'id',
      render: (text: string) => text.slice(0, 8) + '...',
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colorMap: { [key: string]: string } = {
          pending: 'default',
          running: 'processing',
          completed: 'success',
          failed: 'error',
        };
        return <Tag color={colorMap[status] || 'default'}>{status.toUpperCase()}</Tag>;
      },
    },
    {
      title: 'Started At',
      dataIndex: 'started_at',
      key: 'started_at',
      render: (date: string) => date ? new Date(date).toLocaleString() : '-',
    },
    {
      title: 'Duration',
      key: 'duration',
      render: (_: any, record: PipelineExecution) => {
        if (!record.started_at || !record.completed_at) return '-';
        const duration = new Date(record.completed_at).getTime() - new Date(record.started_at).getTime();
        const seconds = Math.floor(duration / 1000);
        const minutes = Math.floor(seconds / 60);
        if (minutes > 0) {
          return `${minutes}m ${seconds % 60}s`;
        }
        return `${seconds}s`;
      },
    },
    {
      title: 'Rows Processed',
      dataIndex: 'rows_processed',
      key: 'rows_processed',
      render: (rows: number) => rows?.toLocaleString() || '-',
    },
    {
      title: 'Error',
      dataIndex: 'error_message',
      key: 'error_message',
      ellipsis: true,
      render: (error: string) => error ? <span style={{ color: '#ff4d4f' }}>{error}</span> : '-',
    },
  ];

  if (loading) {
    return (
      <Layout style={{ minHeight: '100vh' }}>
        <Content style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Spin size="large" />
        </Content>
      </Layout>
    );
  }

  if (!pipeline) {
    return null;
  }

  // Calculate statistics
  const totalExecutions = executions.length;
  const successfulExecutions = executions.filter(e => e.status === 'completed').length;
  const failedExecutions = executions.filter(e => e.status === 'failed').length;
  const avgRowsProcessed = executions
    .filter(e => e.rows_processed)
    .reduce((sum, e) => sum + (e.rows_processed || 0), 0) / (totalExecutions || 1);

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#fff', padding: '0 50px', borderBottom: '1px solid #f0f0f0' }}>
        <div style={{ display: 'flex', alignItems: 'center', height: '100%' }}>
          <Button
            icon={<ArrowLeftOutlined />}
            onClick={() => router.push('/pipelines')}
            style={{ marginRight: 16 }}
          >
            Back
          </Button>
          <h2 style={{ margin: 0, flex: 1 }}>{pipeline.name}</h2>
          <Space>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={handleExecute}
            >
              Execute
            </Button>
            <Link href={`/pipelines/${id}/edit`}>
              <Button icon={<EditOutlined />}>
                Edit
              </Button>
            </Link>
          </Space>
        </div>
      </Header>

      <Content style={{ padding: '24px 50px' }}>
        <Row gutter={[24, 24]}>
          <Col span={24}>
            <Card title="Pipeline Details">
              <Descriptions column={2}>
                <Descriptions.Item label="Name">{pipeline.name}</Descriptions.Item>
                <Descriptions.Item label="Created">
                  {pipeline.id ? new Date().toLocaleDateString() : '-'}
                </Descriptions.Item>
                <Descriptions.Item label="Description" span={2}>
                  {pipeline.description}
                </Descriptions.Item>
                <Descriptions.Item label="Steps">
                  {pipeline.steps.length} steps
                </Descriptions.Item>
                <Descriptions.Item label="Optimization">
                  {pipeline.optimization?.enable_parallelization ? 'Enabled' : 'Disabled'}
                </Descriptions.Item>
              </Descriptions>
            </Card>
          </Col>

          <Col span={6}>
            <Card>
              <Statistic
                title="Total Executions"
                value={totalExecutions}
                suffix="runs"
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Success Rate"
                value={totalExecutions ? (successfulExecutions / totalExecutions * 100).toFixed(1) : 0}
                suffix="%"
                valueStyle={{ color: '#3f8600' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Failed Runs"
                value={failedExecutions}
                valueStyle={{ color: failedExecutions > 0 ? '#cf1322' : undefined }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Avg Rows Processed"
                value={Math.round(avgRowsProcessed)}
                suffix="rows"
              />
            </Card>
          </Col>

          <Col span={24}>
            <Card 
              title="Execution History"
              extra={
                <Button
                  icon={<ReloadOutlined />}
                  onClick={() => fetchExecutions(id as string)}
                  loading={executionsLoading}
                >
                  Refresh
                </Button>
              }
            >
              <Table
                columns={executionColumns}
                dataSource={executions}
                rowKey="id"
                loading={executionsLoading}
                pagination={{
                  pageSize: 10,
                  showSizeChanger: true,
                }}
              />
            </Card>
          </Col>
        </Row>
      </Content>
    </Layout>
  );
}