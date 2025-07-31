import { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Progress,
  Select,
  DatePicker,
  Space,
  Spin,
  Alert,
  Typography,
} from 'antd';
import {
  DollarOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  RiseOutlined,
  FallOutlined,
  SyncOutlined,
  WarningOutlined,
} from '@ant-design/icons';
import dynamic from 'next/dynamic';
import Layout from '@/components/Layout';
import { monitoringService } from '@/services/monitoring';
import styles from '@/styles/Monitoring.module.css';

const { RangePicker } = DatePicker;
const { Title, Text } = Typography;

// Dynamically import charts to avoid SSR issues
const CostTrendChart = dynamic(() => import('@/components/charts/CostTrendChart'), {
  ssr: false,
  loading: () => <Spin />,
});

const ExecutionTrendChart = dynamic(() => import('@/components/charts/ExecutionTrendChart'), {
  ssr: false,
  loading: () => <Spin />,
});

const CostBreakdownChart = dynamic(() => import('@/components/charts/CostBreakdownChart'), {
  ssr: false,
  loading: () => <Spin />,
});

interface DashboardData {
  summary: {
    active_pipelines: number;
    total_executions_period: number;
    success_rate: number;
    total_cost_period: number;
    avg_execution_time: number;
    total_rows_processed: number;
  };
  active_pipelines: any[];
  recent_failures: any[];
  cost_breakdown: {
    llm: number;
    compute: number;
    storage: number;
    by_pipeline: Record<string, number>;
    by_provider: Record<string, number>;
  };
  execution_trend: any[];
  cost_trend: any[];
  period_days: number;
}

export default function MonitoringDashboard() {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [timeRange, setTimeRange] = useState(7); // days
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDashboardData();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      fetchDashboardData(true);
    }, 30000);
    
    return () => clearInterval(interval);
  }, [timeRange]);

  const fetchDashboardData = async (isRefresh = false) => {
    if (isRefresh) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }
    
    try {
      const data = await monitoringService.getDashboard(timeRange);
      setDashboardData(data);
      setError(null);
    } catch (err) {
      setError('Failed to load monitoring data');
      console.error(err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const formatDuration = (seconds: number) => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
  };

  const formatCost = (cost: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 4,
    }).format(cost);
  };

  const activePipelineColumns = [
    {
      title: 'Pipeline',
      dataIndex: 'pipeline_name',
      key: 'pipeline_name',
    },
    {
      title: 'Execution ID',
      dataIndex: 'execution_id',
      key: 'execution_id',
      render: (id: string) => <Text copyable>{id.slice(0, 8)}...</Text>,
    },
    {
      title: 'Duration',
      dataIndex: 'current_duration_seconds',
      key: 'duration',
      render: (duration: number) => (
        <span>
          <ClockCircleOutlined /> {formatDuration(duration)}
        </span>
      ),
    },
    {
      title: 'Progress',
      key: 'progress',
      render: (_: any, record: any) => {
        const progress = (record.completed_steps / record.total_steps) * 100;
        return (
          <Progress
            percent={Math.round(progress)}
            size="small"
            status="active"
          />
        );
      },
    },
  ];

  const failureColumns = [
    {
      title: 'Pipeline',
      dataIndex: 'pipeline_name',
      key: 'pipeline_name',
    },
    {
      title: 'Failed At',
      dataIndex: 'end_time',
      key: 'end_time',
      render: (time: string) => new Date(time).toLocaleString(),
    },
    {
      title: 'Error',
      dataIndex: 'error_message',
      key: 'error_message',
      ellipsis: true,
    },
    {
      title: 'Cost',
      dataIndex: 'total_cost_usd',
      key: 'cost',
      render: (cost: number) => formatCost(cost),
    },
  ];

  if (loading && !dashboardData) {
    return (
      <Layout>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '80vh' }}>
          <Spin size="large" />
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout>
        <div style={{ padding: '50px' }}>
          <Alert
            message="Error"
            description={error}
            type="error"
            showIcon
          />
        </div>
      </Layout>
    );
  }

  const data = dashboardData!;

  return (
    <Layout>
      <div style={{ background: '#fff', padding: '24px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h1 style={{ margin: 0 }}>Monitoring Dashboard</h1>
          <Space>
            <Select
              value={timeRange}
              onChange={setTimeRange}
              style={{ width: 120 }}
            >
              <Select.Option value={1}>Last 24h</Select.Option>
              <Select.Option value={7}>Last 7 days</Select.Option>
              <Select.Option value={14}>Last 14 days</Select.Option>
              <Select.Option value={30}>Last 30 days</Select.Option>
            </Select>
            <SyncOutlined
              spin={refreshing}
              onClick={() => fetchDashboardData(true)}
              style={{ fontSize: 18, cursor: 'pointer' }}
            />
          </Space>
        </div>
      </div>

      <div style={{ padding: '0 24px 24px' }}>
        {/* Summary Cards */}
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} md={8} lg={4}>
            <Card>
              <Statistic
                title="Active Pipelines"
                value={data.summary.active_pipelines}
                prefix={<ThunderboltOutlined />}
                valueStyle={{ color: '#3f8600' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={8} lg={4}>
            <Card>
              <Statistic
                title="Total Executions"
                value={data.summary.total_executions_period}
                suffix={<span style={{ fontSize: 14 }}>/ {timeRange}d</span>}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={8} lg={4}>
            <Card>
              <Statistic
                title="Success Rate"
                value={data.summary.success_rate}
                precision={1}
                suffix="%"
                valueStyle={{
                  color: data.summary.success_rate >= 95 ? '#3f8600' : '#cf1322',
                }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={8} lg={4}>
            <Card>
              <Statistic
                title="Total Cost"
                value={data.summary.total_cost_period}
                precision={2}
                prefix={<DollarOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={8} lg={4}>
            <Card>
              <Statistic
                title="Avg Execution Time"
                value={formatDuration(data.summary.avg_execution_time)}
                prefix={<ClockCircleOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={8} lg={4}>
            <Card>
              <Statistic
                title="Rows Processed"
                value={data.summary.total_rows_processed}
                suffix="rows"
                prefix={<DatabaseOutlined />}
              />
            </Card>
          </Col>
        </Row>

        {/* Charts Row */}
        <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
          <Col xs={24} lg={12}>
            <Card title="Execution Trend">
              <ExecutionTrendChart data={data.execution_trend} />
            </Card>
          </Col>
          <Col xs={24} lg={12}>
            <Card title="Cost Trend">
              <CostTrendChart data={data.cost_trend} />
            </Card>
          </Col>
        </Row>

        {/* Cost Breakdown */}
        <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
          <Col xs={24} md={12} lg={8}>
            <Card title="Cost by Type">
              <CostBreakdownChart
                data={[
                  { name: 'LLM', value: data.cost_breakdown.llm },
                  { name: 'Compute', value: data.cost_breakdown.compute },
                  { name: 'Storage', value: data.cost_breakdown.storage },
                ]}
                type="pie"
              />
            </Card>
          </Col>
          <Col xs={24} md={12} lg={8}>
            <Card title="Cost by Pipeline">
              <CostBreakdownChart
                data={Object.entries(data.cost_breakdown.by_pipeline)
                  .map(([name, value]) => ({ name, value }))
                  .sort((a, b) => b.value - a.value)
                  .slice(0, 5)}
                type="bar"
              />
            </Card>
          </Col>
          <Col xs={24} md={12} lg={8}>
            <Card title="Cost by Provider">
              <CostBreakdownChart
                data={Object.entries(data.cost_breakdown.by_provider)
                  .map(([name, value]) => ({ name, value }))}
                type="pie"
              />
            </Card>
          </Col>
        </Row>

        {/* Active Pipelines */}
        {data.active_pipelines.length > 0 && (
          <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
            <Col span={24}>
              <Card
                title={`Active Pipelines (${data.active_pipelines.length})`}
                extra={<Tag color="processing">Running</Tag>}
              >
                <Table
                  dataSource={data.active_pipelines}
                  columns={activePipelineColumns}
                  rowKey="execution_id"
                  pagination={false}
                  size="small"
                />
              </Card>
            </Col>
          </Row>
        )}

        {/* Recent Failures */}
        {data.recent_failures.length > 0 && (
          <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
            <Col span={24}>
              <Card
                title="Recent Failures"
                extra={
                  <Tag icon={<WarningOutlined />} color="error">
                    {data.recent_failures.length} failures
                  </Tag>
                }
              >
                <Table
                  dataSource={data.recent_failures}
                  columns={failureColumns}
                  rowKey="execution_id"
                  pagination={{ pageSize: 5 }}
                  size="small"
                />
              </Card>
            </Col>
          </Row>
        )}
      </div>
    </Layout>
  );
}