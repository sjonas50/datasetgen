import { useState, useEffect } from 'react';
import { Layout, Table, Button, Space, Tag, Modal, message } from 'antd';
import { PlusOutlined, PlayCircleOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons';
import Link from 'next/link';
import { pipelineService } from '@/services/api';

const { Header, Content } = Layout;

interface Pipeline {
  id: string;
  name: string;
  description: string;
  status: string;
  created_at: string;
}

export default function Pipelines() {
  const [pipelines, setPipelines] = useState<Pipeline[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchPipelines();
  }, []);

  const fetchPipelines = async () => {
    setLoading(true);
    try {
      const response = await pipelineService.list();
      setPipelines(response.data);
    } catch (error) {
      message.error('Failed to fetch pipelines');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await pipelineService.delete(id);
      message.success('Pipeline deleted successfully');
      fetchPipelines();
    } catch (error) {
      message.error('Failed to delete pipeline');
    }
  };

  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: Pipeline) => (
        <Link href={`/pipelines/${record.id}`}>
          <a className="font-medium text-blue-600 hover:text-blue-800">{text}</a>
        </Link>
      ),
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colorMap: { [key: string]: string } = {
          draft: 'default',
          active: 'green',
          paused: 'orange',
          archived: 'red',
        };
        return <Tag color={colorMap[status] || 'default'}>{status.toUpperCase()}</Tag>;
      },
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => new Date(date).toLocaleDateString(),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: Pipeline) => (
        <Space>
          <Button icon={<PlayCircleOutlined />} size="small">Run</Button>
          <Button icon={<EditOutlined />} size="small">Edit</Button>
          <Button 
            icon={<DeleteOutlined />} 
            size="small" 
            danger
            onClick={() => handleDelete(record.id)}
          >
            Delete
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#fff', padding: '0 50px', borderBottom: '1px solid #f0f0f0' }}>
        <h1 style={{ margin: '16px 0' }}>Pipelines</h1>
      </Header>

      <Content style={{ padding: '24px 50px' }}>
        <div style={{ marginBottom: '24px' }}>
          <Link href="/pipelines/builder">
            <Button type="primary" icon={<PlusOutlined />}>
              Create Pipeline
            </Button>
          </Link>
        </div>

        <Table
          columns={columns}
          dataSource={pipelines}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showTotal: (total) => `Total ${total} pipelines`,
          }}
        />
      </Content>
    </Layout>
  );
}