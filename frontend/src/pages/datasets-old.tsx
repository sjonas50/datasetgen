import { useState, useEffect } from 'react';
import { Layout, Table, Button, Space, Tag, Modal, Form, Input, Select, message } from 'antd';
import { PlusOutlined, UploadOutlined, DeleteOutlined, EyeOutlined } from '@ant-design/icons';
import Link from 'next/link';
import { datasetService } from '@/services/api';

const { Header, Content } = Layout;
const { Option } = Select;

interface Dataset {
  id: string;
  name: string;
  description: string;
  status: string;
  source_type: string;
  row_count: number;
  created_at: string;
}

export default function Datasets() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [form] = Form.useForm();

  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    setLoading(true);
    try {
      const response = await datasetService.list();
      setDatasets(response.data.datasets);
    } catch (error) {
      message.error('Failed to fetch datasets');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (values: any) => {
    try {
      await datasetService.create(values);
      message.success('Dataset created successfully');
      setCreateModalVisible(false);
      form.resetFields();
      fetchDatasets();
    } catch (error) {
      message.error('Failed to create dataset');
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await datasetService.delete(id);
      message.success('Dataset deleted successfully');
      fetchDatasets();
    } catch (error) {
      message.error('Failed to delete dataset');
    }
  };

  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: Dataset) => (
        <Link href={`/datasets/${record.id}`}>
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
          created: 'blue',
          processing: 'orange',
          ready: 'green',
          failed: 'red',
        };
        return <Tag color={colorMap[status] || 'default'}>{status.toUpperCase()}</Tag>;
      },
    },
    {
      title: 'Source',
      dataIndex: 'source_type',
      key: 'source_type',
    },
    {
      title: 'Rows',
      dataIndex: 'row_count',
      key: 'row_count',
      render: (count: number) => count.toLocaleString(),
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
      render: (_: any, record: Dataset) => (
        <Space>
          <Button icon={<EyeOutlined />} size="small">View</Button>
          <Button icon={<UploadOutlined />} size="small">Upload</Button>
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
        <h1 style={{ margin: '16px 0' }}>Datasets</h1>
      </Header>

      <Content style={{ padding: '24px 50px' }}>
        <div style={{ marginBottom: '24px' }}>
          <Button 
            type="primary" 
            icon={<PlusOutlined />}
            onClick={() => setCreateModalVisible(true)}
          >
            Create Dataset
          </Button>
        </div>

        <Table
          columns={columns}
          dataSource={datasets}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showTotal: (total) => `Total ${total} datasets`,
          }}
        />

        <Modal
          title="Create Dataset"
          open={createModalVisible}
          onCancel={() => setCreateModalVisible(false)}
          footer={null}
        >
          <Form
            form={form}
            layout="vertical"
            onFinish={handleCreate}
          >
            <Form.Item
              name="name"
              label="Dataset Name"
              rules={[{ required: true, message: 'Please enter a dataset name' }]}
            >
              <Input placeholder="My Dataset" />
            </Form.Item>

            <Form.Item
              name="description"
              label="Description"
            >
              <Input.TextArea rows={3} placeholder="Dataset description..." />
            </Form.Item>

            <Form.Item
              name="source_type"
              label="Source Type"
              rules={[{ required: true, message: 'Please select a source type' }]}
            >
              <Select placeholder="Select source type">
                <Option value="csv">CSV File</Option>
                <Option value="json">JSON File</Option>
                <Option value="database">Database</Option>
                <Option value="api">API</Option>
              </Select>
            </Form.Item>

            <Form.Item
              name="source_config"
              label="Source Configuration"
              initialValue={{}}
            >
              <Input.TextArea 
                rows={4} 
                placeholder='{"file_path": "/path/to/file.csv"}'
              />
            </Form.Item>

            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit">
                  Create
                </Button>
                <Button onClick={() => setCreateModalVisible(false)}>
                  Cancel
                </Button>
              </Space>
            </Form.Item>
          </Form>
        </Modal>
      </Content>
    </Layout>
  );
}