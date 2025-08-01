import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { Layout, Card, Table, Button, Space, Tag, Modal, Form, Input, message, Tabs, Empty, Spin, Typography, Divider, Tooltip } from 'antd';
import {
  PlusOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  UploadOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  FolderOpenOutlined,
  RocketOutlined,
} from '@ant-design/icons';
import PageHeader from '@/components/PageHeader';
import FileUpload from '@/components/FileUpload';
import SmartPipelineCreator from '@/components/SmartPipelineCreator';
import DatasetWizard from '@/components/DatasetWizard';
import api from '@/services/api';

const { TextArea } = Input;
const { Title, Text } = Typography;

export default function DatasetsPage() {
  const router = useRouter();
  const [datasets, setDatasets] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([]);
  const [creating, setCreating] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<any>(null);
  const [showSmartPipeline, setShowSmartPipeline] = useState(false);
  const [showDatasetWizard, setShowDatasetWizard] = useState(false);
  const [form] = Form.useForm();

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      setLoading(true);
      const response = await api.get('/api/v1/datasets');
      setDatasets(response.data);
    } catch (error) {
      message.error('Failed to load datasets');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateDataset = async (values: any) => {
    if (uploadedFiles.length === 0) {
      message.error('Please upload at least one file');
      return;
    }

    try {
      setCreating(true);
      const response = await api.post('/api/v1/datasets', {
        name: values.name,
        description: values.description,
        file_ids: uploadedFiles.map(f => f.file_id),
      });
      
      message.success('Dataset created successfully');
      setCreateModalVisible(false);
      form.resetFields();
      setUploadedFiles([]);
      loadDatasets();
      
      // Show smart pipeline creator for the new dataset
      setSelectedDataset(response.data);
      setShowSmartPipeline(true);
    } catch (error) {
      message.error('Failed to create dataset');
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteDataset = async (datasetId: string) => {
    Modal.confirm({
      title: 'Delete Dataset',
      content: 'Are you sure you want to delete this dataset? This action cannot be undone.',
      okText: 'Delete',
      okType: 'danger',
      onOk: async () => {
        try {
          await api.delete(`/api/v1/datasets/${datasetId}`);
          message.success('Dataset deleted');
          loadDatasets();
        } catch (error) {
          message.error('Failed to delete dataset');
        }
      },
    });
  };

  const getFileTypeIcon = (files: any[]) => {
    if (!files || files.length === 0) return <DatabaseOutlined />;
    
    const fileTypes = files.map(f => f.preview?.type).filter(Boolean);
    if (fileTypes.some(t => t === 'document')) return <FileTextOutlined />;
    if (fileTypes.some(t => t === 'image')) return <FileTextOutlined />;
    return <DatabaseOutlined />;
  };

  const getDatasetTypeTags = (files: any[]) => {
    const types = new Set<string>();
    files?.forEach(file => {
      if (file.preview?.type === 'document') {
        types.add(file.preview.format?.toUpperCase() || 'DOC');
      } else if (file.preview?.type === 'image') {
        types.add('IMAGE');
      } else if (file.preview?.rows) {
        types.add('DATA');
      }
    });
    return Array.from(types);
  };

  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: any) => (
        <Space>
          {getFileTypeIcon(record.files)}
          <span style={{ fontWeight: 500 }}>{text}</span>
        </Space>
      ),
    },
    {
      title: 'Type',
      key: 'type',
      render: (_: any, record: any) => (
        <Space>
          {getDatasetTypeTags(record.files).map(type => (
            <Tag key={type} color={
              type === 'PDF' ? 'red' :
              type === 'WORD' ? 'blue' :
              type === 'IMAGE' ? 'orange' :
              type === 'DATA' ? 'green' :
              'default'
            }>
              {type}
            </Tag>
          ))}
        </Space>
      ),
    },
    {
      title: 'Files',
      key: 'files',
      render: (_: any, record: any) => (
        <Tag icon={<FolderOpenOutlined />}>
          {record.files?.length || 0} files
        </Tag>
      ),
    },
    {
      title: 'Quality',
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
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => new Date(date).toLocaleDateString(),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: any) => (
        <Space>
          <Tooltip title="AI-Powered Dataset Generation">
            <Button
              type="primary"
              size="small"
              icon={<RocketOutlined />}
              onClick={() => {
                setSelectedDataset(record);
                setShowDatasetWizard(true);
              }}
            >
              Generate Dataset
            </Button>
          </Tooltip>
          <Tooltip title="Advanced Pipeline Builder">
            <Button
              size="small"
              icon={<EditOutlined />}
              onClick={() => router.push(`/pipelines?dataset=${record.id}`)}
            >
              Advanced
            </Button>
          </Tooltip>
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => router.push(`/datasets/${record.id}`)}
          >
            View
          </Button>
          <Button
            danger
            size="small"
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteDataset(record.id)}
          />
        </Space>
      ),
    },
  ];

  return (
    <Layout>
      <PageHeader 
        title="Dataset Management"
        subtitle="Upload and manage your data sources"
      />
      
      <Layout.Content style={{ padding: '24px' }}>
        <Card>
          <div style={{ marginBottom: 24 }}>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setCreateModalVisible(true)}
              size="large"
            >
              Create New Dataset
            </Button>
          </div>

          {loading ? (
            <div style={{ textAlign: 'center', padding: '50px' }}>
              <Spin size="large" />
            </div>
          ) : datasets.length === 0 ? (
            <Empty
              description="No datasets created yet"
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            >
              <Button 
                type="primary" 
                onClick={() => setCreateModalVisible(true)}
              >
                Upload Your First Dataset
              </Button>
            </Empty>
          ) : (
            <Table
              dataSource={datasets}
              columns={columns}
              rowKey="id"
              pagination={false}
            />
          )}
        </Card>

        <Modal
          title="Create New Dataset"
          open={createModalVisible}
          onCancel={() => {
            setCreateModalVisible(false);
            form.resetFields();
            setUploadedFiles([]);
          }}
          onOk={() => form.submit()}
          confirmLoading={creating}
          width={700}
        >
          <Form
            form={form}
            layout="vertical"
            onFinish={handleCreateDataset}
          >
            <Form.Item
              name="name"
              label="Dataset Name"
              rules={[{ required: true, message: 'Please enter a dataset name' }]}
            >
              <Input placeholder="e.g., Customer Support Documents" />
            </Form.Item>

            <Form.Item
              name="description"
              label="Description"
            >
              <TextArea 
                rows={3} 
                placeholder="Describe the dataset and its intended use..."
              />
            </Form.Item>

            <Form.Item label="Upload Files">
              <FileUpload 
                onFilesUploaded={setUploadedFiles}
                multiple={true}
              />
              <div style={{ marginTop: 8 }}>
                <Text type="secondary">
                  Upload any combination of files: spreadsheets, documents, PDFs, images, etc.
                </Text>
              </div>
            </Form.Item>
          </Form>
        </Modal>

        <Modal
          title="AI-Powered Pipeline Creation"
          open={showSmartPipeline}
          onCancel={() => {
            setShowSmartPipeline(false);
            setSelectedDataset(null);
          }}
          footer={null}
          width={900}
        >
          {selectedDataset && (
            <SmartPipelineCreator
              dataset={selectedDataset}
              onComplete={() => {
                setShowSmartPipeline(false);
                // Add small delay to allow modal to close properly before navigation
                setTimeout(() => {
                  router.push('/pipelines');
                }, 100);
              }}
              onCancel={() => {
                setShowSmartPipeline(false);
                // Add small delay to allow modal to close properly before navigation
                setTimeout(() => {
                  router.push(`/pipelines?dataset=${selectedDataset.id}`);
                }, 100);
              }}
            />
          )}
        </Modal>

        <Modal
          title="Generate Training Dataset"
          open={showDatasetWizard}
          onCancel={() => {
            setShowDatasetWizard(false);
            setSelectedDataset(null);
          }}
          footer={null}
          width={1000}
        >
          {selectedDataset && (
            <DatasetWizard
              dataset={selectedDataset}
              onComplete={() => {
                setShowDatasetWizard(false);
                loadDatasets();
              }}
              onCancel={() => {
                setShowDatasetWizard(false);
              }}
            />
          )}
        </Modal>
      </Layout.Content>
    </Layout>
  );
}