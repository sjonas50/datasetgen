import { useState } from 'react';
import { Layout, Button, Modal, Form, Input, message } from 'antd';
import { ArrowLeftOutlined, SaveOutlined } from '@ant-design/icons';
import { useRouter } from 'next/router';
import dynamic from 'next/dynamic';
import { pipelineService } from '@/services/api';
import { PipelineConfig } from '@/types/pipeline';

const { Header, Content } = Layout;
const { TextArea } = Input;

// Dynamically import PipelineBuilder to avoid SSR issues with React Flow
const PipelineBuilder = dynamic(
  () => import('@/components/PipelineBuilder/PipelineBuilder'),
  { ssr: false }
);

export default function PipelineBuilderPage() {
  const router = useRouter();
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [pipelineConfig, setPipelineConfig] = useState<PipelineConfig | null>(null);
  const [form] = Form.useForm();

  const handleSave = (config: PipelineConfig) => {
    setPipelineConfig(config);
    setShowSaveModal(true);
  };

  const handleSaveSubmit = async (values: { name: string; description: string }) => {
    if (!pipelineConfig) return;

    try {
      const pipeline = {
        ...pipelineConfig,
        name: values.name,
        description: values.description,
      };

      await pipelineService.create(pipeline);
      message.success('Pipeline saved successfully');
      router.push('/pipelines');
    } catch (error) {
      message.error('Failed to save pipeline');
    }
  };

  const handleExecute = async (config: PipelineConfig) => {
    Modal.confirm({
      title: 'Execute Pipeline',
      content: 'Do you want to execute this pipeline now?',
      onOk: async () => {
        try {
          // Save pipeline first if not saved
          if (!config.id) {
            const result = await pipelineService.create(config);
            config.id = result.data.id;
          }

          // Execute pipeline
          await pipelineService.execute(config.id!, {});
          message.success('Pipeline execution started');
          router.push('/pipelines');
        } catch (error) {
          message.error('Failed to execute pipeline');
        }
      },
    });
  };

  return (
    <Layout style={{ height: '100vh' }}>
      <Header style={{ background: '#fff', padding: '0 24px', borderBottom: '1px solid #f0f0f0' }}>
        <div style={{ display: 'flex', alignItems: 'center', height: '100%' }}>
          <Button
            icon={<ArrowLeftOutlined />}
            onClick={() => router.push('/pipelines')}
            style={{ marginRight: 16 }}
          >
            Back to Pipelines
          </Button>
          <h2 style={{ margin: 0, flex: 1 }}>Pipeline Builder</h2>
        </div>
      </Header>

      <Content style={{ height: 'calc(100vh - 64px)', overflow: 'hidden' }}>
        <PipelineBuilder
          onSave={handleSave}
          onExecute={handleExecute}
        />
      </Content>

      <Modal
        title="Save Pipeline"
        open={showSaveModal}
        onCancel={() => setShowSaveModal(false)}
        footer={null}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSaveSubmit}
        >
          <Form.Item
            name="name"
            label="Pipeline Name"
            rules={[{ required: true, message: 'Please enter a pipeline name' }]}
          >
            <Input placeholder="My Data Pipeline" />
          </Form.Item>

          <Form.Item
            name="description"
            label="Description"
            rules={[{ required: true, message: 'Please enter a description' }]}
          >
            <TextArea
              rows={3}
              placeholder="Describe what this pipeline does..."
            />
          </Form.Item>

          <Form.Item style={{ marginBottom: 0 }}>
            <Button type="primary" htmlType="submit" icon={<SaveOutlined />}>
              Save Pipeline
            </Button>
          </Form.Item>
        </Form>
      </Modal>
    </Layout>
  );
}