import { useState, useEffect } from 'react';
import { Layout, Button, Spin, message } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import { useRouter } from 'next/router';
import dynamic from 'next/dynamic';
import { pipelineService } from '@/services/api';
import { PipelineConfig } from '@/types/pipeline';

const { Header, Content } = Layout;

// Dynamically import PipelineBuilder to avoid SSR issues
const PipelineBuilder = dynamic(
  () => import('@/components/PipelineBuilder/PipelineBuilder'),
  { 
    ssr: false,
    loading: () => <Spin size="large" style={{ display: 'block', margin: '100px auto' }} />
  }
);

export default function EditPipelinePage() {
  const router = useRouter();
  const { id } = router.query;
  const [pipeline, setPipeline] = useState<PipelineConfig | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (id && typeof id === 'string') {
      fetchPipeline(id);
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

  const handleSave = async (config: PipelineConfig) => {
    if (!id || typeof id !== 'string') return;

    try {
      await pipelineService.update(id, config);
      message.success('Pipeline updated successfully');
      router.push('/pipelines');
    } catch (error) {
      message.error('Failed to update pipeline');
    }
  };

  const handleExecute = async (config: PipelineConfig) => {
    if (!id || typeof id !== 'string') return;

    try {
      await pipelineService.execute(id, {});
      message.success('Pipeline execution started');
      router.push(`/pipelines/${id}`);
    } catch (error) {
      message.error('Failed to execute pipeline');
    }
  };

  if (loading) {
    return (
      <Layout style={{ height: '100vh' }}>
        <Content style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Spin size="large" />
        </Content>
      </Layout>
    );
  }

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
          <h2 style={{ margin: 0, flex: 1 }}>Edit Pipeline: {pipeline?.name}</h2>
        </div>
      </Header>

      <Content style={{ height: 'calc(100vh - 64px)', overflow: 'hidden' }}>
        {pipeline && (
          <PipelineBuilder
            pipeline={pipeline}
            onSave={handleSave}
            onExecute={handleExecute}
          />
        )}
      </Content>
    </Layout>
  );
}