import React, { useState } from 'react';
import {
  Card,
  Button,
  Space,
  Typography,
  Radio,
  Input,
  Progress,
  Result,
  Alert,
  Spin,
  message,
  Row,
  Col,
  Tag,
  Divider,
} from 'antd';
import {
  FileTextOutlined,
  QuestionCircleOutlined,
  TagsOutlined,
  FileSearchOutlined,
  EditOutlined,
  CheckCircleOutlined,
  DownloadOutlined,
  RocketOutlined,
} from '@ant-design/icons';
import api from '@/services/api';

const { Title, Paragraph, Text } = Typography;
const { TextArea } = Input;

interface DatasetWizardProps {
  dataset: any;
  onComplete?: () => void;
  onCancel?: () => void;
}

const datasetTypes = [
  {
    id: 'qa_pairs',
    name: 'Q&A Pairs',
    icon: <QuestionCircleOutlined style={{ fontSize: 24 }} />,
    description: 'Generate question-answer pairs for chatbots, FAQ systems, or Q&A models',
    example: 'Q: What is the capital of France?\nA: The capital of France is Paris.',
  },
  {
    id: 'classification',
    name: 'Text Classification',
    icon: <TagsOutlined style={{ fontSize: 24 }} />,
    description: 'Create labeled text for sentiment analysis, topic classification, or spam detection',
    example: 'Text: "This product is amazing!"\nLabel: Positive',
  },
  {
    id: 'ner',
    name: 'Named Entity Recognition',
    icon: <FileSearchOutlined style={{ fontSize: 24 }} />,
    description: 'Label people, places, organizations, and other entities in text',
    example: 'Text: "Apple Inc. was founded by Steve Jobs"\nEntities: Apple Inc. (ORG), Steve Jobs (PERSON)',
  },
  {
    id: 'summarization',
    name: 'Text Summarization',
    icon: <FileTextOutlined style={{ fontSize: 24 }} />,
    description: 'Create text-summary pairs for training summarization models',
    example: 'Text: [Long article...]\nSummary: [Concise summary...]',
  },
  {
    id: 'custom',
    name: 'Custom Format',
    icon: <EditOutlined style={{ fontSize: 24 }} />,
    description: 'Describe your own dataset format and let AI create it',
    example: 'You define the format!',
  },
];

export default function DatasetWizard({ dataset, onComplete, onCancel }: DatasetWizardProps) {
  const [selectedType, setSelectedType] = useState<string>('');
  const [customInstructions, setCustomInstructions] = useState('');
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [generatedDataset, setGeneratedDataset] = useState<any>(null);
  const [error, setError] = useState('');

  const generateDataset = async () => {
    if (!selectedType) {
      message.warning('Please select a dataset type');
      return;
    }

    setGenerating(true);
    setError('');
    setProgress(0);
    
    try {
      // Step 1: Create a simple one-step pipeline
      setStatusMessage('Preparing your dataset generation...');
      setProgress(20);

      const pipelineConfig = {
        name: `${selectedType} Dataset Generation`,
        description: `Generate ${selectedType} dataset from uploaded files`,
        dataset_id: dataset.id,
        steps: [
          {
            id: 'step_1',
            type: 'document_extraction',
            name: 'Extract content from all files',
            config: {
              extraction_method: 'ai_enhanced',
              use_ai_extraction: true,
              extract_images: true,
              extraction_prompt: 'Extract all text content, tables, and important information from this document. Include metadata and structure.'
            }
          },
          {
            id: 'step_2',
            type: 'dataset_generation',
            name: 'Generate Dataset with AI',
            config: {
              dataset_type: selectedType,
              custom_instructions: customInstructions,
              auto_extract: true,
              include_all_files: true,
            }
          }
        ]
      };

      // Create pipeline
      const pipelineRes = await api.post('/api/v1/pipelines', pipelineConfig);
      const pipeline = pipelineRes.data;

      setProgress(40);
      setStatusMessage('AI is reading and understanding your files...');

      // Execute pipeline
      const execRes = await api.post(`/api/v1/pipelines/${pipeline.id}/execute`);
      const executionId = execRes.data.id;

      // Poll for completion
      setProgress(60);
      setStatusMessage('AI is generating your dataset...');

      let completed = false;
      let attempts = 0;
      const maxAttempts = 60; // 2 minutes timeout

      while (!completed && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        const statusRes = await api.get(`/api/v1/pipelines/${pipeline.id}/executions`);
        const execution = statusRes.data.find((e: any) => e.id === executionId);
        
        if (execution) {
          if (execution.status === 'completed') {
            completed = true;
            setProgress(90);
            setStatusMessage('Finalizing your dataset...');
            
            // Get results
            const resultsRes = await api.get(
              `/api/v1/pipelines/${pipeline.id}/executions/${executionId}/results`
            );
            
            // Extract the actual data from results
            const results = resultsRes.data;
            const processedData = {
              pipeline_id: pipeline.id,
              execution_id: executionId,
              row_count: results.results?.rows || 0,
              columns: results.results?.columns || [],
              preview: results.results?.preview || [],
              has_document_extraction: !!results.results?.document_extraction,
              step_results: results.results?.step_results || {}
            };
            
            setGeneratedDataset(processedData);
            setProgress(100);
            setStatusMessage('Dataset generated successfully!');
          } else if (execution.status === 'failed') {
            throw new Error(execution.error || 'Dataset generation failed');
          }
        }
        
        attempts++;
        // Update progress gradually
        if (progress < 80) {
          setProgress(prev => Math.min(prev + 5, 80));
        }
      }

      if (!completed) {
        throw new Error('Dataset generation timed out. Please try again.');
      }

    } catch (err: any) {
      setError(err.message || 'Failed to generate dataset');
      message.error('Dataset generation failed');
    } finally {
      setGenerating(false);
    }
  };

  const downloadDataset = async (format: string) => {
    try {
      // Use the download endpoint
      const response = await api.get(
        `/api/v1/pipelines/${generatedDataset.pipeline_id}/executions/${generatedDataset.execution_id}/download`,
        {
          params: { format },
          responseType: 'blob'
        }
      );
      
      const blob = new Blob([response.data]);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${selectedType}_dataset.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      
      message.success('Dataset downloaded successfully!');
    } catch (err) {
      message.error('Failed to download dataset');
    }
  };

  if (generatedDataset) {
    return (
      <Card>
        <Result
          status="success"
          title="Dataset Generated Successfully!"
          subTitle={`Your ${selectedType.replace('_', ' ')} dataset is ready`}
          extra={[
            <Space key="actions" size="large">
              <Button 
                type="primary" 
                icon={<DownloadOutlined />}
                onClick={() => downloadDataset('csv')}
                size="large"
              >
                Download CSV
              </Button>
              <Button 
                icon={<DownloadOutlined />}
                onClick={() => downloadDataset('json')}
              >
                Download JSON
              </Button>
              <Button 
                onClick={onComplete}
              >
                Done
              </Button>
            </Space>
          ]}
        >
          <div style={{ marginTop: 24 }}>
            <Alert
              message="Dataset Statistics"
              description={
                <Space direction="vertical">
                  <Text>Rows generated: {generatedDataset.row_count || 0}</Text>
                  <Text>Columns: {generatedDataset.columns?.join(', ') || 'N/A'}</Text>
                  {generatedDataset.has_document_extraction && (
                    <Text type="secondary">Documents processed and transformed into {selectedType.replace('_', ' ')} format</Text>
                  )}
                </Space>
              }
              type="info"
            />
            
            {generatedDataset.preview && generatedDataset.preview.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <Title level={5}>Preview (first 5 rows):</Title>
                <div style={{ 
                  maxHeight: 300, 
                  overflow: 'auto', 
                  border: '1px solid #f0f0f0',
                  borderRadius: 4,
                  padding: 8,
                  backgroundColor: '#fafafa'
                }}>
                  <pre style={{ margin: 0, fontSize: 12 }}>
                    {JSON.stringify(generatedDataset.preview.slice(0, 5), null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </div>
        </Result>
      </Card>
    );
  }

  return (
    <div>
      <Card>
        <Title level={3}>
          <RocketOutlined /> Generate Training Dataset
        </Title>
        <Paragraph>
          Choose the type of dataset you want to create. Our AI will analyze your files and generate
          high-quality training data automatically.
        </Paragraph>

        {!generating ? (
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <div>
              <Title level={5}>Select Dataset Type:</Title>
              <Row gutter={[16, 16]}>
                {datasetTypes.map(type => (
                  <Col xs={24} sm={12} md={8} key={type.id}>
                    <Card
                      hoverable
                      bordered={selectedType === type.id}
                      style={{ 
                        height: '100%',
                        borderColor: selectedType === type.id ? '#1890ff' : undefined,
                        borderWidth: selectedType === type.id ? 2 : 1
                      }}
                      onClick={() => setSelectedType(type.id)}
                    >
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div style={{ textAlign: 'center' }}>
                          {type.icon}
                        </div>
                        <Title level={5} style={{ marginBottom: 8 }}>
                          {type.name}
                        </Title>
                        <Paragraph type="secondary" style={{ marginBottom: 12 }}>
                          {type.description}
                        </Paragraph>
                        <Alert
                          message="Example"
                          description={<pre style={{ fontSize: 12, margin: 0 }}>{type.example}</pre>}
                          type="info"
                        />
                      </Space>
                    </Card>
                  </Col>
                ))}
              </Row>
            </div>

            {selectedType === 'custom' && (
              <div>
                <Title level={5}>Describe Your Dataset Format:</Title>
                <TextArea
                  rows={4}
                  placeholder="E.g., I need a dataset with product reviews and ratings, where each row has: review_text, rating (1-5), product_category, and reviewer_sentiment"
                  value={customInstructions}
                  onChange={(e) => setCustomInstructions(e.target.value)}
                />
              </div>
            )}

            {selectedType && (
              <Alert
                message="What will happen:"
                description={
                  <ol style={{ marginBottom: 0, paddingLeft: 20 }}>
                    <li>AI will read and understand all your uploaded files</li>
                    <li>Extract relevant information based on your selected format</li>
                    <li>Generate high-quality training examples</li>
                    <li>Format everything properly for machine learning</li>
                  </ol>
                }
                type="info"
                showIcon
              />
            )}

            <div style={{ textAlign: 'center' }}>
              <Space>
                <Button onClick={onCancel}>
                  Cancel
                </Button>
                <Button
                  type="primary"
                  size="large"
                  icon={<RocketOutlined />}
                  onClick={generateDataset}
                  disabled={!selectedType || (selectedType === 'custom' && !customInstructions)}
                >
                  Generate Dataset
                </Button>
              </Space>
            </div>

            {error && (
              <Alert message={error} type="error" showIcon closable />
            )}
          </Space>
        ) : (
          <div style={{ textAlign: 'center', padding: '40px 0' }}>
            <Space direction="vertical" size="large">
              <Spin size="large" />
              <Progress percent={progress} status="active" />
              <Title level={4}>{statusMessage}</Title>
              <Paragraph type="secondary">
                This may take a minute or two depending on your file size...
              </Paragraph>
            </Space>
          </div>
        )}
      </Card>
    </div>
  );
}