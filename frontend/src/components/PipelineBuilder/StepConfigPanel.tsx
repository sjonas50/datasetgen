import React, { useState, useEffect } from 'react';
import { Form, Input, Select, Switch, Button, Divider, Space, InputNumber, Tag } from 'antd';
import { Node } from 'reactflow';
import CodeEditor from '../CodeEditor';
import styles from './StepConfigPanel.module.css';

const { Option } = Select;
const { TextArea } = Input;

interface StepConfigPanelProps {
  node: Node;
  onUpdate: (config: any) => void;
  onClose: () => void;
}

const StepConfigPanel: React.FC<StepConfigPanelProps> = ({ node, onUpdate, onClose }) => {
  const [form] = Form.useForm();
  const [nodeType, setNodeType] = useState(node.type || 'default');

  useEffect(() => {
    // Initialize form with node config
    form.setFieldsValue({
      label: node.data.label,
      ...node.data.config,
    });
    setNodeType(node.type || 'default');
  }, [node, form]);

  const handleSubmit = (values: any) => {
    const { label, ...config } = values;
    onUpdate(config);
    
    // Update node label if changed
    if (label !== node.data.label) {
      node.data.label = label;
    }
  };

  const renderConnectorConfig = () => (
    <>
      <Form.Item
        name="connector_type"
        label="Connector Type"
        rules={[{ required: true }]}
      >
        <Select>
          <Option value="csv">CSV</Option>
          <Option value="json">JSON</Option>
          <Option value="excel">Excel</Option>
          <Option value="api">API</Option>
        </Select>
      </Form.Item>
      
      <Form.Item
        name="file_path"
        label="File Path"
        rules={[{ required: true }]}
      >
        <Input placeholder="/path/to/data.csv" />
      </Form.Item>
      
      <Form.Item
        name="llm_enhanced"
        label="LLM Enhanced"
        valuePropName="checked"
      >
        <Switch />
      </Form.Item>
    </>
  );

  const renderTransformConfig = () => {
    const transformType = Form.useWatch('type', form);
    
    return (
      <>
        <Form.Item
          name="type"
          label="Transform Type"
          rules={[{ required: true }]}
        >
          <Select>
            <Option value="filter">Filter</Option>
            <Option value="select">Select Columns</Option>
            <Option value="aggregate">Aggregate</Option>
            <Option value="rename">Rename</Option>
            <Option value="custom">Custom Code</Option>
          </Select>
        </Form.Item>
        
        {transformType === 'filter' && (
          <Form.Item
            name="condition"
            label="Filter Condition"
            rules={[{ required: true }]}
          >
            <Input placeholder="age > 18 and status == 'active'" />
          </Form.Item>
        )}
        
        {transformType === 'select' && (
          <Form.Item
            name="columns"
            label="Columns"
            rules={[{ required: true }]}
          >
            <Select mode="tags" placeholder="Enter column names">
              {/* Dynamic options would be loaded based on upstream data */}
            </Select>
          </Form.Item>
        )}
        
        {transformType === 'custom' && (
          <Form.Item
            name="code"
            label="Python Code"
            rules={[{ required: true }]}
          >
            <CodeEditor
              height="200px"
              defaultValue="# df is the input DataFrame\n# Return the transformed df\n\ndf = df"
            />
          </Form.Item>
        )}
      </>
    );
  };

  const renderLLMConfig = () => (
    <>
      <Form.Item
        name="type"
        label="Process Type"
        rules={[{ required: true }]}
      >
        <Select>
          <Option value="enrich">Enrich Data</Option>
          <Option value="generate_code">Generate Code</Option>
          <Option value="validate">Validate Quality</Option>
        </Select>
      </Form.Item>
      
      <Form.Item
        name="llm_provider"
        label="LLM Provider"
      >
        <Select>
          <Option value="claude">Claude Sonnet 4</Option>
          <Option value="openai">GPT-4</Option>
          <Option value="gemini">Gemini 2.5</Option>
        </Select>
      </Form.Item>
      
      <Form.Item
        name="use_extended_thinking"
        label="Extended Thinking"
        valuePropName="checked"
        extra="Enable for complex reasoning tasks"
      >
        <Switch />
      </Form.Item>
      
      <Form.Item
        name="temperature"
        label="Temperature"
      >
        <InputNumber min={0} max={1} step={0.1} />
      </Form.Item>
    </>
  );

  const renderMultiModalConfig = () => {
    const stepType = Form.useWatch('type', form) || 'document_parse';
    
    return (
      <>
        <Form.Item
          name="type"
          label="Step Type"
        >
          <Select>
            <Option value="document_parse">Document Parser</Option>
            <Option value="pdf_table_extract">PDF Table Extract</Option>
            <Option value="image_data_extract">Image Data Extract</Option>
            <Option value="multimodal_fusion">Multi-Modal Fusion</Option>
          </Select>
        </Form.Item>
        
        {stepType === 'multimodal_fusion' && (
          <Form.Item
            name="fusion_strategy"
            label="Fusion Strategy"
            extra="How to combine multiple data modalities"
          >
            <Select>
              <Option value="early">Early Fusion</Option>
              <Option value="late">Late Fusion</Option>
              <Option value="hybrid">Hybrid Fusion</Option>
              <Option value="hierarchical">Hierarchical</Option>
              <Option value="attention">Attention-Based</Option>
            </Select>
          </Form.Item>
        )}
        
        {stepType === 'pdf_table_extract' && (
          <Form.Item
            name="confidence_threshold"
            label="Confidence Threshold"
            extra="Minimum confidence for table extraction"
          >
            <InputNumber min={0} max={1} step={0.05} />
          </Form.Item>
        )}
        
        {stepType === 'image_data_extract' && (
          <Form.Item
            name="analysis_type"
            label="Analysis Type"
          >
            <Select>
              <Option value="auto">Auto-Detect</Option>
              <Option value="chart">Chart Data</Option>
              <Option value="table">Table</Option>
              <Option value="dashboard">Dashboard</Option>
            </Select>
          </Form.Item>
        )}
      </>
    );
  };

  const renderValidateConfig = () => (
    <>
      <Form.Item
        name="fail_on_error"
        label="Fail on Error"
        valuePropName="checked"
        extra="Stop pipeline if validation fails"
      >
        <Switch />
      </Form.Item>
      
      <Divider>Validation Rules</Divider>
      
      <Form.List name="rules">
        {(fields, { add, remove }) => (
          <>
            {fields.map(({ key, name, ...restField }) => (
              <Space key={key} style={{ display: 'flex', marginBottom: 8 }} align="baseline">
                <Form.Item
                  {...restField}
                  name={[name, 'type']}
                  rules={[{ required: true }]}
                >
                  <Select placeholder="Rule type" style={{ width: 120 }}>
                    <Option value="not_null">Not Null</Option>
                    <Option value="unique">Unique</Option>
                    <Option value="range">Range</Option>
                    <Option value="regex">Regex</Option>
                  </Select>
                </Form.Item>
                <Form.Item
                  {...restField}
                  name={[name, 'columns']}
                  rules={[{ required: true }]}
                >
                  <Select
                    mode="tags"
                    placeholder="Columns"
                    style={{ width: 200 }}
                  />
                </Form.Item>
                <Button onClick={() => remove(name)}>Remove</Button>
              </Space>
            ))}
            <Form.Item>
              <Button type="dashed" onClick={() => add()} block>
                Add Rule
              </Button>
            </Form.Item>
          </>
        )}
      </Form.List>
    </>
  );

  const renderConfigByType = () => {
    switch (nodeType) {
      case 'connector':
        return renderConnectorConfig();
      case 'transform':
        return renderTransformConfig();
      case 'llm_process':
        return renderLLMConfig();
      case 'multimodal':
        return renderMultiModalConfig();
      case 'validate':
        return renderValidateConfig();
      default:
        return (
          <Form.Item
            name="config"
            label="Configuration"
          >
            <TextArea rows={4} placeholder="JSON configuration" />
          </Form.Item>
        );
    }
  };

  return (
    <div className={styles.configPanel}>
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        autoComplete="off"
      >
        <Form.Item
          name="label"
          label="Step Name"
          rules={[{ required: true }]}
        >
          <Input />
        </Form.Item>
        
        <Divider>Step Configuration</Divider>
        
        {renderConfigByType()}
        
        <Form.Item style={{ marginTop: 24 }}>
          <Space>
            <Button type="primary" htmlType="submit">
              Update
            </Button>
            <Button onClick={onClose}>
              Cancel
            </Button>
          </Space>
        </Form.Item>
      </Form>
    </div>
  );
};

export default StepConfigPanel;