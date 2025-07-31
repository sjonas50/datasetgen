import { useState, useEffect } from 'react';
import { Layout, Card, Row, Col, Typography, Tag, Button, Modal, Form, Input, message } from 'antd';
import { DatabaseOutlined, FileTextOutlined, ApiOutlined, CloudOutlined, CheckCircleOutlined } from '@ant-design/icons';
import { connectorService } from '@/services/api';

const { Header, Content } = Layout;
const { Title, Paragraph } = Typography;

interface Connector {
  type: string;
  name: string;
  description: string;
  supported_formats: string[];
  features: string[];
}

const iconMap: { [key: string]: any } = {
  csv: <FileTextOutlined style={{ fontSize: '48px', color: '#2563eb' }} />,
  json: <FileTextOutlined style={{ fontSize: '48px', color: '#2563eb' }} />,
  postgresql: <DatabaseOutlined style={{ fontSize: '48px', color: '#2563eb' }} />,
  api: <ApiOutlined style={{ fontSize: '48px', color: '#2563eb' }} />,
};

export default function Connectors() {
  const [connectors, setConnectors] = useState<Connector[]>([]);
  const [loading, setLoading] = useState(false);
  const [testModalVisible, setTestModalVisible] = useState(false);
  const [selectedConnector, setSelectedConnector] = useState<string>('');
  const [form] = Form.useForm();

  useEffect(() => {
    fetchConnectors();
  }, []);

  const fetchConnectors = async () => {
    setLoading(true);
    try {
      const response = await connectorService.list();
      setConnectors(response.data);
    } catch (error) {
      message.error('Failed to fetch connectors');
    } finally {
      setLoading(false);
    }
  };

  const handleTestConnection = async (values: any) => {
    try {
      const config = JSON.parse(values.config);
      const response = await connectorService.test(selectedConnector, { config });
      
      if (response.data.status === 'success') {
        message.success('Connection successful!');
      } else {
        message.error('Connection failed: ' + response.data.message);
      }
      
      setTestModalVisible(false);
      form.resetFields();
    } catch (error) {
      message.error('Invalid configuration format');
    }
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#fff', padding: '0 50px', borderBottom: '1px solid #f0f0f0' }}>
        <h1 style={{ margin: '16px 0' }}>Data Connectors</h1>
      </Header>

      <Content style={{ padding: '24px 50px' }}>
        <Title level={4} style={{ marginBottom: '24px' }}>
          Available Connectors
        </Title>

        <Row gutter={[24, 24]}>
          {connectors.map((connector) => (
            <Col xs={24} sm={12} md={8} key={connector.type}>
              <Card
                hoverable
                style={{ height: '100%' }}
                actions={[
                  <Button 
                    key="test"
                    onClick={() => {
                      setSelectedConnector(connector.type);
                      setTestModalVisible(true);
                    }}
                  >
                    Test Connection
                  </Button>
                ]}
              >
                <div style={{ textAlign: 'center', marginBottom: '20px' }}>
                  {iconMap[connector.type] || <CloudOutlined style={{ fontSize: '48px', color: '#2563eb' }} />}
                </div>
                
                <Title level={4}>{connector.name}</Title>
                <Paragraph style={{ color: '#64748b', marginBottom: '16px' }}>
                  {connector.description}
                </Paragraph>

                <div style={{ marginBottom: '12px' }}>
                  <strong>Supported Formats:</strong>
                  <div style={{ marginTop: '8px' }}>
                    {connector.supported_formats.map((format) => (
                      <Tag key={format} style={{ marginBottom: '4px' }}>
                        {format}
                      </Tag>
                    ))}
                  </div>
                </div>

                <div>
                  <strong>Features:</strong>
                  <div style={{ marginTop: '8px' }}>
                    {connector.features.map((feature) => (
                      <div key={feature} style={{ color: '#10b981', marginBottom: '4px' }}>
                        <CheckCircleOutlined /> {feature}
                      </div>
                    ))}
                  </div>
                </div>
              </Card>
            </Col>
          ))}
        </Row>

        <Modal
          title={`Test ${selectedConnector.toUpperCase()} Connection`}
          open={testModalVisible}
          onCancel={() => {
            setTestModalVisible(false);
            form.resetFields();
          }}
          footer={null}
        >
          <Form
            form={form}
            layout="vertical"
            onFinish={handleTestConnection}
          >
            <Form.Item
              name="config"
              label="Connection Configuration (JSON)"
              rules={[{ required: true, message: 'Please enter connection configuration' }]}
            >
              <Input.TextArea 
                rows={6} 
                placeholder={`{
  "host": "localhost",
  "port": 5432,
  "database": "mydb",
  "username": "user",
  "password": "pass"
}`}
              />
            </Form.Item>

            <Form.Item>
              <Button type="primary" htmlType="submit" block>
                Test Connection
              </Button>
            </Form.Item>
          </Form>
        </Modal>
      </Content>
    </Layout>
  );
}