import React, { useState } from 'react';
import { Upload, Button, message, Card, Space, Tag, Typography, Progress, List } from 'antd';
import {
  InboxOutlined,
  FileTextOutlined,
  FilePdfOutlined,
  FileWordOutlined,
  FileImageOutlined,
  FileExcelOutlined,
  FileUnknownOutlined,
} from '@ant-design/icons';
import type { UploadProps } from 'antd';
import api from '@/services/api';

const { Dragger } = Upload;
const { Text, Title } = Typography;

interface FileUploadProps {
  onFilesUploaded?: (files: any[]) => void;
  multiple?: boolean;
}

export default function FileUpload({ onFilesUploaded, multiple = true }: FileUploadProps) {
  const [uploading, setUploading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([]);
  const [fileList, setFileList] = useState<any[]>([]);

  const getFileIcon = (fileType: string) => {
    if (fileType?.includes('pdf')) return <FilePdfOutlined style={{ fontSize: 48, color: '#ff4d4f' }} />;
    if (fileType?.includes('word') || fileType?.includes('document')) return <FileWordOutlined style={{ fontSize: 48, color: '#1890ff' }} />;
    if (fileType?.includes('sheet') || fileType?.includes('excel')) return <FileExcelOutlined style={{ fontSize: 48, color: '#52c41a' }} />;
    if (fileType?.includes('image')) return <FileImageOutlined style={{ fontSize: 48, color: '#faad14' }} />;
    if (fileType?.includes('text') || fileType?.includes('plain')) return <FileTextOutlined style={{ fontSize: 48, color: '#722ed1' }} />;
    return <FileUnknownOutlined style={{ fontSize: 48, color: '#8c8c8c' }} />;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const uploadProps: UploadProps = {
    name: 'file',
    multiple: multiple,
    showUploadList: false,
    customRequest: async ({ file, onSuccess, onError, onProgress }) => {
      const formData = new FormData();
      formData.append('file', file as any);

      try {
        setUploading(true);
        const response = await api.post('/api/v1/files/upload', formData, {
          // Let axios set the Content-Type header automatically for multipart/form-data
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total!);
            onProgress?.({ percent: percentCompleted });
          },
        });

        const uploadedFile = response.data;
        const newUploadedFiles = [...uploadedFiles, uploadedFile];
        setUploadedFiles(newUploadedFiles);
        onSuccess?.(uploadedFile);
        message.success(`${(file as any).name} uploaded successfully`);
        
        // Call the callback with the updated array
        if (onFilesUploaded) {
          onFilesUploaded(newUploadedFiles);
        }
      } catch (error: any) {
        console.error('File upload error:', error);
        console.error('Response:', error.response);
        console.error('Auth token:', localStorage.getItem('auth_token'));
        onError?.(error);
        message.error(error.response?.data?.detail || error.message || 'Upload failed');
      } finally {
        setUploading(false);
      }
    },
    onChange(info) {
      const { status } = info.file;
      if (status === 'done') {
        message.success(`${info.file.name} file uploaded successfully.`);
      } else if (status === 'error') {
        message.error(`${info.file.name} file upload failed.`);
      }
    },
  };

  return (
    <div>
      <Dragger {...uploadProps} style={{ marginBottom: 24 }}>
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">Click or drag files to upload</p>
        <p className="ant-upload-hint">
          Supports: CSV, JSON, Excel, PDF, Word (DOC/DOCX), Text, Markdown, Images
        </p>
        {uploading && (
          <div style={{ marginTop: 16 }}>
            <Progress percent={50} size="small" />
          </div>
        )}
      </Dragger>

      {uploadedFiles.length > 0 && (
        <Card title="Uploaded Files" size="small">
          <List
            dataSource={uploadedFiles}
            renderItem={(file) => (
              <List.Item>
                <Space align="center" style={{ width: '100%' }}>
                  {getFileIcon(file.file_type)}
                  <div style={{ flex: 1 }}>
                    <div>
                      <Text strong>{file.filename}</Text>
                    </div>
                    <div>
                      <Space>
                        <Text type="secondary">{formatFileSize(file.size)}</Text>
                        {file.preview?.type === 'document' && (
                          <Tag color="blue">{file.preview.format.toUpperCase()} Document</Tag>
                        )}
                        {file.preview?.type === 'image' && (
                          <Tag color="orange">Image</Tag>
                        )}
                        {file.preview?.rows && (
                          <Tag color="green">{file.preview.rows} rows</Tag>
                        )}
                        {file.preview?.message && (
                          <Text type="secondary" style={{ fontSize: 12 }}>{file.preview.message}</Text>
                        )}
                      </Space>
                    </div>
                    {file.preview?.content_preview && (
                      <div style={{ marginTop: 8 }}>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          Preview: {file.preview.content_preview.slice(0, 100)}...
                        </Text>
                      </div>
                    )}
                  </div>
                </Space>
              </List.Item>
            )}
          />
        </Card>
      )}
    </div>
  );
}