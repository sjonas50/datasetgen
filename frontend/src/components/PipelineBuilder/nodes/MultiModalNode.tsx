import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Card, Tag, Badge } from 'antd';
import { FileImageOutlined, FilePdfOutlined, MergeCellsOutlined } from '@ant-design/icons';
import styles from './StepNode.module.css';

interface MultiModalNodeData {
  label: string;
  config?: {
    type?: string;
    fusion_strategy?: string;
    analysis_type?: string;
  };
  status?: 'idle' | 'running' | 'completed' | 'failed';
}

const iconMap: Record<string, React.ReactNode> = {
  document_parse: <FileImageOutlined />,
  pdf_table_extract: <FilePdfOutlined />,
  multimodal_fusion: <MergeCellsOutlined />,
};

const MultiModalNode: React.FC<NodeProps<MultiModalNodeData>> = ({ data, selected }) => {
  const nodeType = data.config?.type || 'document_parse';
  const fusionStrategy = data.config?.fusion_strategy;
  const statusClass = data.status ? styles[`status-${data.status}`] : '';
  const icon = iconMap[nodeType] || <FileImageOutlined />;

  return (
    <Card
      className={`${styles.stepNode} ${styles.multimodalNode} ${statusClass} ${selected ? styles.selected : ''}`}
      size="small"
    >
      <Handle
        type="target"
        position={Position.Top}
        className={styles.handle}
      />
      <div className={styles.content}>
        <div className={styles.icon} style={{ position: 'relative' }}>
          <span style={{ color: '#fa8c16' }}>{icon}</span>
          <Badge 
            count="MM" 
            style={{ 
              backgroundColor: '#52c41a',
              fontSize: 8,
              minWidth: 20,
              height: 14,
              lineHeight: '14px',
              position: 'absolute',
              top: -8,
              right: -12
            }} 
          />
        </div>
        <div className={styles.label}>{data.label}</div>
        <div className={styles.tags}>
          <Tag color="orange" style={{ fontSize: "12px" }}>
            {nodeType.replace(/_/g, ' ')}
          </Tag>
          {fusionStrategy && (
            <Tag color="gold" style={{ fontSize: "12px" }}>
              {fusionStrategy}
            </Tag>
          )}
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        className={styles.handle}
      />
    </Card>
  );
};

export default memo(MultiModalNode);