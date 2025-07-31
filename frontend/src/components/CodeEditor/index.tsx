import React from 'react';
import { Input } from 'antd';

const { TextArea } = Input;

interface CodeEditorProps {
  value?: string;
  onChange?: (value: string) => void;
  height?: string;
  defaultValue?: string;
  placeholder?: string;
  readOnly?: boolean;
}

const CodeEditor: React.FC<CodeEditorProps> = ({
  value,
  onChange,
  height = '200px',
  defaultValue,
  placeholder,
  readOnly = false,
}) => {
  return (
    <TextArea
      value={value}
      defaultValue={defaultValue}
      onChange={(e) => onChange?.(e.target.value)}
      placeholder={placeholder}
      readOnly={readOnly}
      style={{
        height,
        fontFamily: 'Monaco, Consolas, "Courier New", monospace',
        fontSize: '13px',
        lineHeight: '1.5',
        backgroundColor: '#f5f5f5',
      }}
    />
  );
};

export default CodeEditor;