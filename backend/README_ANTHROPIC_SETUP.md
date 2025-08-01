# Setting up Anthropic API for DatasetGen

## Quick Setup

1. **Get your Anthropic API key**:
   - Go to https://console.anthropic.com/
   - Sign up or log in
   - Navigate to API Keys section
   - Create a new API key
   - Copy the key (it starts with `sk-ant-api...`)

2. **Add the API key to your environment**:
   
   Option A: Update the .env file
   ```bash
   # Edit /Users/sjonas/dataset-gen/backend/.env
   ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
   ```
   
   Option B: Export as environment variable
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-api03-your-actual-key-here"
   ```

3. **Restart the backend**:
   ```bash
   # Kill the current process
   pkill -f "python.*main_full"
   
   # Start with the new environment
   cd /Users/sjonas/dataset-gen/backend
   python3 main_full.py
   ```

## Testing the AI Features

1. Upload a document (PDF, Word, CSV, etc.)
2. Create a dataset
3. Click "Smart Pipeline"
4. The AI will analyze your documents and suggest a pipeline

## What the AI Analyzer Does

- **Document Analysis**: Examines file types, content structure, and data patterns
- **Data Type Detection**: Identifies structured vs unstructured data
- **Quality Assessment**: Detects missing values, duplicates, formatting issues
- **PII Detection**: Identifies potential sensitive information
- **Pipeline Generation**: Creates appropriate processing steps based on your data

## Troubleshooting

If you see "AI features disabled":
- Check that ANTHROPIC_API_KEY is set correctly
- Ensure the API key is valid and has credits
- Check the backend logs for any error messages

## Security Note

- Never commit your API key to version control
- The .env file is already in .gitignore
- For production, use environment variables or a secrets manager