# DatasetWizard User Guide

## Overview
The DatasetWizard is a user-friendly interface that allows you to generate training datasets from your documents with just a few clicks. AI (Claude) handles all the complex work of extracting content and transforming it into properly formatted training data.

## How to Use

1. **Upload Files**
   - Go to the Datasets page
   - Click "Create New Dataset"
   - Upload your files (PDFs, Word documents, CSVs, images, etc.)
   - Give your dataset a name and create it

2. **Generate Training Dataset**
   - Click the "Generate Dataset" button next to your uploaded dataset
   - The DatasetWizard modal will open

3. **Choose Dataset Type**
   - **Q&A Pairs**: For chatbots, FAQ systems, or question-answering models
   - **Text Classification**: For sentiment analysis, topic classification, or spam detection
   - **Named Entity Recognition**: For identifying people, places, organizations in text
   - **Text Summarization**: For creating text-summary pairs
   - **Custom Format**: Describe your own format and let AI create it

4. **Generate**
   - Click "Generate Dataset"
   - AI will:
     - Extract content from all your files
     - Analyze and understand the content
     - Generate high-quality training examples in your chosen format
   - Wait for the process to complete (usually 1-2 minutes)

5. **Download**
   - Once complete, you can download your dataset in CSV or JSON format
   - Preview the first few rows to ensure it meets your needs

## What Happens Behind the Scenes

1. **Document Extraction**: AI reads and extracts content from all file types
   - PDFs are parsed and text is extracted
   - Images are analyzed using Claude Vision
   - Word documents, CSVs, and other formats are processed

2. **Dataset Generation**: Claude analyzes the extracted content and creates:
   - Properly formatted training examples
   - Diverse, high-quality data points
   - Structured output ready for machine learning

## Tips for Best Results

- **Upload Related Content**: For best results, upload files that are related to your use case
- **Be Specific with Custom Format**: When using custom format, clearly describe what you need
- **Check the Preview**: Always review the preview before downloading to ensure quality

## Example Use Cases

- **Customer Support Bot**: Upload FAQs, support tickets, and documentation → Generate Q&A pairs
- **Document Classification**: Upload various document types → Generate classification training data
- **Information Extraction**: Upload reports and articles → Generate NER training data
- **Content Summarization**: Upload long documents → Generate summarization pairs

## Advanced Users

If you need more control over the pipeline, click the "Advanced" button to access the visual pipeline builder where you can customize each step of the process.