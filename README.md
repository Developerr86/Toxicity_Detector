# Comment Toxicity Detector - Streamlit App

A user-friendly web application for detecting toxicity in text comments using a trained deep learning model.

## Features

- 🛡️ **Real-time Toxicity Detection**: Analyze text comments for 6 types of toxicity
- 📊 **Detailed Results**: View probability scores and binary predictions
- 🎯 **Visual Feedback**: Progress bars and color-coded results
- 📝 **Example Texts**: Try predefined examples to test the model
- 🎭 **Overall Assessment**: Get a summary of the toxicity analysis

## Toxicity Categories

The model detects the following types of toxic content:

1. **Toxic** - General toxicity
2. **Severe Toxic** - Severely toxic content
3. **Obscene** - Obscene language
4. **Threat** - Threatening language
5. **Insult** - Insulting content
6. **Identity Hate** - Identity-based hate speech

## Prerequisites

Make sure you have the following files in your project directory:

- `toxicity.h5` - The trained model file
- `jigsaw-toxic-comment-classification-challenge/train.csv/train.csv` - Training dataset (needed to re/create the vectorizer)
- `streamlit_app.py` - The Streamlit application
- `requirements.txt` - Python dependencies

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify file structure:**
   ```
   CommentToxicity/
   ├── streamlit_app.py
   ├── toxicity.h5
   ├── requirements.txt
   ├── jigsaw-toxic-comment-classification-challenge/
   │   └── train.csv/
   │       └── train.csv
   └── README_STREAMLIT.md
   ```

## Running the App

1. **Navigate to the project directory:**
   ```bash
   cd /path/to/CommentToxicity
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser:**
   - The app will automatically open in your default browser
   - If not, navigate to `http://localhost:8501`

## How to Use

1. **Enter Text**: Type or paste the comment you want to analyze in the text area
2. **Analyze**: Click the "🔍 Analyze Toxicity" button
3. **View Results**: 
   - See probability scores for each toxicity category
   - Check binary predictions (>50% threshold)
   - Review the overall assessment
   - Examine visual breakdown with progress bars

## Example Usage

### Clean Comment
```
"Thank you for your helpful contribution to this discussion!"
```
**Expected Result**: All categories show low probability scores

### Toxic Comment
```
"You are such an idiot and I hate everything you say!"
```
**Expected Result**: High scores for "toxic" and "insult" categories

## Technical Details

- **Model**: Bidirectional LSTM with Dense layers
- **Framework**: TensorFlow/Keras
- **Interface**: Streamlit
- **Threshold**: 0.5 (50%) for binary classification
- **Vocabulary**: 200,000 tokens
- **Sequence Length**: 1,800 tokens

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure `toxicity.h5` exists in the project directory
   - Check that the file is not corrupted

2. **Vectorizer Error**
   - Verify the training dataset path is correct
   - Ensure `train.csv` is accessible

3. **Import Errors**
   - Install all required dependencies: `pip install -r requirements.txt`
   - Check TensorFlow compatibility with your system

4. **Performance Issues**
   - The first prediction may take longer due to model loading
   - Subsequent predictions should be faster due to caching

### Performance Notes

- Model and vectorizer are cached for better performance
- First load may take 10-30 seconds depending on your system
- Predictions are typically fast (<1 second) after initial load

## Customization

You can customize the app by modifying `streamlit_app.py`:

- Change the threshold for binary classification (currently 0.5)
- Modify the UI layout and styling
- Add new example texts
- Adjust the progress bar colors and styling

## License

This project uses the Jigsaw Toxic Comment Classification dataset and is intended for educational purposes.
