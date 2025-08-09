import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pickle
import os

# Load training data
df = pd.read_csv(os.path.join('jigsaw-toxic-comment-classification-challenge','train.csv', 'train.csv'))

# Recreate vectorizer with same parameters
MAX_FEATURES = 200000
vectorizer = TextVectorization(
    max_tokens=MAX_FEATURES,
    output_sequence_length=1800,
    output_mode='int'
)

# Adapt vectorizer to training data
X = df['comment_text']
vectorizer.adapt(X.values)

# Save vectorizer config and vocabulary
vectorizer_config = vectorizer.get_config()
vectorizer_weights = vectorizer.get_weights()

with open('vectorizer_config.pkl', 'wb') as f:
    pickle.dump(vectorizer_config, f)
    
with open('vectorizer_weights.pkl', 'wb') as f:
    pickle.dump(vectorizer_weights, f)

print("Vectorizer saved successfully!")