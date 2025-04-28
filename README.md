# PoetryModel

## Introduction

Recurrent Neural Networks (RNNs) are exceptionally powerful for processing and generating sequential data, such as text. In this project, we leverage Long Short-Term Memory (LSTM) networks to train models capable of generating poetry in the style of renowned English poets like William Shakespeare, William Wordsworth, and Robert Frost.

By training on authentic sonnets and poems, the models learn the patterns, rhythms, and structures characteristic of each poet's style. The models are designed to predict the next character based on a sequence of previous characters, allowing them to generate coherent and stylistically rich pieces of text. This project showcases how deep learning can be applied creatively to mimic literary artistry.

## Project Structure

```
PoetryModel/
├── William_Shakespeare/
│   ├── William_Shakespeare_sonnet.ipynb      # Jupyter Notebook for training and generating Shakespearean sonnets
│   ├── William_Shakespeare_sonnet.pdf        # PDF export of the notebook
│   ├── output_sonnet.txt                     # Generated Shakespearean sonnet
│   ├── sonnets.keras                         # Trained LSTM model
│   └── sonnets.txt                           # Training dataset (Shakespeare's sonnets)
│
├── William_Wordsworth_and_Robert_Frost/
│   ├── William_Wordsworth_and_Robert_Frost.ipynb  # Jupyter Notebook for training and generating poems
│   ├── William_Wordsworth_and_Robert_Frost.pdf    # PDF export of the notebook
│   ├── output.txt                                # Generated poem
│   ├── poemgenerator.keras                       # Trained LSTM model
│   └── poems.txt                                 # Training dataset (poems by Wordsworth and Frost)
│
├── Data Source.txt       # Information about data sources
└── README.md             # Project description and documentation
```

## Data Sources

- **William Shakespeare Sonnets**:  
  [GitHub Repository](https://github.com/martin-gorner/tensorflow-rnn-shakespeare/blob/master/shakespeare/sonnets.txt)  
  Approximately 2633 lines of sonnets written by William Shakespeare, the most influential playwright and poet in the English language.

- **Poems by William Wordsworth and Robert Frost**:  
  [Kaggle Dataset](https://www.kaggle.com/datasets/charunisa/english-poems-dataset?resource=download)  
  A curated collection of approximately 2000 lines of poems authored by two of the most celebrated figures in English literature.

## Requirements

To run this project, you will need:

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib (for optional visualization)

You can install the required packages using pip:

```bash
pip install tensorflow numpy matplotlib
```

We recommend running the Jupyter notebooks in a virtual environment to keep dependencies isolated:

```bash
python -m venv poetryenv
source poetryenv/bin/activate   # On Windows use `poetryenv\Scripts\activate`
pip install -r requirements.txt
```

*(Optional)*: If you wish, create a `requirements.txt` with:

```
tensorflow
numpy
matplotlib
```

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/abhiseksarkar2001/PoetryModel.git
cd PoetryModel
```

2. Open the desired Jupyter notebook:

```bash
jupyter notebook William_Shakespeare/William_Shakespeare_sonnet.ipynb
```

or

```bash
jupyter notebook William_Wordsworth_and_Robert_Frost/William_Wordsworth_and_Robert_Frost.ipynb
```

3. Run all cells to:

- Load the training data
- Preprocess the text
- Train an LSTM-based model
- Generate original poetry based on the learned style

4. The generated outputs are saved to text files (`output_sonnet.txt` and `output.txt`) for review.

## Highlights

- **Character-level text generation**: The model predicts the next character based on a sequence of previous characters.
- **Customizable training**: Parameters such as sequence length, batch size, and number of epochs can be easily adjusted.
- **Model saving and reloading**: Trained models are saved in Keras format (`.keras`) and can be reloaded for further text generation without retraining.
- **Creative AI Application**: Demonstrates how machine learning can be applied to emulate literary styles and create novel artistic content.

## Future Improvements

- Experimenting with different architectures like GRUs or Transformer-based models.
- Using word-level embeddings instead of character-level models for richer semantic understanding.
- Fine-tuning temperature sampling to control the creativity of generated poems.

## License

This project is for educational and research purposes.

---
