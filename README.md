# Product Hunt Post Success Prediction

This repository contains code and analysis for predicting the success of Product Hunt posts. It leverages historical data from the Product Hunt API, performs exploratory data analysis, and builds a classification model using machine learning.

## Project Structure

```
├── microtorch.py
├── char_level_model.py
├── helper_funcs.py
└── EDA_and_classification_model.ipynb
```

## Files and Descriptions

- **`EDA_and_classification_model.ipynb`:** This Jupyter Notebook contains the core analysis and model building process. It fetches Product Hunt post data, performs exploratory data analysis to uncover insights, preprocesses the data for modeling, trains an XGBoost classification model, and evaluates its performance.

- **`char_level_model.py`:** This script implements a character-level neural network language model using PyTorch. It's trained on a dataset of Reddit comments and can be used to generate text.

- **`helper_funcs.py`:** This file defines helper functions for interacting with the Product Hunt API and processing data. It includes functions for authentication, fetching posts, and constructing n-gram datasets.

- **`microtorch.py`:** This file defines a minimal PyTorch-like framework called "microtorch" for building and training neural networks. It includes classes for common neural network layers and activation functions.

## Key Features

- **Data Acquisition and Preprocessing:** Fetches data from the Product Hunt API, cleans and transforms it, and prepares it for analysis and modeling.
- **Exploratory Data Analysis (EDA):** Provides insights into Product Hunt posting patterns, success factors, and relationships between different features.
- **Classification Model:** Builds and trains an XGBoost model to predict the success level of Product Hunt posts based on their characteristics.
- **Character-Level Language Model:** Implements a neural network model for character-level text generation, trained on Reddit comments.
- **Modular Code:** Organizes code into reusable functions and classes for easier maintenance and extension.

## How to Use

1. Clone the repository: `git clone https://github.com/your-username/product-hunt-success-prediction.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Jupyter Notebook: `jupyter notebook EDA_and_classification_model.ipynb`
4. Explore the code and analysis.

## Future Work

- Experiment with different classification models and hyperparameters to improve prediction accuracy.
- Incorporate additional features, such as sentiment analysis of post descriptions and comments.
- Develop a web application to provide real-time predictions for new Product Hunt posts.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
