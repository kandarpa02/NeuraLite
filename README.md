# NeuraLite: A Minimalistic Approach to Machine Learning
![](images/image1.jpeg)

**NeuraLite** is an exploration of applying classical Machine Learning algorithms to hand written digit classification tasks on the MNIST dataset, using only `Numpy` and a bit of mathematical theory. Unlike Neural Networks, which often require heavy configurations such as CUDA, ROCm, and high-performance hardware, **NeuraLite** uses the simplicity of logistic regression for efficient classification. 

While deep learning techniques, like Neural Networks, often give near-perfect accuracy, **NeuraLite** focuses on achieving competitive results with minimal computational overhead. It’s not about replacing Neural Networks, but rather about testing the boundaries of traditional Machine Learning methods.

### Performance
While modern Neural Networks typically deliver near 100% accuracy, **NeuraLite** achieved around **90%** accuracy after hours of training and hyperparameter tuning. It's a great example of how simple algorithms can still perform well on complex tasks when optimized properly.

### Challenges
Building a logistic regression model from scratch without relying on high-level Machine Learning libraries (like Sklearn or TensorFlow) presented its challenges. Implementing mathematical equations for optimization and backpropagation was both a fun and difficult experience. However, by sticking to the basics and utilizing **Numpy**, I was able to create an efficient and functioning model.

### Thoughts
Although building algorithms from the ground up is an excellent way to learn, I do not recommend using this approach in a production environment. Writing math equations from scratch for each project can be inefficient, especially when trying to find the best model and tuning hyperparameters. However, this project served as a valuable learning experience, allowing me to gain deeper insights into Machine Learning techniques while keeping things fun and hands-on.

### Key Features
- **Lightweight Model**: Uses logistic regression for efficient training and inference.
- **Pure Numpy**: No Machine Learning frameworks like TensorFlow or Scikit-Learn—just pure math and Numpy.
- **Simple and Elegant**: Focuses on understanding the core concepts behind classification tasks.

### Requirements
1. Python
2. Numpy
3. Matplotlib
4. Streamlit (for web deployment)

### How to Run
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/kandarpa02/NeuraLite.git
    cd NeuraLite
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```

4. Upload an image for classification and see the results!
