# Introduction

In this book, we explore mid-price prediction in financial markets through the combined lens of statistical filtering and machine learning. The mid-price (halfway between the best bid and best ask) captures the evolving consensus of market participants and serves as a natural target for short-term price forecasting.

We begin by implementing a Kalman Filter as a statistical baseline for sequential state estimation. From there, we train and evaluate a range of machine learning models to assess how modern approaches compare with classical inference methods.

Our goals are two-fold:

- Implement classical inference algorithms such as the Kalman Filter in C++ with Python bindings for experimentation.
- Compare these algorithms against machine learning models in terms of predictive accuracy, robustness, and computational performance.

The comparison will be carried out on classical mid-price forecasting datasets, including:
- FI-2010: a publicly available benchmark dataset for mid-price forecasting for limit order book data
- LOBster: a real limit order book dataset with millisecond-level resolution

By the end, we should have a practical understanding of how statistical filters and machine learning can be applied to mid-price prediction.