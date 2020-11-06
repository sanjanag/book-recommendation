# Book Recommendation System

## Introduction and Background
Goodreads is a social cataloging website where users can review books, maintain their libraries, and generate polls, blogs, and surveys. The purpose of this project is to build a recommendation system for Goodreads that helps users find the perfect book to read. This task can be broken down into two parts: Predicting a book’s rating based on certain attributes and recommending books based on a user’s prior interaction and ratings. Recommender systems employ different approaches such as collaborative filtering, content-based filtering, a hybrid of both, or knowledge-based systems to generate recommendations for a user. Predicting the rating of a book employs supervised techniques such as Linear Regression and Neural Networks.
 


## Problem Definition
We have a set of books *B* with attributes of each book like description, author, number of pages and so on. We also have an interaction matrix *I*, where *I*[*u*][*b*] entry tells what rating user u gave to book *b*. Using these two sets of information we aim to solve the following tasks.
- Given a book *b* with its attributes, predict it's average rating in the range of 0 to 5.
- Given a user's prior interaction with the books i.e. *I*[*u*], suggest top *n* recommendations for that user.

## Data Collection




## Data Exploration

<img src="images/regression/avg_rating.png" alt="hi" class="inline"/>

<img src="images/regression/correlation.png" alt="hi" class="inline"/>

<img src="images/regression/pairplot.png" alt="hi" class="inline"/>
 
## Feature Selection

<img src="images/regression/lang_vs_rating.png" alt="hi" class="inline"/>

## Methods
### Dataset
For our project, we will be using a Goodreads dataset released by UCSD that was collected in late 2017 by scraping data off of the public shelves of users. The main dataset has data of about 2,300,000 books and 900,000 users. The books are divided into different genres such as Children, Young Adult, Comics, Fantasy, History, etc. Since the original dataset is very large, we will be using a subset of books from each genre dataset. For recommendations, we will be using the user-book interaction dataset, which contains information such as user ID, book ID, rating score, and book review.

### Algorithms
- **Supervised** <br>
For predicting ratings we plan to use a supervised approach. We plan to use linear regression as a baseline [1] and then move on to neural networks [2].

- **Unsupervised** <br>
For the task of recommending books to users, we will be experimenting with unsupervised approaches in the following two paradigms.
    1. **Content-based filtering** <br>
Content-based filtering makes recommendations to users based on their preferences for content. For this, we’ll be modeling each book using a vector of TF-IDF features of the description of the book and check the similarity of a book with the user’s prior rated books and return recommendations based on the highest similarity.
    2. **Collaborative filtering** <br>
Collaborative filtering makes recommendations based on other users’ ratings along with the user in question. We plan to use matrix factorization for this approach.

### Experiments
We plan to assess both traditional and neural network classifiers, to determine the best model for our predicting ratings. We also aim to build an perform an ablation study on metadata features to see if they enhance the predictions in addition to text. For the recommnedations, we plan to contrast content filtering and collaborative filtering methods.

### Evaluation Metrics
We will use Root-Mean-Squared-Error (RMSE) to determine the accuracy of our rating prediction algorithm, and Normalized-Discounted-Cumulative-Gain (nDCG) to evaluate the recommendations.

## Potential results
We expect to find that using both metadata and textual features will give a superior performance for the predicting ratings. We also expect that the content-based filtering approach might be better at predicting individual user ratings since they are specific to a user. On the contrary, we expect the collaborative filtering approach to be more diverse since it models user behavior instead of a specific user.

## Discussion
The objective of this project is to use machine learning models to first identify a personalized reading list and then predict the rating of books. Some potential difficulties could be caused by the computational complexity associated with neural networks. We also anticipate some challenges related to hyperparameter tuning for matrix factorization and our ability to handle first time users. Overall, we believe our project would help us explore various supervised and unsupervised learning techniques and use insights to build a recommendation system.


## References

[1] Hsu PY., Shen YH., Xie XA. (2014) Predicting Movies User Ratings with Imdb Attributes. In: Miao D., Pedrycz W., Ślȩzak D., Peters G., Hu Q., Wang R. (eds) Rough Sets and Knowledge Technology. RSKT 2014. Lecture Notes in Computer Science, vol 8818. Springer, Cham.  <br>
[2] Logé, C., Yoffe, A., & ceciloge Building the optimal Book Recommender and measuring the role of Book Covers in predicting user ratings.<br>
[3] Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", in RecSys'18.<br>
[4] Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19.
