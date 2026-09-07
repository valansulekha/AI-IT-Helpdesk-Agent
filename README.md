# AI IT Helpdesk Agent



An AI-powered IT Helpdesk system that automatically understands user issues, retrieves relevant IT knowledge, provides solutions, creates support tickets, and escalates complex or sensitive issues to human support.



## Project Overview



Traditional IT helpdesk systems often require users to manually select categories and wait for support staff to respond.



The AI IT Helpdesk Agent automates this process using:



* Machine Learning for issue classification

* RAG (Retrieval-Augmented Generation) for knowledge retrieval

* An AI Agent to coordinate the workflow

* Tools for ticket creation, status management, and human escalation

* Explainable AI to show important terms influencing classification

* A dashboard for monitoring support tickets



## Workflow



```text

User Issue

    |

    v

AI Helpdesk Agent

    |

    v

ML Issue Classification

    |

    v

RAG Knowledge Retrieval

    |

    v

Solution Recommendation

    |

    v

Explainable AI

    |

    v

Tool Execution

    |

    +-------------------+--------------------+

    |                   |                    |

    v                   v                    v

Create Ticket      Update Status       Human Escalation

    |

    v

Support Dashboard

```



## Key Features



### 1. AI Issue Classification



The system automatically classifies IT problems into different categories using a Machine Learning model.



Supported categories:



* Network Issue

* Performance Issue

* Printer Issue

* Account Issue

* Security Issue

* Hardware Issue

* Email Issue

* Software Issue

* Access/Permission Issue

* Database Issue



The system also provides a classification confidence score.



### 2. RAG Knowledge Retrieval



The system uses Retrieval-Augmented Generation techniques to find the most relevant IT support article for the user's problem.



The knowledge base contains IT troubleshooting information such as:



* Network troubleshooting

* Password and account problems

* Printer problems

* Slow computers

* Security incidents

* Software problems

* Hardware troubleshooting



The system uses Sentence Transformers to generate embeddings and FAISS for similarity-based retrieval.



### 3. AI Helpdesk Agent



The HelpdeskAgent coordinates the complete support workflow.



It:



1. Receives the user's issue

2. Classifies the issue

3. Searches the knowledge base

4. Selects the most relevant solution

5. Determines ticket priority

6. Decides whether human review is required

7. Creates a support ticket

8. Escalates sensitive or uncertain issues



### 4. Tool-Based Actions



The agent can execute support actions through Python tools.



Available tools:



* `classify_issue()`

* `search_knowledge_base()`

* `create_ticket()`

* `get_ticket_status()`

* `update_ticket_status()`

* `escalate_to_human()`



### 5. Human Escalation



The system automatically requests human review when:



* The issue is a Security Issue

* Classification confidence is below 60%



This allows the system to handle routine issues automatically while sending uncertain or sensitive cases to human support.



### 6. Ticket Management



Every submitted issue can generate a unique support ticket.



Ticket information includes:



* Ticket ID

* Issue

* Category

* Confidence

* Priority

* Status

* Created time



Supported ticket statuses:



```text

Open

In Progress

Resolved

```



### 7. Explainable AI



The application displays important terms that influenced the ML classification.



This makes the prediction easier for users and administrators to understand instead of providing only a category and confidence score.



### 8. Support Dashboard



The Streamlit dashboard provides an overview of:



* Total tickets

* Open tickets

* In-progress tickets

* Resolved tickets

* Human review cases

* Complete ticket history



## Project Architecture



```text

                    +----------------+

                    |      User      |

                    |   IT Problem   |

                    +-------+--------+

                            |

                            v

                    +---------------+

                    | HelpdeskAgent |

                    |   agent.py    |

                    +-------+-------+

                            |

                 +----------+----------+

                 |                     |

                 v                     v

          +-------------+       +---------------+

          | ML          |       | RAG           |

          | Classifier  |       | Retrieval     |

          | model.py    |       | knowledge_    |

          |             |       | base.py       |

          +------+------+       +-------+-------+

                 |                      |

                 +----------+-----------+

                            |

                            v

                  +--------------------+

                  | Intelligent Support|

                  | Response           |

                  +---------+----------+

                            |

                            v

                     +-------------+

                     |    Tools    |

                     |  tools.py   |

                     +------+------+ 

                            |

              +-------------+-------------+

              |             |             |

              v             v             v

        Create Ticket  Update Status  Human Escalation

              |

              v

       +-------------------+

       | Support Dashboard |

       |      app.py       |

       +-------------------+

```



## Technologies Used



| Technology            | Purpose                               |

| --------------------- | ------------------------------------- |

| Python                | Core development                      |

| Streamlit             | Web application and dashboard         |

| Scikit-learn          | Machine Learning classification       |

| TF-IDF                | Text feature extraction               |

| Logistic Regression   | Issue classification                  |

| Sentence Transformers | Text embeddings                       |

| FAISS                 | Vector similarity search              |

| Pandas                | Data processing and ticket management |

| NumPy                 | Numerical operations                  |

| Joblib                | Model serialization                   |

| Git and GitHub        | Version control                       |



## Project Structure



```text

AI-IT-Helpdesk-Agent/

|

+-- app.py

+-- agent.py

+-- tools.py

+-- model.py

+-- knowledge_base.py

|

+-- helpdesk_model.pkl

+-- vectorizer.pkl

|

+-- it_knowledge.pkl

+-- it_knowledge.index

|

+-- tickets.csv

+-- requirements.txt

+-- README.md

+-- .gitignore

```



## Machine Learning Model



The issue classification model uses:



* TF-IDF text vectorization

* Unigrams and bigrams

* Logistic Regression

* Stratified train/test split

* Balanced class weights



### Model Performance



The current model achieved approximately:



**92.5% test accuracy**



The model predicts the IT issue category and returns a confidence score.



## RAG Implementation



The RAG component uses the following process:



```text

User Query

    |

    v

Sentence Transformer

    |

    v

Text Embedding

    |

    v

FAISS Similarity Search

    |

    v

Top Relevant Knowledge Articles

    |

    v

Best Matching Solution

```



The embedding model used is:



```text

all-MiniLM-L6-v2

```



## Installation



### 1. Clone the repository



```bash

git clone https://github.com/valansulekha/AI-IT-Helpdesk-Agent.git

```



### 2. Open the project folder



```bash

cd AI-IT-Helpdesk-Agent

```



### 3. Create a virtual environment



```bash

python -m venv venv

```



### 4. Activate the virtual environment



Windows:



```cmd

venv\Scripts\\activate

```



### 5. Install dependencies



```bash

pip install -r requirements.txt

```



### 6. Run the application



```bash

streamlit run app.py

```



The application will open in your browser.



## Application Modules



### AI Analysis



Displays:



* Predicted issue category

* Confidence score

* Priority

* Human review decision



### Knowledge Retrieval



Displays:



* Relevant knowledge article

* Category

* Similarity score

* Recommended solution



### Ticket Management



Allows the system to:



* Create tickets

* View ticket details

* Update ticket status

* Escalate issues



### Dashboard



Provides an overall view of the helpdesk ticket system.



## Security and Escalation



Security-related issues receive High priority and are automatically marked for human review.



This provides a safety mechanism where sensitive IT problems are not handled entirely automatically.



## Project Objective



The main objective of this project is to develop an intelligent IT support system that combines:



```text

Machine Learning

       +

RAG

       +

AI Agent

       +

Tools

       =

Intelligent IT Helpdesk

```



Instead of simply predicting an issue category, the system combines classification, knowledge retrieval, automated actions, and human escalation into one workflow.



## Future Enhancements



Possible future improvements include:



* Integration with real LLM APIs

* Real enterprise knowledge bases

* Email and notification integration

* User authentication

* Admin roles

* Cloud deployment

* Real-time monitoring

* Advanced analytics

* Integration with enterprise ticketing systems

* Multilingual IT support



## Project



**AI IT Helpdesk Agent**



Developed as an AI/IT support automation project demonstrating Machine Learning, RAG, Agent orchestration, and tool-based automation.



## License



This project is intended for educational and demonstration purposes.



