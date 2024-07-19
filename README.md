<div align="center">
<h1 align="center">RAG-GPT</h1>
Quickly launch an intelligent customer service system with Flask, LLM, RAG, including frontend, backend, and admin console.
<br/>
<a href="https://langchain-bot.open-sora.ai/" target="_blank"> Live Demo </a>
<br/>
<img style="display: block; margin: auto; width: 70%;" src="./doc/rag_overview.jpg">
</div>


# Contents

- [Features](#features)
- [Online Retrieval Architecture](#online-retrieval-architecture)
- [Deploy the RAG-GPT Service](#deploy-the-rag-gpt-service)
  - [Step 1: Download repository code](#step-1-download-repository-code)
  - [Step 2: Configure variables of .env](#step-2-configure-variables-of-env)
    - [Using OpenAI as the LLM base](#using-openai-as-the-llm-base)
    - [Using ZhipuAI as the LLM base](#using-zhipuai-as-the-llm-base)
    - [Using DeepSeek as the LLM base](#using-deepseek-as-the-llm-base)
    - [Using Moonshot as the LLM base](#using-moonshot-as-the-llm-base)
    - [Using local LLMs](#using-local-llms)
  - [Step 3: Deploy RAG-GPT](#step-3-deploy-rag-gpt)
    - [Deploy RAG-GPT using Docker](#deploy-rag-gpt-using-docker)
    - [Deploy RAG-GPT from source code](#deploy-rag-gpt-from-source-code)
      - [Set up the Python running environment](#set-up-the-python-running-environment)
        - [Create and activate a virtual environment](#create-and-activate-a-virtual-environment)
        - [Install dependencies with pip](#install-dependencies-with-pip)
      - [Create SQLite Database](#create-sqlite-database)
      - [Start the service](#start-the-service)
- [Configure the admin console](#configure-the-admin-console)
  - [Login to the admin console](#login-to-the-admin-console)
  - [Import your data](#import-your-data)
    - [import websites](#import-websites)
    - [import isolated urls](#import-isolated-urls)
    - [import local files](#import-local-files)
  - [Test the chatbot](#test-the-chatbot)
  - [Embed on your website](#embed-on-your-website)
  - [Dashboard of user's historical request](#dashboard-of-users-historical-request)
- [The frontend of admin console and chatbot](#the-frontend-of-admin-console-and-chatbot)
  - [admin console](#admin-console)
  - [chatbot](#chatbot)


## Features
- **Built-in LLM Support**: Support cloud-based LLMs and local LLMs.
- **Quick Setup**: Enables deployment of production-level conversational service robots within just five minutes.
- **Diverse Knowledge Base Integration**: Supports multiple types of knowledge bases, including websites, isolated URLs, and local files.
- **Flexible Configuration**: Offers a user-friendly backend equipped with customizable settings for streamlined management.
- **Attractive UI**: Features a customizable and visually appealing user interface.


## Online Retrieval Architecture

<div align="center">
<img style="display: block; margin: auto; width: 100%;" src="./doc/online_retrieve.jpg">
</div>


## Deploy the RAG-GPT Service

### Step 1: Download repository code

Clone the repository:

```shell
git clone https://github.com/open-kf/rag-gpt.git && cd rag-gpt
```

### Step 2: Configure variables of .env

Before starting the RAG-GPT service, you need to modify the related configurations for the program to initialize correctly. 

#### Using OpenAI as the LLM base

```shell
cp env_of_openai .env
```

The variables in .env

```shell
LLM_NAME="OpenAI"
OPENAI_API_KEY="xxxx"
GPT_MODEL_NAME="gpt-4o-mini"
MIN_RELEVANCE_SCORE=0.4
BOT_TOPIC="xxxx"
URL_PREFIX="http://127.0.0.1:7000/"
USE_PREPROCESS_QUERY=1
USE_RERANKING=1
USE_DEBUG=0
USE_LLAMA_PARSE=0
LLAMA_CLOUD_API_KEY="xxxx"
USE_GPT4O=0
```

- Don't modify **`LLM_NAME`**
- Modify the **`OPENAI_API_KEY`** with your own key. Please log in to the [OpenAI website](https://platform.openai.com/api-keys) to view your API Key.
- Update the **`GPT_MODEL_NAME`** setting, replacing `gpt-4o-mini` with `gpt-4-turbo` or `gpt-4o` if you want to use GPT-4.
- Change **`BOT_TOPIC`** to reflect your Bot's name. This is very important, as it will be used in `Prompt Construction`. Please try to use a concise and clear word, such as `OpenIM`, `LangChain`.
- Adjust **`URL_PREFIX`** to match your website's domain. This is mainly for generating accessible URL links for uploaded local files. Such as `http://127.0.0.1:7000/web/download_dir/2024_05_20/d3a01d6a-90cd-4c2a-b926-9cda12466caf/openssl-cookbook.pdf`.
- Set **`USE_LLAMA_PARSE`** to 1 if you want to use `LlamaParse`.
- Modify the **`LLAMA_CLOUD_API_KEY `** with your own key. Please log in to the [LLamaCloud website](https://cloud.llamaindex.ai/api-key) to view your API Key.
- Set **`USE_GPT4O`** to 1 if you want to use `GPT-4o` mode.
- For more information about the meanings and usages of constants, you can check under the `server/constant` directory.

#### Using ZhipuAI as the LLM base

If you cannot use OpenAI's API services, consider using ZhipuAI as an alternative. 


```shell
cp env_of_zhipuai .env
```

The variables in .env

```shell
LLM_NAME="ZhipuAI"
ZHIPUAI_API_KEY="xxxx"
GLM_MODEL_NAME="glm-4-air"
MIN_RELEVANCE_SCORE=0.4
BOT_TOPIC="xxxx"
URL_PREFIX="http://127.0.0.1:7000/"
USE_PREPROCESS_QUERY=1
USE_RERANKING=1
USE_DEBUG=0
USE_LLAMA_PARSE=0
LLAMA_CLOUD_API_KEY="xxxx"
```

- Don't modify **`LLM_NAME`**
- Modify the **`ZHIPUAI_API_KEY`** with your own key. Please log in to the [ZhipuAI website](https://open.bigmodel.cn/usercenter/apikeys) to view your API Key.
- Update the **`GLM_MODEL_NAME`** setting, the model list is `['glm-3-turbo', 'glm-4', 'glm-4-0520', 'glm-4-air', 'glm-4-airx', 'glm-4-flash']`.
- Change **`BOT_TOPIC`** to reflect your Bot's name. This is very important, as it will be used in `Prompt Construction`. Please try to use a concise and clear word, such as `OpenIM`, `LangChain`.
- Adjust **`URL_PREFIX`** to match your website's domain. This is mainly for generating accessible URL links for uploaded local files. Such as `http://127.0.0.1:7000/web/download_dir/2024_05_20/d3a01d6a-90cd-4c2a-b926-9cda12466caf/openssl-cookbook.pdf`.
- Set **`USE_LLAMA_PARSE`** to 1 if you want to use `LlamaParse`.
- Modify the **`LLAMA_CLOUD_API_KEY `** with your own key. Please log in to the [LLamaCloud website](https://cloud.llamaindex.ai/api-key) to view your API Key.
- For more information about the meanings and usages of constants, you can check under the `server/constant` directory.

#### Using DeepSeek as the LLM base

If you cannot use OpenAI's API services, consider using DeepSeek as an alternative.

> [!NOTE]
> DeepSeek does not provide an `Embedding API`, so here we use ZhipuAI's `Embedding API`.


```shell
cp env_of_deepseek .env
```

The variables in .env

```shell
LLM_NAME="DeepSeek"
ZHIPUAI_API_KEY="xxxx"
DEEPSEEK_API_KEY="xxxx"
DEEPSEEK_MODEL_NAME="deepseek-chat"
MIN_RELEVANCE_SCORE=0.4
BOT_TOPIC="xxxx"
URL_PREFIX="http://127.0.0.1:7000/"
USE_PREPROCESS_QUERY=1
USE_RERANKING=1
USE_DEBUG=0
USE_LLAMA_PARSE=0
LLAMA_CLOUD_API_KEY="xxxx"
```

- Don't modify **`LLM_NAME`**
- Modify the **`ZHIPUAI_API_KEY`** with your own key. Please log in to the [ZhipuAI website](https://open.bigmodel.cn/usercenter/apikeys) to view your API Key.
- Modify the **`DEEPKSEEK_API_KEY`** with your own key. Please log in to the [DeepSeek website](https://platform.deepseek.com/api_keys) to view your API Key.
- Update the **`DEEPSEEK_MODEL_NAME `** setting if you want to use other models of DeepSeek.
- Change **`BOT_TOPIC`** to reflect your Bot's name. This is very important, as it will be used in `Prompt Construction`. Please try to use a concise and clear word, such as `OpenIM`, `LangChain`.
- Adjust **`URL_PREFIX`** to match your website's domain. This is mainly for generating accessible URL links for uploaded local files. Such as `http://127.0.0.1:7000/web/download_dir/2024_05_20/d3a01d6a-90cd-4c2a-b926-9cda12466caf/openssl-cookbook.pdf`.
- Set **`USE_LLAMA_PARSE`** to 1 if you want to use `LlamaParse`.
- Modify the **`LLAMA_CLOUD_API_KEY `** with your own key. Please log in to the [LLamaCloud website](https://cloud.llamaindex.ai/api-key) to view your API Key.
- For more information about the meanings and usages of constants, you can check under the `server/constant` directory.


#### Using Moonshot as the LLM base

If you cannot use OpenAI's API services, consider using Moonshot as an alternative.

> [!NOTE]
> Moonshot does not provide an `Embedding API`, so here we use ZhipuAI's `Embedding API`.


```shell
cp env_of_moonshot .env
```

The variables in .env

```shell
LLM_NAME="Moonshot"
ZHIPUAI_API_KEY="xxxx"
MOONSHOT_API_KEY="xxxx"
MOONSHOT_MODEL_NAME="moonshot-v1-8k"
MIN_RELEVANCE_SCORE=0.4
BOT_TOPIC="xxxx"
URL_PREFIX="http://127.0.0.1:7000/"
USE_PREPROCESS_QUERY=1
USE_RERANKING=1
USE_DEBUG=0
USE_LLAMA_PARSE=0
LLAMA_CLOUD_API_KEY="xxxx"
```

- Don't modify **`LLM_NAME`**
- Modify the **`ZHIPUAI_API_KEY`** with your own key. Please log in to the [ZhipuAI website](https://open.bigmodel.cn/usercenter/apikeys) to view your API Key.
- Modify the **`MOONSHOT_API_KEY`** with your own key. Please log in to the [Moonshot website](https://platform.moonshot.cn/console/api-keys) to view your API Key.
- Update the **`MOONSHOT_MODEL_NAME `** setting if you want to use other models of Moonshot.
- Change **`BOT_TOPIC`** to reflect your Bot's name. This is very important, as it will be used in `Prompt Construction`. Please try to use a concise and clear word, such as `OpenIM`, `LangChain`.
- Adjust **`URL_PREFIX`** to match your website's domain. This is mainly for generating accessible URL links for uploaded local files. Such as `http://127.0.0.1:7000/web/download_dir/2024_05_20/d3a01d6a-90cd-4c2a-b926-9cda12466caf/openssl-cookbook.pdf`.
- Set **`USE_LLAMA_PARSE`** to 1 if you want to use `LlamaParse`.
- Modify the **`LLAMA_CLOUD_API_KEY `** with your own key. Please log in to the [LLamaCloud website](https://cloud.llamaindex.ai/api-key) to view your API Key.
- For more information about the meanings and usages of constants, you can check under the `server/constant` directory.


#### Using local LLMs

If your knowledge base involves **sensitive information** and you prefer not to use cloud-based LLMs, consider using `Ollama` to deploy large models locally.


> [!NOTE]
> First, refer to [ollama](https://github.com/ollama/ollama) to **Install Ollama**, and download the embedding model `mxbai-embed-large` and the LLM model such as `llama3`.


```shell
cp env_of_ollama .env
```

The variables in .env

```shell
LLM_NAME="Ollama"
OLLAMA_MODEL_NAME="xxxx"
OLLAMA_BASE_URL="http://127.0.0.1:11434"
MIN_RELEVANCE_SCORE=0.4
BOT_TOPIC="xxxx"
URL_PREFIX="http://127.0.0.1:7000/"
USE_PREPROCESS_QUERY=1
USE_RERANKING=1
USE_DEBUG=0
USE_LLAMA_PARSE=0
LLAMA_CLOUD_API_KEY="xxxx"
```

- Don't modify **`LLM_NAME`**
- Update the **`OLLAMA_MODEL_NAME `** setting, select an appropriate model from [ollama library](https://ollama.com/library).
- If you have changed the default `IP:PORT` when starting `Ollama`, please update **`OLLAMA_BASE_URL`**. Please pay special attention, only enter the IP (domain) and PORT here, without appending a URI.
- Change **`BOT_TOPIC`** to reflect your Bot's name. This is very important, as it will be used in `Prompt Construction`. Please try to use a concise and clear word, such as `OpenIM`, `LangChain`.
- Adjust **`URL_PREFIX`** to match your website's domain. This is mainly for generating accessible URL links for uploaded local files. Such as `http://127.0.0.1:7000/web/download_dir/2024_05_20/d3a01d6a-90cd-4c2a-b926-9cda12466caf/openssl-cookbook.pdf`.
- Set **`USE_LLAMA_PARSE`** to 1 if you want to use `LlamaParse`.
- Modify the **`LLAMA_CLOUD_API_KEY `** with your own key. Please log in to the [LLamaCloud website](https://cloud.llamaindex.ai/api-key) to view your API Key.
- For more information about the meanings and usages of constants, you can check under the `server/constant` directory.


### Step 3: Deploy RAG-GPT
#### Deploy RAG-GPT using Docker

> [!NOTE]
> When deploying with Docker, pay special attention to the host of **URL_PREFIX** in the `.env` file. If using `Ollama`, also pay special attention to the host of **OLLAMA_BASE_URL** in the `.env` file. They need to use the actual IP address of the host machine.


```shell
docker-compose up --build
```

#### Deploy RAG-GPT from source code

> [!NOTE]
> Please use Python version 3.10.x or above.

##### Set up the Python running environment

It is recommended to install Python-related dependencies in a Python virtual environment to avoid affecting dependencies of other projects.

###### Create and activate a virtual environment

If you have not yet created a virtual environment, you can create one with the following command:

```shell
python3 -m venv myenv
```

After creation, activate the virtual environment:

```shell
source myenv/bin/activate
```

###### Install dependencies with pip

Once the virtual environment is activated, you can use `pip` to install the required dependencies. 

```shell
pip install -r requirements.txt
```

##### Create SQLite Database

The RAG-GPT service uses SQLite as its storage DB. Before starting the RAG-GPT service, you need to execute the following command to initialize the database and add the default configuration for admin console.

```shell
python3 create_sqlite_db.py
```

##### Start the service

If you have completed the steps above, you can try to start the RAG-GPT service by executing the following command.

- **Start single process:**

```shell
python3 rag_gpt_app.py
```

- **Start multiple processes:**

```shell
sh start.sh
```

> [!NOTE]
> - The service port for RAG-GPT is **`7000`**. During the first test, please try not to change the port so that you can quickly experience the entire product process.
> - We recommend starting the RAG-GPT service using **`start.sh`** in multi-process mode for a smoother user experience.



## Configure the admin console

### Login to the admin console

Access the admin console through the link **`http://your-server-ip:7000/open-kf-admin/`** to reach the login page. The default username and password are **`admin`** and **`open_kf_AIGC@2024`** (can be checked in `create_sqlite_db.py`).

<div align="center">
<img style="display: block; margin: auto; width: 70%;" src="./doc/screenshot-2.jpg">
</div>

After logging in successfully, you will be able to see the configuration page of the admin console.

<div align="center">
<img style="display: block; margin: auto; width: 70%;" src="./doc/screenshot-3.jpg">
</div>

On the page **`http://your-server-ip:7000/open-kf-admin/#/`**, you can set the following configurations:
- Choose the LLM base, currently only the `gpt-3.5-turbo` option is available, which will be gradually expanded.
- Initial Messages
- Suggested Messages
- Message Placeholder
- Profile Picture (upload a picture)
- Display name
- Chat icon (upload a picture)

### Import your data

#### Import websites

After submitting the website URL, once the server retrieves the list of all web page URLs via crawling, you can select the web page URLs you need as the knowledge base (all selected by default). The initial `Status` is **`Recorded`**.

<div align="center">
<img style="display: block; margin: auto; width: 70%;" src="./doc/screenshot-12.jpg">
</div>

You can actively refresh the page **`http://your-server-ip:7000/open-kf-admin/#/source`** in your browser to get the progress of web page URL processing. After the content of the web page URL has been crawled, and the Embedding calculation and storage are completed, you can see the corresponding `Size` in the admin console, and the `Status` will also be updated to **`Trained`**.

<div align="center">
<img style="display: block; margin: auto; width: 70%;" src="./doc/screenshot-13.jpg">
</div>

Clicking on a webpage's URL reveals how many sub-pages the webpage is divided into, and the text size of each sub-page.

<div align="center">
<img style="display: block; margin: auto; width: 70%;" src="./doc/screenshot-10.jpg">
</div>

Clicking on a sub-page allows you to view its full text content. This will be very helpful for verifying the effects during the experience testing process.

<div align="center">
<img style="display: block; margin: auto; width: 70%;" src="./doc/screenshot-11.jpg">
</div>

#### Import isolated urls

Collect the URLs of the required web pages. You can submit up to `10` web page URLs at a time, and these pages can be from different domains.

<div align="center">
<img style="display: block; margin: auto; width: 70%;" src="./doc/screenshot-14.jpg">
</div>

#### Import local files

Upload the required local files. You can upload up to `10` files at a time, and each file cannot exceed `30MB`. The following file types are currently supported: `[".txt", ".md", ".pdf", ".epub", ".mobi", ".html", ".docx", ".pptx", ".xlsx", ".csv"]`.

<div align="center">
<img style="display: block; margin: auto; width: 70%;" src="./doc/screenshot-15.jpg">
</div>


### Test the chatbot

After importing website data in the admin console, you can experience the chatbot service through the link **`http://your-server-ip:7000/open-kf-chatbot/`**.

<div align="center">
<img style="display: block; margin: auto; width: 70%;" src="./doc/screenshot-6.jpg">
</div>

### Embed on your website

Through the admin console link **`http://your-server-ip:7000/open-kf-admin/#/embed`**, you can see the detailed tutorial for configuring the iframe in your website.

<div align="center">
<img style="display: block; margin: auto; width: 70%;" src="./doc/screenshot-7.jpg">
</div>

<div align="center">
<br/>
<a href="https://docs.openim.io/" target="_blank"> OpenIM chatbot </a>
<br/>
<img style="display: block; margin: auto; width: 70%;" src="./doc/screenshot-8.jpg">
</div>

### Dashboard of user's historical request

Through the admin console link **`http://your-server-ip:7000/open-kf-admin/#/dashboard`**, you can view the historical request records of all users within a specified time range.

<div align="center">
<img style="display: block; margin: auto; width: 70%;" src="./doc/screenshot-9.jpg">
</div>


## The frontend of admin console and chatbot
The RAG-GPT service integrates 2 frontend modules, and their source code information is as follows:

### admin console
> [Code Repository](https://github.com/open-kf/smart-qa-admin)

An intuitive web-based admin interface for Smart QA Service, offering comprehensive control over content, configuration, and user interactions. Enables effortless management of the knowledge base, real-time monitoring of queries and feedback, and continuous improvement based on user insights.

### chatbot
> [Code Repository](https://github.com/open-kf/smart-qa-h5)

An HTML5 interface for Smart QA Service designed for easy integration into websites via iframe, providing users direct access to a tailored knowledge base without leaving the site, enhancing functionality and immediate query resolution.

