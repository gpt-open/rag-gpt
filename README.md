# smart-qa-service
Smart QA Service, leveraging LLM and RAG technology, learns from user-customized knowledge bases to provide contextually relevant answers for a wide range of queries, ensuring rapid and accurate information retrieval.

---

# Quick Start Guide

## 1. Download Repository Code

Clone the repository:

```
git clone git@github.com:open-kf/smart-qa-service.git
```

## 2. Install Dependencies with pip

It is recommended to install Python-related dependencies in a Python virtual environment to avoid affecting dependencies of other projects.

### Create and Activate a Virtual Environment

If you have not yet created a virtual environment, you can create one with the following command:

```bash
python3 -m venv myenv
```

After creation, activate the virtual environment according to your operating system:

- **Windows:**

  ```
  myenv\Scripts\activate.bat
  ```

- **macOS or Linux:**

  ```
  source myenv/bin/activate
  ```

### Install Dependencies with pip

Once the virtual environment is activated, you can use `pip` to install the required dependencies. After activating the virtual environment, `pip` will only install and manage packages within this virtual environment, not affecting the global Python environment. The command format for installing dependencies is as follows:

```
pip install -r requirements.txt
```

## 3. Configure .env File Variables

Before starting OpenKF, you need to modify the related configurations for the program to initialize correctly. The method is to create a `.env` file and copy the contents of the `env_template` file into the .env file.
You need to modify the following configuration:
- OPEN_API_KEY

Modify `OPENAI_API_KEY` in .env to your own key, which can be obtained by logging into the [OpenAI dashboard](https://platform.openai.com/api-keys):

```
OPENAI_API_KEY=your_open_api_key_here
```

## 4. Create sqlite Database
OpenKF uses sqlite as its data storage. Before starting OpenKF, you need to execute the following file to initialize the database and data tables.

```
python3 create_sqlite_db.py
```
## 5. Install Redis
The OpenKF service relies on Redis as a chat content cache, requiring Redis installation. If Redis is already installed, start Redis and listen on port 6379. If not installed, refer to the following method for installation.

- **Windows:**

Redis can be installed using Docker. Install Docker, and if not installed, refer to the [official website](https://www.docker.com/products/docker-desktop/) for installation. After installing, pull the Redis image, and start Redis, listening on port 6379:

```
docker run --name some-redis -d -p 6379:6379 redis
```

- **macOS or Linux:**

Refer to the Redis [official website](https://redis.io/docs/install/install-redis/install-redis-on-linux/) for installation.

## 6. Start the Service

If you have completed the steps above, you can try to start the OpenKF  service by executing the following command.

- **Windows:**

```
flask --app .\open_kf_app.py run
```

- **macOS or Linux:**

```
sh start.sh
```

Congratulations, enjoy it!
