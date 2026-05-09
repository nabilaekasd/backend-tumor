# 🧠 Axon: AI-Powered Brain Tumor Detection (Backend)

This repository contains the backend system for the Brain Tumor Detection application. It is built using **FastAPI**, **PostgreSQL**, and integrated with the **MedNeXt** deep learning model for medical image segmentation.

To ensure a seamless setup, this project is fully **Dockerized**. You do not need to manually install Python, PyTorch, or PostgreSQL on your local machine.

---
## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:
1. **Git** - [Download here](https://git-scm.com/downloads)
2. **Docker Desktop** - [Download here](https://www.docker.com/products/docker-desktop/)

> **Note:** Make sure Docker Desktop is running before proceeding with the installation steps.

---
## ✨ Installation and Setup

Follow these steps to get the system up and running:

### 1. Clone the Repository
Open your terminal or Command Prompt and run:
```bash
git clone [https://github.com/nabilaekasd/backend-tumor.git](https://github.com/nabilaekasd/backend-tumor.git)
cd backend-tumor
```

> **Optional:** If you want to view or inspect the project's source code, you can open it in VS Code by running:
> ```bash
> code .
> ```

### 2. Build and Start the Containers
Run the following command to automatically download the environment, install dependencies, and start the servers:
```bash
docker compose up --build -d
```
*Note: The first run may take 5-15 minutes depending on your internet speed, as it needs to download large AI libraries (PyTorch, MedNeXt, etc).*

### 3. Initialize the Super Admin Account
Once the containers are successfully started, you need to create the initial accounts (Admin, Doctors, and Patients data):
```bash
docker exec -it axon-backend python seed.py
```
If successful, you will see the success messages in your terminal.

---

## 🔗 Accessing the Application

### 🛰️ API Documentation
Once the server is running, you can access the interactive API documentation (Swagger UI) at:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

### 🔑 Default Credentials
Use these credentials for your first login:
- **Username:** `admin`
- **Password:** `admin123`

- **Username:** `radiolog`
- **Password:** `password123`

- **Username:** `dokter`
- **Password:** `password123`

---

## 🛑 Stopping the Services

To stop the backend and database containers, run:
```bash
docker compose down