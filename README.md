
### 🚀 How to Run

```bash
python main.py
```



## Prerequisites
Ensure you have the following installed:
- Python 3.x
- pip
- Jupyter Notebook

---

## Setup Steps

### 1. Create a Virtual Environment
Create a Python virtual environment named `venv`:
```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```
- On Windows:
  ```bash
  venv\Scripts\activate
  ```

Once activated, your terminal prompt should reflect the virtual environment.

### 3. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Ensure Jupyter Notebook Has Access to the Virtual Environment
Add your virtual environment as a kernel in Jupyter Notebook:
```bash
python -m ipykernel install --user --name=venv --display-name "Python (mapper-venv)"
```

### 5. Select the Virtual Environment Kernel
1. Open the minimal_agent_framework notebook or create a new one.
2. Select the kernel dropdown (usually at the top-right).
3. Choose `Python (mapper-venv)` from the list.

---
