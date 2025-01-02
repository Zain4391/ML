# Regression Project

This project demonstrates **Simple Linear Regression** for predicting house prices based on house size. It uses both **Python** and **JavaScript (Node.js)** implementations.

---

## **Folder Structure**
```
ML/
|-- Regression/
|   |-- node_modules/ (ignored in Git)
|   |-- house_prices.csv
|   |-- SLR.py (Python script)
|   |-- SLR.js (Node.js script)
|   |-- README.md (This file)
|   |-- package.json
|   |-- .gitignore
|-- .gitignore
```

---

## **Python Setup**

### **1. Create a Virtual Environment**
```bash
python3 -m venv venv
```

### **2. Activate Virtual Environment**
- **Linux/MacOS:**
  ```bash
  source venv/bin/activate
  ```
- **Windows:**
  ```bash
  .\venv\Scripts\activate
  ```

### **3. Install Required Python Packages**
```bash
pip install -r requirements.txt
```

### **4. Run Python Script**
```bash
python3 SLR.py
```

---

## **Node.js Setup**

### **1. Initialize Node.js Project (if not already done)**
```bash
npm init -y
```

### **2. Install Dependencies**
```bash
npm install ml-regression csv-parser nodeplotlib
```

### **3. Run Node.js Script**
```bash
node Regression/SLR.js
```

---

## **Working with Git and Node Modules**

### **1. Ignore `node_modules` in Git**
Ensure `.gitignore` includes:
```
node_modules/
```

### **2. Clear Cached `node_modules` (if accidentally committed)**
```bash
git rm -r --cached Regression/node_modules
git commit -m "Remove node_modules folder from tracking"
```

### **3. Push Changes**
```bash
git push origin <branch-name>
```

---

## **Project Features**
- Implements Linear Regression to predict house prices.
- Visualizes data and predictions using **Matplotlib** (Python) and **nodeplotlib** (JavaScript).
- Evaluates model performance using **Mean Squared Error (MSE)**.

---

## **Contact and Support**
For questions, contact the project maintainer at **your-email@example.com**.

