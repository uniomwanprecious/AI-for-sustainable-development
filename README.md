# 🌍 AI for Sustainable Development: Dengue Outbreak Prediction

**Project Theme:** Machine Learning Meets the UN Sustainable Development Goals (SDGs).

---

## 1. SDG Problem and ML Approach

* **SDG Addressed:** **Goal 3 (Good Health and Well-being)**, specifically using predictive modeling to manage disease outbreaks.
* **Problem Statement:** Dengue fever is highly sensitive to climate factors. Health officials need a system to proactively predict high-risk weeks to allocate resources efficiently, rather than reactively responding to outbreaks.
* **ML Approach:** **Supervised Learning** Classification.
* **Methodology:**
    * The continuous `total_cases` was converted to a **binary target: `High_Outbreak` (1)**, defined as weeks exceeding the 75th percentile of case counts.
    * Missing climate values were imputed using **Forward/Backward Fill**.
    * The data was split by city, and separate models were trained for San Juan (Puerto Rico) and Iquitos (Peru).

---

## 2. Results and Optimization

Performance was optimized by comparing multiple algorithms (Decision Tree, Random Forest, Logistic Regression) to find the model with the highest **F1-Score**, which is critical for balancing **Precision** and **Recall** when predicting rare events.

| City | Best Model (Optimized for F1-Score) | F1-Score (Outbreak Class 1) | Accuracy |
| :--- | :--- | :--- | :--- |
| **San Juan** | **Logistic Regression** | **0.2381** | 0.4894 |
| **Iquitos** | **All Models Failed** | **0.0000** | 0.9519 |

**Key Findings:**

* **San Juan Solution:** The **Logistic Regression** model was selected for deployment as it achieved the highest F1-Score, demonstrating the best capability to identify the actual `High_Outbreak` class.
* **Iquitos Challenge:** The F1-Score of 0.0000 across all models indicates a total failure to predict the positive class, likely due to high class imbalance. This limitation suggests the need for advanced techniques like oversampling to improve the model's performance in Iquitos.

**Screenshot of Model Comparison:**
![Terminal output showing F1-Score and Accuracy for three models.]

---

## 3. Ethical and Social Reflection

* **Data Bias:** The primary ethical concern is **Reporting Bias** (underreporting). If outbreaks are consistently underreported in vulnerable areas, the model will learn to ignore patterns there, leading to false negatives and the unfair withholding of essential resources.
* **Fairness and Sustainability:**
    * **Promotes Fairness:** The solution promotes fairness by enabling health officials to **proactively and equitably allocate resources** (e.g., mosquito abatement teams, medical supplies) based on predictive risk, rather than simply reacting to confirmed crises.
    * **Supports Sustainability (SDG 13):** By predicting the precise timing of risk, the model reduces the need for constant, large-scale intervention (like insecticide spraying), making public health efforts more resource-efficient and environmentally conscious.

---

### **Submission Deliverables**

* **Code:** `dengue_data_preparation.py` (complete ML pipeline) and `app.py` (Streamlit web application).


* **Live Demo Link:** **[Dengue Outbreak Predictor Live Demo](https://ai-for-sustainable-development-cubkphsnexhsdnvfaubdgp.streamlit.app/)**
* **Demo Purpose:** The app allows a user to input current weekly climate parameters and instantly receive a **HIGH** or **LOW** risk prediction for the next dengue outbreak in San Juan, enabling health officials to act proactively.