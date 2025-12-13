# AgriMitra

**AgriMitra** is a complete, end-to-end **AI-powered smart agriculture platform** designed to support Indian farmers and agricultural authorities with data-driven, localized, and accessible decision-making.

Built for **Smart India Hackathon (SIH) 2025 – Problem Statement ID 25044**, AgriMitra integrates **Machine Learning, Agentic AI, IoT sensing, and Satellite Intelligence** to deliver yield prediction, optimization, disease detection, and governance-level monitoring at scale.

> **Philosophy:** Beyond a tool, *a trusted digital companion that understands every farmer, their land, their crops, and their language, irrespective of their level of education or digital literacy.*

---

## Key Capabilities

* 1. **Crop Yield Prediction** using real-time soil, weather, and historical data
* 2. **Agentic AI Optimization** for fertilizer, irrigation, and soil health
* 3. **Satellite-Based Farm Intelligence** (NDVI, crop stress, vegetation indices)
* 4. **Disease & Pest Detection** using deep learning
* 5. **Voice-Enabled Multilingual Farmer App**
* 6. **Fertilizer Authenticity Verification** via barcode validation
* 7. **Government Monitoring Dashboard** for data-driven policy and advisories

---

## System Architecture

AgriMitra follows a **layered, modular architecture** optimized for scalability and rural deployment constraints.

```
IoT Sensors / Satellite Data / Weather APIs / Farmer Inputs
                ↓
          Data Ingestion Layer
                ↓
     ML & Agentic Intelligence Layer
   (Prediction | Optimization | Vision)
                ↓
        Application Service Layer
      (APIs | Business Logic)
                ↓
 Farmer Mobile App & Govt Web Dashboard
```

---

## Technical Components & Models Involved

### 1. Crop Yield Prediction Engine

**Objective:** Predict expected yield before harvest to guide early interventions.

**Inputs**

* Soil parameters: NPK, moisture, pH
* Weather data: rainfall, temperature, humidity
* Crop metadata: type, sowing date, region
* Historical yield records

**Main Model Used**

* Random Forest Regressor

**Why Random Forest?**

* Handles non-linear agricultural relationships
* Robust to noisy, incomplete field data
* Interpretable feature importance

---

### 2. Agentic AI Optimization Layer

**Objective:** Convert predictions into actionable farming decisions.

**Capabilities**

* Fertilizer dosage optimization
* Soil correction recommendations
* Irrigation scheduling
* Crop-specific advisory

**Design**

* Rule-guided AI agents augmented with ML predictions
* Continuous feedback loop from updated sensor and weather data

---

### 3. Satellite Intelligence Module

**Objective:** Enable farm health monitoring without mandatory IoT hardware.

**Technology**

* Google Earth Engine (GEE)

**Outputs**

* NDVI and vegetation health maps
* Nutrient stress indicators
* Crop growth trend analysis

**Impact**

* Scales to sensor-scarce rural regions
* Reduces hardware dependency

---

### 4. Disease & Pest Detection System

**Objective:** Early identification of crop diseases and pests from images.

**Approach**

* Computer Vision using Deep Learning

**Models**

* Convolutional Neural Networks (CNNs)
* EfficientNet for accuracy–efficiency tradeoff

**Datasets**

* PlantVillage
* PlantDoc
* Kaggle agricultural datasets

**Outputs**

* Disease/pest classification
* Symptoms
* Recommended treatment and pesticides
* Helpline and advisory guidance

---

### 5. Multilingual Voice-First Farmer App

**Objective:** Maximize adoption across literacy and language barriers.

**Features**

* Voice input and output
* Regional language support (English, Hindi, Tamil, Odia)
* Minimal, icon-driven UI

**Result**

* High accessibility for small and marginal farmers

---

### 6. Fertilizer Authenticity Verification

**Problem Addressed:** Counterfeit fertilizers impacting yield and income.

**Solution**

* Barcode scanning
* Validation against government-approved databases

---

### 7. Government Monitoring Dashboard

**Stakeholders**

* Agricultural departments
* Policy makers

**Capabilities**

* Yield performance analytics
* Pest outbreak heatmaps
* Crop distribution insights
* Subsidy and scheme monitoring
* Data-driven advisory generation

---

## Deployment & Scalability

* Cloud-hosted backend architecture
* API-first, service-oriented design
* Offline and low-connectivity support
* Scalable from village to national level

---

## Challenges & Mitigation

| Challenge            | Mitigation Strategy used in AgriMitra |
| -------------------- | ------------------------------------- |
| Data inconsistency   | Hybrid global + local datasets        |
| Low digital literacy | Voice-first UX + simple UI            |
| Connectivity gaps    | Offline-capable workflows             |
| Sensor affordability | Govt/NGO subsidy models               |

---

## Impact

* 1. **Economic:** Improved yield and farmer income
* 2. **Social:** Inclusive access regardless of literacy
* 3. **Environmental:** Optimized fertilizer and water usage
* 4. **Governance:** Data-driven agricultural planning

---

## Tech Stack

* 1. **ML / DL:** Scikit-learn, TensorFlow / PyTorch
* 2. **Satellite:** Google Earth Engine
* 3. **Backend:** Python, REST APIs
* 4. **Frontend:** Mobile app + Web dashboard
* 5. **Data:** Open agricultural datasets + real-time APIs

---

## Demo & Repository

* **YouTube Demo:** [https://youtu.be/vdfzczCk864](https://youtu.be/vdfzczCk864)
* **GitHub Repository:** [https://github.com/SIH-Project/Agri-Mitra](https://github.com/SIH-Project/Agri-Mitra)

---

## Team & Acknowledgements

Developed by a multidisciplinary student team, including Atharva Shukla, Sagar Awasthi, Michelle Ellen Joseph, Uday Trivedi, Eshaan Adyanthaya, and Divyanshu Karmakar as part of **Smart India Hackathon 2025**, achieving a place among the **Top 20 software teams selection** at the college level.

---
