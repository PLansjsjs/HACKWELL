import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap

class DiseaseModel:
    def __init__(self, dataset_path, target_col, disease_name):
        self.disease_name = disease_name
        self.df = pd.read_csv(dataset_path)

        # Features & target
        self.X = self.df.drop(target_col, axis=1)
        self.y = self.df[target_col]

        # Preprocessing
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42
        )

        # Train model
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        print(f"Model Performance for {self.disease_name}:\n",
              classification_report(y_test, y_pred))

        # SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_names = self.X.columns.tolist()

    def predict(self, user_input: dict):
        user_df = pd.DataFrame([user_input])
        user_scaled = self.scaler.transform(user_df)
        risk = self.model.predict(user_scaled)[0]
        prob = self.model.predict_proba(user_scaled)[0][1]
        return risk, prob

    def explain(self, user_input: dict):
        user_df = pd.DataFrame([user_input])
        user_scaled = self.scaler.transform(user_df)
        shap_values = self.explainer.shap_values(user_scaled)
        return shap_values, user_df
        

class WellnessAssistant:
    def __init__(self):
        # Load all three disease models
        self.diabetes_model = DiseaseModel("diabetes.csv", "Outcome", "Diabetes")
        self.heart_model = DiseaseModel("heart.csv", "target", "Heart Disease")
        self.hypertension_model = DiseaseModel("hypertension.csv", "target", "Hypertension")

    def assess_health(self, diabetes_input: dict, heart_input: dict, hypertension_input: dict):
        results = {}

        # Diabetes
        risk_d, prob_d = self.diabetes_model.predict(diabetes_input)
        results["diabetes"] = {
            "risk": risk_d,
            "probability": prob_d,
            "recs": self._diabetes_recommendations(diabetes_input, risk_d),
            "shap": self.diabetes_model.explain(diabetes_input)
        }

        # Heart
        risk_h, prob_h = self.heart_model.predict(heart_input)
        results["heart"] = {
            "risk": risk_h,
            "probability": prob_h,
            "recs": self._heart_recommendations(heart_input, risk_h),
            "shap": self.heart_model.explain(heart_input)
        }

        # Hypertension
        risk_y, prob_y = self.hypertension_model.predict(hypertension_input)
        results["hypertension"] = {
            "risk": risk_y,
            "probability": prob_y,
            "recs": self._hypertension_recommendations(hypertension_input, risk_y),
            "shap": self.hypertension_model.explain(hypertension_input)
        }

        return results

    def _diabetes_recommendations(self, user_input, risk):
        recs = []
        if risk == 1:
            recs.append("⚠️ Possible diabetes risk. Consult a doctor.")
        if user_input.get("BMI", 25) > 25:
            recs.append("🏃 Exercise regularly to control weight.")
        if user_input.get("Glucose", 100) > 125:
            recs.append("🥗 Reduce sugar intake, eat a balanced diet.")
        if not recs:
            recs.append("✅ Healthy glucose & BMI levels.")
        return recs

    def _heart_recommendations(self, user_input, risk):
        recs = []
        if risk == 1:
            recs.append("⚠️ Possible heart disease risk. Seek medical check-up.")
        if user_input.get("chol", 200) > 240:
            recs.append("🥑 Lower cholesterol through diet.")
        if user_input.get("trestbps", 120) > 140:
            recs.append("💧 Monitor blood pressure, reduce salt intake.")
        if not recs:
            recs.append("✅ Healthy heart profile.")
        return recs

    def _hypertension_recommendations(self, user_input, risk):
        recs = []
        if risk == 1:
            recs.append("⚠️ Hypertension risk detected. Consult a doctor.")
        if user_input.get("systolicBP", 120) > 140:
            recs.append("🧘 Practice stress reduction and reduce salt intake.")
        if user_input.get("BMI", 25) > 25:
            recs.append("🏃 Maintain healthy weight to lower BP.")
        if not recs:
            recs.append("✅ Healthy blood pressure levels.")
        return recs
