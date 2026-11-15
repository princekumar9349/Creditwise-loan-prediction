"""Train the loan approval model and save it. Run: python train.py"""
import json, joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("loan_approval_data.csv")
df = df.dropna(subset=["Loan_Approved"])            # never impute the target
y = (df["Loan_Approved"] == "Yes").astype(int)
X = df.drop(columns=["Loan_Approved", "Applicant_ID", "Gender", "Marital_Status"])  # ID + sensitive cols

num = X.select_dtypes("number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()

def make(model, scale=False):
    n = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    pre = ColumnTransformer([("num", Pipeline(n), num),
                             ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                               ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    return Pipeline([("pre", pre), ("model", model)])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
models = {"Logistic Regression": make(LogisticRegression(max_iter=1000), True),
          "Random Forest": make(RandomForestClassifier(n_estimators=300, random_state=42))}

report = {}
for name, m in models.items():
    cv = cross_val_score(m, Xtr, ytr, cv=5, scoring="f1").mean()
    m.fit(Xtr, ytr); p = m.predict(Xte)
    report[name] = dict(precision=precision_score(yte, p), recall=recall_score(yte, p), f1=f1_score(yte, p),
                        accuracy=accuracy_score(yte, p), cv_f1=cv, cm=confusion_matrix(yte, p).tolist())
    print(name, {k: round(v, 3) for k, v in report[name].items() if k != "cm"})

best = max(report, key=lambda k: report[k]["cv_f1"])
model = models[best]
names = model.named_steps["pre"].get_feature_names_out()
imp = pd.Series(model.named_steps["model"].feature_importances_ if best == "Random Forest"
                else abs(model.named_steps["model"].coef_[0]), index=names).sort_values(ascending=False)
meta = dict(best=best, report=report, importance=imp.head(10).to_dict(), num=num, cat=cat,
            ranges={c: [float(X[c].min()), float(X[c].max()), float(X[c].median())] for c in num},
            options={c: sorted(X[c].dropna().unique().tolist()) for c in cat},
            n_rows=len(df), approval_rate=float(y.mean()))
joblib.dump(model, "model.joblib"); json.dump(meta, open("meta.json", "w"), indent=1)
print("Best:", best)
