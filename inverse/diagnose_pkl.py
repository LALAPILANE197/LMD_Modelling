# diagnose_pkl.py  — run this in your inverse/ directory
import joblib, inspect

obj = joblib.load("multi_output_gpr.pkl")

print(f"Type       : {type(obj)}")
print(f"Module     : {type(obj).__module__}")
print(f"Has estimators_ : {hasattr(obj, 'estimators_')}")
print(f"Has estimator   : {hasattr(obj, 'estimator')}")   # unfitted wrapper
print(f"Has predict     : {hasattr(obj, 'predict')}")
print(f"Attributes : {[a for a in dir(obj) if not a.startswith('__')]}")
