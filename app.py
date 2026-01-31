from flask import Flask, render_template, request, session, redirect, url_for
import pickle
import numpy as np
import re
from functools import lru_cache

app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_this_in_production'  # Change this!
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

# Cache loaded models to avoid reloading on every request
@lru_cache(maxsize=1)
def load_model():
    with open("job_recommender_model.pkl", "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=1)
def load_vectorizer():
    with open("vectorizer.pkl", "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=1)
def load_job_skill_map():
    with open("job_skill_map.pkl", "rb") as f:
        return pickle.load(f)

# Initialize models at startup
model = load_model()
vectorizer = load_vectorizer()
job_skill_map = load_job_skill_map()

def clean_skills_input(skills_string):
    """Clean and normalize skills input"""
    # Convert to lowercase and remove extra whitespace
    skills_string = skills_string.lower().strip()
    # Replace multiple commas/spaces with single space
    skills_string = re.sub(r'[,\s]+', ' ', skills_string)
    return skills_string

def recommend_jobs_and_skills(student_skills, top_n=5):
    """Recommend jobs and identify missing skills"""
    # Clean input
    student_skills = clean_skills_input(student_skills)
    
    # Create skill set
    student_skill_set = set(filter(None, student_skills.split()))
    
    # Vectorize and predict
    vec = vectorizer.transform([student_skills])
    
    # Check if model supports predict_proba
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(vec)[0]
    else:
        # For models without probability (like SVM without probability=True)
        decision_scores = model.decision_function(vec)[0]
        probabilities = np.exp(decision_scores) / np.sum(np.exp(decision_scores))
    
    classes = model.classes_
    top_indices = np.argsort(probabilities)[::-1][:top_n]
    
    recommendations = []
    
    for idx in top_indices:
        job = classes[idx]
        required_skills = job_skill_map.get(job, set())
        missing_skills = required_skills - student_skill_set
        # Filter out empty strings and sort
        skills_more_to_learn = sorted([skill for skill in missing_skills if skill])
        
        recommendations.append({
            "Job Role": job,
            "Confidence": round(probabilities[idx], 3),
            "Skills More to Learn": skills_more_to_learn
        })
    
    return recommendations

@app.route("/", methods=["GET", "POST"])
def home():
    """Home page - informational only"""
    return render_template("index.html")

@app.route("/search", methods=["GET", "POST"])
def search():
    """Search page with form and results"""
    if 'history' not in session:
        session['history'] = []
    
    results = None
    skills_input = ""
    
    if request.method == "POST":
        skills_input = request.form.get("skills", "").strip()
        
        if skills_input:
            results = recommend_jobs_and_skills(skills_input)
            
            # Limit history to last 10 entries
            session['history'].insert(0, {
                "skills": skills_input,
                "results": results,
                "timestamp": len(session['history'])  # Simple counter as timestamp
            })
            
            # Keep only last 10 entries
            session['history'] = session['history'][:10]
            session.modified = True
    
    history = session.get('history', [])
    return render_template(
        "search.html", 
        results=results, 
        history=history,
        skills_input=skills_input
    )

@app.route("/clear-history", methods=["POST"])
def clear_history():
    """Clear session history"""
    session.pop('history', None)
    return {"success": True}

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)