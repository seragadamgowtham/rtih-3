import os
import sqlite3
import json
import requests
import PyPDF2
from PIL import Image
import pytesseract
import io
from flask import Flask, render_template, request, jsonify

# --- FLASK SETUP ---
app = Flask(__name__)

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"
DB_PATH = 'medmap_ai.db'
TRAINING_DB_PATH = 'medmap_training.db'
OLLAMA_OPTIONS = {
    "num_predict": 2048,
    "num_ctx": 4096,
    "num_thread": 8,
    "temperature": 0.1
}

# --- DATABASE SETUP ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Patients Table
    cursor.execute('''CREATE TABLE IF NOT EXISTS patients (
                        pid INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL)''')
                        
    # Doctors Table
    cursor.execute('''CREATE TABLE IF NOT EXISTS doctors (
                        did INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        specialization TEXT)''')
                        
    # Consultations Table
    cursor.execute('''CREATE TABLE IF NOT EXISTS consultations (
                        cid INTEGER PRIMARY KEY AUTOINCREMENT,
                        pid INTEGER,
                        symptoms TEXT,
                        ai_situation TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(pid) REFERENCES patients(pid))''')

    # Medicines Table
    cursor.execute('''CREATE TABLE IF NOT EXISTS medicines (
                        mid TEXT PRIMARY KEY,
                        disease TEXT,
                        brand_name TEXT,
                        generic_name TEXT,
                        form TEXT,
                        strength TEXT,
                        similarity_percentage REAL,
                        confidence TEXT,
                        description TEXT,
                        count INTEGER DEFAULT 1)''')

    # Prescriptions Table
    cursor.execute('''CREATE TABLE IF NOT EXISTS prescriptions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cid INTEGER,
                        did INTEGER,
                        mid TEXT,
                        form TEXT,
                        frequency TEXT,
                        duration_days INTEGER,
                        generic_name TEXT,
                        strength TEXT,
                        similarity REAL,
                        confidence TEXT,
                        ai_report TEXT,
                        doctor_notes TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(cid) REFERENCES consultations(cid),
                        FOREIGN KEY(did) REFERENCES doctors(did),
                        FOREIGN KEY(mid) REFERENCES medicines(mid))''')
    
    # Default Dummy Data
    cursor.execute("SELECT COUNT(*) FROM doctors")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO doctors (name, specialization) VALUES ('Dr. Smith', 'General Physician')")
        cursor.execute("INSERT INTO doctors (name, specialization) VALUES ('Dr. Adams', 'Cardiologist')")
        cursor.execute("INSERT INTO doctors (name, specialization) VALUES ('Dr. Lee', 'Neurologist')")

    cursor.execute("SELECT COUNT(*) FROM patients")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO patients (name) VALUES ('John Doe')")
        cursor.execute("INSERT INTO patients (name) VALUES ('Jane Roe')")

    conn.commit()
    conn.close()

init_db()

def get_training_db_connection():
    conn = sqlite3.connect(TRAINING_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_training_db():
    conn = get_training_db_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS knowledge (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT,
                        content TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_training_db()

def get_training_context():
    conn = get_training_db_connection()
    rows = conn.execute('SELECT content FROM knowledge').fetchall()
    conn.close()
    if not rows: return ""
    context = "\n".join([r['content'] for r in rows])
    return f"Use this medical knowledge directly from trained documents to answer/instruct accurately:\n{context}\n\n"


# --- AI LOGIC ---
def analyze_symptoms_with_ai(symptoms, known_diseases=None):
    """Uses Ollama to identify the situation from symptoms, preferring known DB diseases."""
    context = get_training_context()
    
    known_disease_prompt = ""
    if known_diseases and len(known_diseases) > 0:
        known_str = ", ".join([f"'{d}'" for d in known_diseases if d])
        known_disease_prompt = f"""
    CRITICAL: Try to match the patient's condition to one of these known diseases in our database: [{known_str}]. 
    If it matches, output the exact string from this list as the 'situation'. 
    If it is completely unrelated, then define the new situation concisely.
    """

    prompt = f"""
    {context}
    Act as a medical triage system. A patient has the following symptoms: "{symptoms}".
    1. Define the type of situation/condition concisely (e.g., 'Mild Viral Infection', 'Potential Cardiac Issue', etc.).
    {known_disease_prompt}
    
    Output strictly in this JSON format:
    {{
        "situation": "Defined condition here"
    }}
    """
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": OLLAMA_OPTIONS
        }, timeout=None)
        
        if response.status_code == 200:
            data = response.json()
            if "response" in data:
                res_json = json.loads(data["response"])
                return res_json.get("situation", "Unknown Condition")
    except Exception as e:
        print(f"AI Error: {e}")
        return "Unable to determine situation at this time."
    return "Unknown Condition"

def get_or_create_patient(name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT pid FROM patients WHERE name = ?', (name,))
    row = cursor.fetchone()
    if row:
        pid = row['pid']
    else:
        cursor.execute('INSERT INTO patients (name) VALUES (?)', (name,))
        pid = cursor.lastrowid
    conn.commit()
    conn.close()
    return pid

def get_or_create_doctor(name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT did FROM doctors WHERE name = ?', (name,))
    row = cursor.fetchone()
    if row:
        did = row['did']
    else:
        cursor.execute('INSERT INTO doctors (name, specialization) VALUES (?, ?)', (name, 'General Provider'))
        did = cursor.lastrowid
    conn.commit()
    conn.close()
    return did

def get_or_create_medicine(generic_name, form, strength, disease="", brand_name="", similarity=0.0, confidence="Unknown", description=""):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Generate Base ID: First 2 letters + Last 2 letters + Dosage
    gn_clean = generic_name.replace(" ", "").upper()
    str_clean = strength.replace(" ", "").upper()
    if len(gn_clean) >= 4:
        base_id = gn_clean[:2] + gn_clean[-2:] + str_clean
    else:
        base_id = gn_clean + str_clean
        
    cursor.execute('SELECT mid, count FROM medicines WHERE generic_name = ? AND form = ? AND strength = ?', (generic_name, form, strength))
    row = cursor.fetchone()
    
    if row:
        mid = row['mid']
        new_count = row['count'] + 1
        cursor.execute('UPDATE medicines SET count = ? WHERE mid = ?', (new_count, mid))
    else:
        # Find how many starting with this base ID exist to append unique number
        cursor.execute("SELECT COUNT(*) FROM medicines WHERE mid LIKE ?", (base_id + '-%',))
        count = cursor.fetchone()[0] + 1
        mid = f"{base_id}-{count}"
        cursor.execute('''INSERT INTO medicines 
                          (mid, disease, brand_name, generic_name, form, strength, similarity_percentage, confidence, description) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                          (mid, disease, brand_name, generic_name, form, strength, similarity, confidence, description))
        
    conn.commit()
    conn.close()
    return mid

def generate_medicine_instructions(generic_name, form, strength, frequency, duration_days, condition, context=""):
    prompt = f"""
    {context}
    A doctor prescribed the following medicine to a patient diagnosed with: "{condition}":
    
    Medicine: {generic_name} ({strength})
    Form: {form}
    Frequency: {frequency}
    Duration: {duration_days} days
    
    Provide a detailed patient report strictly following these limits based on the patient's condition:
    - total_quantity: Mathematically calculate (Frequency per day * Duration days). If it is liquid or continuous, just estimate or write "1 Bottle" or "Continuous".
    - report: A 1-line summary of what the prescription is for. Include the Total Quantity in this line.
    - advantages: Exactly 2 lines explaining the benefits of this medicine specifically for their condition.
    - disadvantages: Exactly 3 lines explaining the side effects or drawbacks if the medicine is NOT taken properly.
    - instructions: Exactly 2 lines on how to safely take the medicine.
    
    Output strictly in this JSON format without any markdown wrapper:
    {{
        "total_quantity": "XX Tablets/Units",
        "report": "Line 1",
        "advantages": "Line 1\\nLine 2",
        "disadvantages": "Line 1\\nLine 2\\nLine 3",
        "instructions": "Line 1\\nLine 2"
    }}
    """
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": OLLAMA_OPTIONS
        }, timeout=None)
        
        if response.status_code == 200:
            data = response.json()
            if "response" in data:
                return data["response"]
    except Exception as e:
        print(f"AI Error: {e}")
        return json.dumps({
            "report": "Prescription received.", 
            "advantages": "Follow doctor's advice.", 
            "disadvantages": "Consult doctor if issues arise.", 
            "instructions": "Take as prescribed."
        })
    return json.dumps({
            "report": "Prescription received.", 
            "advantages": "Follow doctor's advice.", 
            "disadvantages": "Consult doctor if issues arise.", 
            "instructions": "Take as prescribed."
        })

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/patients', methods=['GET'])
def get_patients():
    conn = get_db_connection()
    patients = conn.execute('SELECT * FROM patients').fetchall()
    conn.close()
    return jsonify([dict(p) for p in patients])

@app.route('/api/doctors', methods=['GET'])
def get_doctors():
    conn = get_db_connection()
    doctors = conn.execute('SELECT * FROM doctors').fetchall()
    conn.close()
    return jsonify([dict(d) for d in doctors])

@app.route('/api/medicines', methods=['GET'])
def get_medicines():
    conn = get_db_connection()
    medicines = conn.execute('SELECT * FROM medicines').fetchall()
    conn.close()
    return jsonify([dict(m) for m in medicines])
    
@app.route('/api/medicines/add', methods=['POST'])
def add_medicine():
    data = request.json
    
    # Check if bulk array exists
    meds_to_add = []
    if 'medicines_list' in data and isinstance(data['medicines_list'], list):
        meds_to_add = data['medicines_list']
    elif 'matched_medicine' in data:
        meds_to_add = [data['matched_medicine']]
    else:
        return jsonify({'error': 'Missing matched_medicine or medicines_list payload'}), 400
        
    required = ['id', 'brand_name', 'generic_name', 'strength', 'form']
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    added_count = 0
    for med in meds_to_add:
        missing = [req for req in required if req not in med]
        if missing:
            continue # Skip invalid items in batch
            
        cursor.execute('SELECT count FROM medicines WHERE mid = ?', (med['id'],))
        row = cursor.fetchone()
        if row:
            cursor.execute('UPDATE medicines SET count = count + 1 WHERE mid = ?', (med['id'],))
        else:
            cursor.execute('''INSERT INTO medicines 
                              (mid, disease, brand_name, generic_name, form, strength, similarity_percentage, confidence, description) 
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                              (med['id'], med.get('disease_type', ''), med['brand_name'], med['generic_name'], med['form'], med['strength'], med.get('similarity_percentage', 0), med.get('confidence', 'Unknown'), json.dumps(med.get('description', {}))))
        added_count += 1
        
    conn.commit()
    conn.close()
    
    if added_count == 0:
        return jsonify({"error": "No valid medicines could be added."}), 400
        
    return jsonify({"message": f"Successfully added {added_count} medicines."})

@app.route('/api/medicines/extract', methods=['POST'])
def extract_medicine():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    text = ""
    try:
        if file.filename.lower().endswith('.pdf'):
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(file.stream)
            text = pytesseract.image_to_string(image)
        else:
            return jsonify({'error': 'Unsupported file type.'}), 400
            
        if not text.strip():
            return jsonify({'error': 'Could not extract text.'}), 400
            
        prompt = f"""
        Extract ALL medicine details from the following medical text and map them to a JSON array. 
        If there are multiple distinct medicines in the text, extract every single one of them.
        Return strictly JSON with NO markdown wrappers.
        
        Format Requirement:
        {{
          "medicines_list": [
            {{
              "disease_type": "What disease does this treat?",
              "id": "Generate a unique ID (e.g., MED123)",
              "brand_name": "Name",
              "generic_name": "Name",
              "strength": "Dose",
              "form": "Tablet/Syrup/etc",
              "similarity_percentage": 95.0,
              "confidence": "High/Medium/Low",
              "description": {{
                  "how_to_use": "Short instruction",
                  "advantages": "2 lines",
                  "disadvantages": "3 lines"
              }}
            }},
            {{
              ... (next medicine if available)
            }}
          ]
        }}
        
        Text:
        {text}
        """
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": 4096,
                "num_ctx": 8192,
                "num_thread": 8,
                "temperature": 0.1
            }
        }, timeout=None)
        
        if response.status_code == 200:
            import re
            data = response.json()
            if "response" in data:
                raw_response = data["response"].strip()
                parsed = None
                
                # Try standard parsing first
                try:
                    parsed = json.loads(raw_response)
                except Exception:
                    # Fallback Regex Recovery: Look for an object { }
                    obj_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                    if obj_match:
                        try:
                            parsed = json.loads(obj_match.group(0))
                        except Exception:
                            pass
                    # If still fails, look for a root array [ ]
                    if not parsed:
                        arr_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
                        if arr_match:
                            try:
                                parsed = {"medicines_list": json.loads(arr_match.group(0))}
                            except Exception:
                                pass
                                
                if parsed:
                    # Enforce that the output has the 'medicines_list' key
                    if "medicines_list" not in parsed:
                        if isinstance(parsed, list):
                            parsed = {"medicines_list": parsed}
                        else:
                            parsed = {"medicines_list": [parsed]}
                    return jsonify(parsed)
                else:
                    return jsonify({'error': 'AI generated unrecoverable corrupt JSON.'}), 500
                        
        return jsonify({'error': 'AI failed to parse document details or exceeded max RAM.'}), 500
    except Exception as e:
        return jsonify({'error': f'Parsing failed: {str(e)}'}), 500

@app.route('/api/patient/ask', methods=['POST'])
def patient_ask():
    data = request.json
    patient_name = data.get('patient_name')
    symptoms = data.get('symptoms')
    
    if not patient_name or not symptoms:
        return jsonify({'error': 'Missing patient name or symptoms'}), 400
        
    conn = get_db_connection()
    pid = get_or_create_patient(patient_name)
    
    # Fetch known diseases from DB
    diseases_rows = conn.execute('SELECT DISTINCT disease FROM medicines WHERE disease IS NOT NULL AND disease != ""').fetchall()
    known_diseases = [row['disease'] for row in diseases_rows]
    
    situation = analyze_symptoms_with_ai(symptoms, known_diseases)
    
    cursor = conn.cursor()
    cursor.execute('INSERT INTO consultations (pid, symptoms, ai_situation) VALUES (?, ?, ?)', (pid, symptoms, situation))
    cid = cursor.lastrowid
    conn.commit()
    
    doctors = conn.execute('SELECT * FROM doctors LIMIT 3').fetchall()
    conn.close()
    
    # Return situation and suggested doctors
    return jsonify({
        'cid': cid,
        'situation': situation,
        'suggested_doctors': [dict(d) for d in doctors]
    })


@app.route('/api/consultations', methods=['GET'])
def get_consultations():
    # Includes patient name and pending prescription check
    conn = get_db_connection()
    query = '''
    SELECT c.cid, c.pid, p.name as patient_name, c.symptoms, c.ai_situation, c.timestamp 
    FROM consultations c
    JOIN patients p ON c.pid = p.pid
    ORDER BY c.timestamp DESC
    '''
    consults = conn.execute(query).fetchall()
    
    # Let's attach prescriptions if any
    res = []
    for row in consults:
        c_dict = dict(row)
        presc_count = conn.execute('SELECT COUNT(*) FROM prescriptions WHERE cid = ?', (c_dict['cid'],)).fetchone()[0]
        c_dict['has_prescription'] = presc_count > 0
        res.append(c_dict)
        
    conn.close()
    return jsonify(res)

@app.route('/api/prescriptions/patient/<path:patient_name>', methods=['GET'])
def get_patient_prescriptions(patient_name):
    conn = get_db_connection()
    query = '''
    SELECT pr.*, d.name as doctor_name, c.symptoms, c.ai_situation
    FROM prescriptions pr
    JOIN doctors d ON pr.did = d.did
    JOIN consultations c ON pr.cid = c.cid
    JOIN patients p ON c.pid = p.pid
    WHERE p.name = ?
    ORDER BY pr.timestamp DESC
    '''
    prescriptions = conn.execute(query, (patient_name,)).fetchall()
    conn.close()
    return jsonify([dict(p) for p in prescriptions])

@app.route('/api/doctor/auto_fill', methods=['POST'])
def auto_fill_prescription():
    data = request.json
    cid = data.get('cid')
    if not cid:
        return jsonify({'error': 'Missing consultation ID'}), 400
        
    conn = get_db_connection()
    row = conn.execute('SELECT symptoms, ai_situation FROM consultations WHERE cid = ?', (cid,)).fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Consultation not found'}), 404
        
    disease = row['ai_situation']
    symptoms = row['symptoms']
    
    # Simple matching logic: find a medicine where the target disease is similar to the patient's disease
    # A more advanced app would use embeddings or LIKE '%word%', for now we'll do our best string match
    # Try looking for keywords
    disease_words = [w for w in disease.lower().replace(',', '').replace('.', '').split() if len(w) > 3]
    
    query = "SELECT * FROM medicines WHERE "
    conditions = []
    params = []
    
    if disease_words:
        for word in disease_words:
            conditions.append("LOWER(disease) LIKE ?")
            params.append(f"%{word}%")
        query += " OR ".join(conditions) + " ORDER BY count DESC LIMIT 1"
    else:
        query = "SELECT * FROM medicines ORDER BY count DESC LIMIT 1" # Fallback
        
    med_row = conn.execute(query, params).fetchone()
    conn.close()
    
    if med_row:
        # AI Verification Step
        prompt = f"""
        A doctor is considering prescribing {med_row['generic_name']} ({med_row['strength']}) for a patient diagnosed with {disease} who has the following symptoms: "{symptoms}".
        Is this medicine safe, relevant, and appropriate given the symptoms?
        
        Provide your assessment strictly in the following JSON format:
        {{
            "safe": true/false,
            "reason": "1-2 sentences explaining why it is safe or why it should not be prescribed.",
            "suggested_frequency": "e.g., 2 times/day or 1 time/day based on severity",
            "suggested_duration_days": "Number of days (e.g. 5, 10). *CRITICAL: If the disease is incurable (e.g. cancer) or requires surgery, output 'Continuous until surgery' or 'Continuous' instead of a number.*"
        }}
        """
        
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": OLLAMA_OPTIONS
            }, timeout=None)
            
            if response.status_code == 200:
                data = response.json()
                if "response" in data:
                    ai_verdict = json.loads(data["response"])
                    if not ai_verdict.get("safe", False):
                        return jsonify({'error': f"AI Verification Failed: {ai_verdict.get('reason', 'Medicine flagged as potentially unsafe.')}", 'ai_rejected': True}), 200
                        
                    # It is safe, return everything but inject our verification into dict
                    med_dict = dict(med_row)
                    med_dict['verification_note'] = f"AI Verified: {ai_verdict.get('reason', 'Safe for these symptoms.')}"
                    return jsonify(med_dict)
                    
        except Exception as e:
            # If verification fails unexpectedly, we'll allow it to pass but notify it failed to verify
            med_dict = dict(med_row)
            med_dict['verification_note'] = "AI Verification Skipped: Could not reach LLM for safety check."
            return jsonify(med_dict)
            
        return jsonify(dict(med_row))
    else:
        # Fallback to AI Generation when no DB match exists
        fallback_prompt = f"""
        A doctor needs a temporary prescription for a patient with {disease} who has symptoms: "{symptoms}".
        THERE ARE NO MEDICINES IN THE SYSTEM DATABASE. 
        You must suggest a safe, standard temporary medication to manage this condition.
        
        Provide your emergency substitute strictly in this JSON format matching the database schema:
        {{
            "generic_name": "Suggested Generic Name",
            "strength": "Suggested Dose",
            "form": "Tablet/Cream/etc",
            "suggested_frequency": "e.g., 2 times/day",
            "suggested_duration_days": "Number of days. If incurable or needs surgery, write 'Continuous'.",
            "similarity_percentage": [Estimate 1-100 how effective this is for {disease}],
            "confidence": "High/Medium/Low",
            "description": {{
                "how_to_use": "Short instructions. YOU MUST INCLUDE THIS EXACT PHRASE: 'If the problem persists, visit again or contact medical emergencies.'",
                "advantages": "2 lines",
                "disadvantages": "2 lines"
            }}
        }}
        """
        
        try:
            fb_response = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": fallback_prompt,
                "stream": False,
                "format": "json",
                "options": OLLAMA_OPTIONS
            }, timeout=None)
            
            if fb_response.status_code == 200:
                fb_data = fb_response.json()
                if "response" in fb_data:
                    import re
                    json_match = re.search(r'\{.*\}', fb_data["response"], re.DOTALL)
                    if json_match:
                        ai_med = json.loads(json_match.group(0))
                        
                        # Format it exactly like a DB hit so the frontend doesn't crash
                        return jsonify({
                            "generic_name": ai_med.get("generic_name", "Unknown AI Match"),
                            "strength": ai_med.get("strength", ""),
                            "form": ai_med.get("form", "Tablet"),
                            "suggested_frequency": ai_med.get("suggested_frequency", ""),
                            "suggested_duration_days": ai_med.get("suggested_duration_days", ""),
                            "similarity_percentage": ai_med.get("similarity_percentage", 50.0),
                            "confidence": ai_med.get("confidence", "AI Estimate"),
                            "description": json.dumps(ai_med.get("description", {})),
                            "verification_note": "⚠️ AI SUGGESTED SUBSTITUTE: No exact match in Medical DB. Please verify this temporary prescription."
                        })
        except Exception as e:
            pass # Fall through to the original 404 error below if AI fails too
            
        return jsonify({'error': f'No medicine found in database natively matching the disease: {disease}, and AI fallback failed.'}), 404

@app.route('/api/doctor/prescribe', methods=['POST'])
def prescribe():
    data = request.json
    required_fields = ['cid', 'doctor_name', 'form', 'frequency', 'duration_days', 'generic_name', 'strength', 'similarity', 'confidence']
    for f in required_fields:
        if f not in data:
            return jsonify({'error': f'Missing field {f}'}), 400
            
    conn = get_db_connection()
    # Fetch patient's condition for tailored AI instructions and ID generation
    condition = "Unknown condition"
    row = conn.execute('SELECT ai_situation FROM consultations WHERE cid = ?', (data['cid'],)).fetchone()
    if row:
        condition = row['ai_situation']
            
    did = get_or_create_doctor(data['doctor_name'])
    mid = get_or_create_medicine(data['generic_name'], data['form'], data['strength'], condition, "Unknown Brand", data['similarity'], data['confidence'])
    
        
    context = get_training_context()
    
    ai_report_json_str = generate_medicine_instructions(
        data['generic_name'], data['form'], data['strength'], data['frequency'], data['duration_days'], condition, context
    )
            
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO prescriptions (cid, did, mid, form, frequency, duration_days, generic_name, strength, similarity, confidence, ai_report, doctor_notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['cid'], did, mid, data['form'], data['frequency'],
        data['duration_days'], data['generic_name'], data['strength'], data['similarity'], data['confidence'], ai_report_json_str, data.get('doctor_notes', '')
    ))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Prescription saved successfully'})

# Removed upload_pdf route

@app.route('/api/doctor/parse_prescription_file', methods=['POST'])
def parse_prescription_file():
        # Check for raw text submission or file
    if 'raw_text' in request.form:
        text = request.form['raw_text']
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        try:
            if file.filename.lower().endswith('.pdf'):
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(file.stream)
                text = pytesseract.image_to_string(image)
            else:
                return jsonify({'error': 'Unsupported file type.'}), 400
        except Exception as e:
            return jsonify({'error': f'File extraction failed: {str(e)}'}), 500
    else:
        return jsonify({'error': 'No document or text provided'}), 400
                
    if not text.strip():
        return jsonify({'error': 'Could not extract text from the input.'}), 400
        
    conn = get_db_connection()
    medicines_rows = conn.execute('SELECT mid, brand_name, generic_name, strength, form FROM medicines').fetchall()
    conn.close()
    
    known_meds_str = json.dumps([dict(r) for r in medicines_rows])
        
    # Send text to AI to parse
    prompt = f"""
Extract the prescription details from the following raw medical text.
You MUST map the extraction to the following STRICT JSON format containing an 'extracted_medicines' array.

CRITICAL INSTRUCTIONS:
1. Extract 'brand_name' and 'brand_variant' separately. For example, in "Amoxiclav 625", 'Amoxiclav' is the brand name, and '625' is the brand variant.
2. DO NOT assume the numeric brand variant is the true strength. '625' does NOT mean 625mg strength.
3. Calculate the mathematical integer for 'frequency_per_day' based on the medical abbreviations (e.g. QID = 4, BID = 2, TDS = 3, OD = 1).
4. Identify the medical cause (disease) that this specific tablet/formulation is commonly used to treat.
5. You have access to the local database track records here: {known_meds_str}.
6. Match the extracted variant and cause to the database records to find the true Generic Name, true Strength, and matched Disease (e.g., if the DB says Amoxiclav 625 is "Amoxicillin + Clavulanic Acid" at "500mg+125mg" for "Bacterial Infection", use those DB variables for 'generic_name', 'strength', and 'disease_cause'). 

Format Requirement:
{{
  "extracted_medicines": [
    {{
      "raw_input": "The exact raw text chunk for this medicine",
      "structured_data": {{
        "brand_name": "Brand",
        "brand_variant": "Label Variant (e.g. 625)",
        "form": "Tablet/Drop/etc",
        "frequency_per_day": 2,
        "duration_days": 5,
        "disease_cause": "The specific disease/cause this treats"
      }},
      "matched_medicine": {{
        "id": "Mapped ID from local DB (or empty if unknown)",
        "brand_name": "Mapped Brand",
        "generic_name": "Mapped TRUE Generic Name",
        "strength": "Mapped TRUE Strength (DO NOT GUESS. Match DB or leave empty)",
        "form": "Mapped Form",
        "disease_type": "Mapped Disease from DB",
        "confidence": "High/Medium/Low"
      }}
    }}
  ]
}}

Text:
{text}
"""
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": OLLAMA_OPTIONS
        }, timeout=None)
        
        if response.status_code == 200:
            data = response.json()
            if "response" in data:
                return jsonify(json.loads(data["response"]))
        return jsonify({'error': 'AI failed to parse details.'}), 500

    except Exception as e:
        return jsonify({'error': f'Failed to parse file: {str(e)}'}), 500

@app.route('/api/doctor/analyze_cause', methods=['POST'])
def analyze_cause():
    data = request.json
    symptoms = data.get('symptoms')
    if not symptoms:
        return jsonify({'error': 'Missing symptoms'}), 400

    context = get_training_context()
    
    conn = get_db_connection()
    diseases_rows = conn.execute('SELECT DISTINCT disease FROM medicines WHERE disease IS NOT NULL AND disease != ""').fetchall()
    known_diseases = [row['disease'] for row in diseases_rows]
    conn.close()
    
    known_str = ", ".join([f"'{d}'" for d in known_diseases]) if known_diseases else "None explicitly tracked"

    prompt = f"""
    {context}
    A patient has the following symptoms: "{symptoms}".
    
    You have access to a local medical database tracking these diseases: [{known_str}].
    
    1. Identify the most likely medical cause by cross-referencing global medical knowledge (LLM training data/references) and the known diseases in the local database.
    2. Suggest alternative treatments or home remedies.
    3. EXPLICITLY check for treatment routes: If a specific tablet or medication helps this condition, name it clearly so the doctor knows what to prescribe or add to the database. Match the disease name EXPLICITLY to one of the known database diseases if applicable.
    
    Output strictly in this JSON format:
    {{
        "cause": "Cause explanation (Reference DB if matched)",
        "alternatives": "Alternative treatments. Include specific generic tablet names if medication helps."
    }}
    """
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": OLLAMA_OPTIONS
        }, timeout=None)

        if response.status_code == 200:
            data = response.json()
            if "response" in data:
                return jsonify(json.loads(data["response"]))
    except Exception as e:
        print(f"AI Error: {e}")
        return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Failed to generate analysis'}), 500

@app.route('/api/db/delete', methods=['POST'])
def delete_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # We only delete the user-generated tracking data
        cursor.execute('DELETE FROM patients')
        cursor.execute('DELETE FROM doctors')
        cursor.execute('DELETE FROM consultations')
        cursor.execute('DELETE FROM prescriptions')
        
        # Re-inject the default dummy users so the dropdowns aren't entirely empty
        cursor.execute("INSERT INTO doctors (name, specialization) VALUES ('Dr. Smith', 'General Physician')")
        cursor.execute("INSERT INTO doctors (name, specialization) VALUES ('Dr. Adams', 'Cardiologist')")
        cursor.execute("INSERT INTO doctors (name, specialization) VALUES ('Dr. Lee', 'Neurologist')")
        
        cursor.execute("INSERT INTO patients (name) VALUES ('John Doe')")
        cursor.execute("INSERT INTO patients (name) VALUES ('Jane Roe')")
        
        conn.commit()
        conn.close()
        return jsonify({'message': 'Volatile Database forms cleared successfully. Medical Database intact!'})
    except Exception as e:
        return jsonify({'message': f'Error resetting database: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
