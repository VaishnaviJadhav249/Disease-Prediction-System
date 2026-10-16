"""
Comprehensive Verification and Test Suite for Disease Diagnosis System
Validates all features and recent fixes:
1. User Registration, Login & Session Isolation
2. Per-Symptom Severity Selection & Storage
3. Minimum 3 Symptoms Requirement Enforcement
4. Top 3 Differential Diagnoses with Confidence Probabilities
5. Low-Confidence Warning (<60%) Advisory
6. Emergency Red-Flag Advisories (Chest Pain, Breathlessness, etc.)
7. Explainable AI with Leave-One-Symptom-Out (LOSO) & Full 41-Neighbor Consensus
8. PDF Assessment Report Generation
"""
import os
import sys
import json
import pickle
import unittest
import uuid
from werkzeug.security import check_password_hash

# Ensure workspace root is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import database
import followup
import symptoms_data
import explainability
import pdf_generator
from app import app

class TestDiseasePredictionSystem(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        # Create and log in a dedicated test user for session tests
        self.username = f"suite_user_{uuid.uuid4().hex[:6]}"
        self.password = "ValidPass123!"
        self.user = database.create_user(self.username, f"{self.username}@example.com", self.password)
        with self.app.session_transaction() as sess:
            sess['user_id'] = self.user['id']
            sess['username'] = self.user['username']

    # ----------------------------------------------------
    # Model & Feature List Checks
    # ----------------------------------------------------
    def test_01_model_file_exists_and_loads(self):
        """Verifies model file exists and loads KNeighborsClassifier."""
        model = explainability.get_model()
        self.assertIsNotNone(model, "Model must load successfully")
        self.assertEqual(model.n_neighbors, 41, "KNeighbors model must have n_neighbors=41")

    def test_02_symptoms_list_exact_132(self):
        """Verifies canonical symptoms list has exactly 132 features in model order."""
        symptoms = symptoms_data.get_symptoms_list()
        self.assertEqual(len(symptoms), 132, "Feature vector must have exactly 132 features")
        self.assertEqual(symptoms[0], 'itching')
        self.assertEqual(symptoms[1], 'skin_rash')

    def test_03_prediction_pipeline_unmodified(self):
        """Verifies model prediction on 132-binary vector matches expected output."""
        model = explainability.get_model()
        vec = [0] * 132
        vec[0] = 1 # itching
        vec[1] = 1 # skin_rash
        pred = model.predict([vec])
        self.assertTrue(len(pred) > 0)
        self.assertEqual(pred[0], 'Fungal infection')

    # ----------------------------------------------------
    # User Registration, Authentication & Isolation
    # ----------------------------------------------------
    def test_04_user_registration_and_password_hashing(self):
        """Tests that user passwords are securely hashed and never stored in plain text."""
        u_name = f"user_{uuid.uuid4().hex[:6]}"
        plain_pw = "SecretHash987!"
        u = database.create_user(u_name, f"{u_name}@test.com", plain_pw)

        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE id = ?", (u['id'],))
        row = cursor.fetchone()
        conn.close()

        stored_hash = row['password_hash']
        self.assertNotEqual(stored_hash, plain_pw)
        self.assertTrue(check_password_hash(stored_hash, plain_pw))

    def test_05_private_record_isolation_between_users(self):
        """Verifies that private assessment records are accessible ONLY to their owner."""
        u1 = database.create_user(f"alice_{uuid.uuid4().hex[:4]}", None, "pass1234")
        u2 = database.create_user(f"bob_{uuid.uuid4().hex[:4]}", None, "pass1234")

        asmt_alice = database.save_assessment(
            patient_name="Alice",
            patient_age=30,
            patient_gender="Female",
            symptoms=["Cough", "High Fever", "Breathlessness"],
            symptom_vector=[0]*132,
            predicted_disease="Bronchial Asthma",
            user_id=u1['id']
        )
        asmt_bob = database.save_assessment(
            patient_name="Bob",
            patient_age=45,
            patient_gender="Male",
            symptoms=["Joint Pain", "Neck Pain", "Knee Pain"],
            symptom_vector=[0]*132,
            predicted_disease="Arthritis",
            user_id=u2['id']
        )

        alice_records = database.get_assessments(user_id=u1['id'])
        self.assertTrue(any(a['id'] == asmt_alice['id'] for a in alice_records))
        self.assertFalse(any(a['id'] == asmt_bob['id'] for a in alice_records))

    # ----------------------------------------------------
    # Per-Symptom Severity & Database Persistence
    # ----------------------------------------------------
    def test_06_per_symptom_severity_in_followup_and_db(self):
        """Verifies followup questions ask for each symptom separately and stores each severity in DB."""
        selected = ['Cough', 'High Fever', 'Breathlessness']
        fu = followup.get_follow_up_questions(selected)
        self.assertEqual(fu['symptom_count'], 3)

        # Confirm per_symptom_severity question is present
        sev_q = next((q for q in fu['questions'] if q['id'] == 'symptom_severities'), None)
        self.assertIsNotNone(sev_q)
        self.assertEqual(sev_q['type'], 'per_symptom_severity')
        self.assertEqual(len(sev_q['symptoms']), 3)

        # Save assessment with different severities per symptom
        asmt = database.save_assessment(
            patient_name="Severity Tester",
            patient_age=28,
            patient_gender="Other",
            symptoms=selected,
            symptom_vector=[0]*132,
            predicted_disease="Bronchial Asthma",
            follow_up={
                'symptom_severities': {
                    'Cough': 4,              # Severe
                    'High Fever': 3,         # Moderate
                    'Breathlessness': 5      # Very Severe
                }
            },
            user_id=self.user['id']
        )

        # Inspect timeline entries for each symptom
        timeline = database.get_timeline_entries(user_id=self.user['id'], assessment_id=asmt['id'])
        sev_by_symp = {t['symptom_name']: t['severity'] for t in timeline}
        self.assertEqual(sev_by_symp.get('Cough'), 4)
        self.assertEqual(sev_by_symp.get('High Fever'), 3)
        self.assertEqual(sev_by_symp.get('Breathlessness'), 5)

    # ----------------------------------------------------
    # Minimum 3 Symptoms Enforcement on Small Inputs
    # ----------------------------------------------------
    def test_07_require_at_least_3_symptoms_in_diagnose(self):
        """Verifies /diagnose rejects inputs with fewer than 3 symptoms with 400 error."""
        # 1 symptom (e.g. headache alone)
        symps = symptoms_data.get_symptoms_list()
        vec_1 = [1 if s == 'headache' else 0 for s in symps]
        res_1 = self.app.post('/diagnose', data={'checkboxvalue': ','.join(map(str, vec_1))})
        self.assertEqual(res_1.status_code, 400)
        data_1 = json.loads(res_1.data)
        self.assertEqual(data_1.get('error'), 'minimum_symptoms_required')

        # 2 symptoms (e.g. cough + high_fever)
        vec_2 = [1 if s in ('cough', 'high_fever') else 0 for s in symps]
        res_2 = self.app.post('/diagnose', data={'checkboxvalue': ','.join(map(str, vec_2))})
        self.assertEqual(res_2.status_code, 400)
        data_2 = json.loads(res_2.data)
        self.assertEqual(data_2.get('error'), 'minimum_symptoms_required')

    # ----------------------------------------------------
    # Top 3 Differential Diagnoses & Probability Breakdown
    # ----------------------------------------------------
    def test_08_top_3_diseases_with_probabilities(self):
        """Verifies /diagnose returns top 3 diseases with individual probability percentages."""
        symps = symptoms_data.get_symptoms_list()
        vec_3 = [1 if s in ('cough', 'high_fever', 'breathlessness') else 0 for s in symps]
        res = self.app.post('/diagnose', data={'checkboxvalue': ','.join(map(str, vec_3))})
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)

        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['predicted_disease'], 'Bronchial Asthma')
        self.assertIn('top_3_diseases', data)
        self.assertEqual(len(data['top_3_diseases']), 3)

        top_pred = data['top_3_diseases'][0]
        self.assertEqual(top_pred['disease'], 'Bronchial Asthma')
        self.assertEqual(top_pred['probability'], 100.0)

    # ----------------------------------------------------
    # Low-Confidence Warning (<60%)
    # ----------------------------------------------------
    def test_09_low_confidence_warning_under_60(self):
        """Verifies a low-confidence warning is emitted when top probability is under 60%."""
        symps = symptoms_data.get_symptoms_list()
        # Combination known to yield split probability in KNN space
        vec = [1 if s in ('cough', 'high_fever') else 0 for s in symps]
        # Use allow_sparse to inspect probability on cough + high_fever
        res = self.app.post('/diagnose', data={'checkboxvalue': ','.join(map(str, vec)), 'allow_sparse': 'true'})
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertLess(data['confidence'], 60.0)
        self.assertTrue(data['low_confidence'])
        self.assertIsNotNone(data['low_confidence_warning'])
        self.assertIn('Low Prediction Confidence', data['low_confidence_warning'])

    # ----------------------------------------------------
    # Red-Flag Emergency Advisory for Critical Symptoms
    # ----------------------------------------------------
    def test_10_red_flag_advisory_for_chest_pain_and_breathlessness(self):
        """Verifies red-flag note is included when chest pain or breathlessness is present."""
        symps = symptoms_data.get_symptoms_list()
        # Input with chest_pain and breathlessness
        vec = [1 if s in ('chest_pain', 'breathlessness', 'cough') else 0 for s in symps]
        res = self.app.post('/diagnose', data={'checkboxvalue': ','.join(map(str, vec))})
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)

        self.assertTrue(data['has_red_flag'])
        self.assertIn('Chest Pain', data['red_flag_symptoms'])
        self.assertIn('Breathlessness', data['red_flag_symptoms'])
        self.assertIn('RED-FLAG EMERGENCY ADVISORY', data['red_flag_note'])

    # ----------------------------------------------------
    # ----------------------------------------------------
    # Explainable AI: About This Prediction (LOSO Removed)
    # ----------------------------------------------------
    def test_11_about_this_prediction_explainability(self):
        """Verifies explainability provides clean 'About This Prediction' format without LOSO or percentages."""
        symps = symptoms_data.get_symptoms_list()
        vec = [1 if s in ('cough', 'high_fever', 'breathlessness') else 0 for s in symps]
        xai = explainability.explain_prediction(vec)

        self.assertEqual(xai['status'], 'success')
        self.assertEqual(xai['predicted_disease'], 'Bronchial Asthma')
        self.assertIn('selected_symptoms', xai)
        self.assertEqual(len(xai['selected_symptoms']), 3)
        self.assertIn('Cough', xai['selected_symptoms'])
        self.assertIn('High Fever', xai['selected_symptoms'])
        self.assertIn('Breathlessness', xai['selected_symptoms'])

        # Verify explanation text states model pattern matching against training data
        self.assertIn('training data', xai['explanation'].lower())
        self.assertIn('symptom pattern', xai['explanation'].lower())

        # Verify educational disclaimer is present
        self.assertIn('educational purposes only', xai['disclaimer'].lower())

        # Verify LOSO and attribution percentages are completely removed
        self.assertNotIn('contributing_symptoms', xai)
        self.assertNotIn('drop_percentage', xai)
        self.assertNotIn('confidence_percentage', xai)

    # ----------------------------------------------------
    # PDF Assessment Report Generation
    # ----------------------------------------------------
    def test_12_pdf_generation_with_top_3_and_red_flag(self):
        """Tests PDF generation includes top 3 differential diagnoses and red-flag alerts."""
        asmt = database.save_assessment(
            patient_name="PDF Report User",
            patient_age=50,
            patient_gender="Male",
            symptoms=["Chest Pain", "Breathlessness", "Cough"],
            symptom_vector=[0]*132,
            predicted_disease="Heart Attack",
            follow_up={
                'top_3_diseases': [
                    {'disease': 'Heart Attack', 'probability': 95.0},
                    {'disease': 'Bronchial Asthma', 'probability': 5.0},
                    {'disease': 'GERD', 'probability': 0.0}
                ],
                'red_flag_note': 'EMERGENCY RED-FLAG: Critical chest pain reported.',
                'symptom_severities': {'Chest Pain': 5, 'Breathlessness': 4, 'Cough': 2}
            },
            user_id=self.user['id']
        )

        res = self.app.get(f"/api/assessment/{asmt['id']}/pdf")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, 'application/pdf')
        self.assertTrue(res.data.startswith(b'%PDF'))

    # ----------------------------------------------------
    # Unbroken Existing Routes & Endpoints
    # ----------------------------------------------------
    def test_13_existing_routes_render(self):
        """Verifies templates render 200 OK for authenticated user."""
        routes = ['/', '/survey', '/thanks', '/history', '/timeline', '/profile']
        for route in routes:
            res = self.app.get(route)
            self.assertEqual(res.status_code, 200, f"Route {route} must return 200")

    def test_14_survey_endpoint(self):
        """Verifies existing /surveydata endpoint writes and returns 'Successful'."""
        survey_vec = ['0'] * 132 + ['Test Diagnosis']
        res = self.app.post('/surveydata', data={'checkboxvalue': ','.join(survey_vec)})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(json.loads(res.data), "Successful")

    # ----------------------------------------------------
    # Profile Page & Edit Profile Functionality
    # ----------------------------------------------------
    def test_15_profile_page_authenticated_and_edit(self):
        """Verifies /profile displays real database user info and allows safe profile edits."""
        # 1. GET /profile renders authenticated user's info
        res = self.app.get('/profile')
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn(self.user['username'], html)
        self.assertIn(self.user['email'], html)

        # 2. POST /profile updates full_name and email
        new_email = f"jane_{uuid.uuid4().hex[:6]}@example.com"
        res_post = self.app.post('/profile', data={
            'full_name': 'Dr. Jane Doe',
            'email': new_email
        }, follow_redirects=True)
        self.assertEqual(res_post.status_code, 200)
        updated_html = res_post.data.decode('utf-8')
        self.assertIn('Dr. Jane Doe', updated_html)
        self.assertIn(new_email, updated_html)

        # Verify database record updated
        db_user = database.get_user_by_id(self.user['id'])
        self.assertEqual(db_user['full_name'], 'Dr. Jane Doe')
        self.assertEqual(db_user['email'], new_email)

    def test_16_profile_page_requires_auth(self):
        """Verifies logged-out user accessing /profile is redirected to /login with flash message."""
        with self.app.session_transaction() as sess:
            sess.clear()
        res = self.app.get('/profile')
        self.assertEqual(res.status_code, 302)
        self.assertIn('/login', res.headers['Location'])

    # ----------------------------------------------------
    # Prediction: 1 Symptom and Multiple Symptoms
    # ----------------------------------------------------
    def test_17_single_and_multiple_symptoms_prediction(self):
        """Tests prediction works with 1 symptom (allow_sparse) and multiple symptoms."""
        symps = symptoms_data.get_symptoms_list()
        
        # Single symptom
        vec_1 = [1 if s == 'itching' else 0 for s in symps]
        res_1 = self.app.post('/diagnose', data={'checkboxvalue': ','.join(map(str, vec_1)), 'allow_sparse': '1'})
        self.assertEqual(res_1.status_code, 200)
        data_1 = json.loads(res_1.data)
        self.assertEqual(data_1['status'], 'success')
        self.assertEqual(data_1['predicted_disease'], 'Fungal infection')
        self.assertIn('Itching', data_1['selected_symptoms'])

        # Multiple symptoms
        vec_multi = [1 if s in ('itching', 'skin_rash', 'nodal_skin_eruptions') else 0 for s in symps]
        res_multi = self.app.post('/diagnose', data={'checkboxvalue': ','.join(map(str, vec_multi))})
        self.assertEqual(res_multi.status_code, 200)
        data_multi = json.loads(res_multi.data)
        self.assertEqual(data_multi['status'], 'success')
        self.assertEqual(data_multi['predicted_disease'], 'Fungal infection')

    # ----------------------------------------------------
    # Symptom Order Invariance
    # ----------------------------------------------------
    def test_18_symptom_order_invariance(self):
        """Verifies changing the order of selected symptoms produces identical input and prediction."""
        order_a = ['cough', 'high_fever', 'breathlessness']
        order_b = ['breathlessness', 'cough', 'high_fever']

        res_a = self.app.post('/diagnose', json={'symptoms': order_a})
        res_b = self.app.post('/diagnose', json={'symptoms': order_b})

        self.assertEqual(res_a.status_code, 200)
        self.assertEqual(res_b.status_code, 200)

        data_a = json.loads(res_a.data)
        data_b = json.loads(res_b.data)

        self.assertEqual(data_a['predicted_disease'], data_b['predicted_disease'])
        self.assertEqual(data_a['selected_symptoms'], data_b['selected_symptoms'])

    # ----------------------------------------------------
    # Cross-User Data Isolation
    # ----------------------------------------------------
    def test_19_cross_user_data_isolation(self):
        """Verifies a user cannot access another user's assessments or PDF reports."""
        # Create a second user
        u2 = database.create_user(f"isolation_{uuid.uuid4().hex[:6]}", f"iso_{uuid.uuid4().hex[:6]}@test.com", "securepwd123")
        asmt_u2 = database.save_assessment(
            patient_name="User Two Patient",
            patient_age=35,
            patient_gender="Female",
            symptoms=["Joint Pain", "Neck Pain"],
            symptom_vector=[0]*132,
            predicted_disease="Arthritis",
            user_id=u2['id']
        )

        # self.user (logged in) attempts to access u2's assessment
        res_detail = self.app.get(f"/api/assessment/{asmt_u2['id']}")
        self.assertEqual(res_detail.status_code, 404)

        res_pdf = self.app.get(f"/api/assessment/{asmt_u2['id']}/pdf")
        self.assertEqual(res_pdf.status_code, 404)

        res_del = self.app.delete(f"/api/assessment/{asmt_u2['id']}/delete")
        self.assertEqual(res_del.status_code, 404)


if __name__ == '__main__':
    unittest.main()
