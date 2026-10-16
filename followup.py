"""
Smart Follow-Up Questions Module
Provides category-based contextual follow-up questions for disease diagnosis.
Answers are kept strictly separate from the 132-symptom ML feature vector.
"""

SYMPTOM_CATEGORIES = {
    'dermatological': [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'pus_filled_pimples',
        'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting',
        'small_dents_in_nails', 'inflammatory_nails', 'blister',
        'red_sore_around_nose', 'yellow_crust_ooze', 'dischromic _patches',
        'red_spots_over_body', 'bruising', 'internal_itching'
    ],
    'respiratory': [
        'continuous_sneezing', 'cough', 'breathlessness', 'phlegm',
        'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
        'runny_nose', 'congestion', 'chest_pain', 'loss_of_smell',
        'mucoid_sputum', 'rusty_sputum', 'blood_in_sputum', 'patches_in_throat'
    ],
    'gastrointestinal': [
        'stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting',
        'indigestion', 'nausea', 'loss_of_appetite', 'constipation',
        'abdominal_pain', 'diarrhoea', 'acute_liver_failure', 'fluid_overload',
        'swelling_of_stomach', 'passage_of_gases', 'belly_pain',
        'distention_of_abdomen', 'stomach_bleeding', 'yellowish_skin',
        'yellowing_of_eyes', 'pain_during_bowel_movements',
        'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus'
    ],
    'musculoskeletal': [
        'joint_pain', 'muscle_wasting', 'back_pain', 'neck_pain',
        'cramps', 'knee_pain', 'hip_joint_pain', 'muscle_weakness',
        'stiff_neck', 'swelling_joints', 'movement_stiffness',
        'painful_walking', 'muscle_pain', 'weakness_in_limbs',
        'spinning_movements', 'loss_of_balance', 'unsteadiness',
        'weakness_of_one_body_side', 'headache', 'dizziness'
    ],
    'systemic': [
        'shivering', 'chills', 'fatigue', 'high_fever', 'mild_fever',
        'sweating', 'dehydration', 'malaise', 'swelled_lymph_nodes',
        'palpitations', 'toxic_look_(typhos)', 'anxiety', 'restlessness',
        'lethargy', 'depression', 'irritability', 'cold_hands_and_feets',
        'fast_heart_rate'
    ],
    'metabolic_urinary': [
        'burning_micturition', 'spotting_ urination', 'dark_urine',
        'yellow_urine', 'bladder_discomfort', 'foul_smell_of urine',
        'continuous_feel_of_urine', 'polyuria', 'weight_gain',
        'weight_loss', 'irregular_sugar_level', 'excessive_hunger',
        'increased_appetite', 'enlarged_thyroid', 'puffy_face_and_eyes',
        'swollen_legs', 'swollen_extremeties'
    ]
}

def identify_categories(selected_symptoms):
    """
    Identifies which symptom categories are represented in the user's selected symptoms.
    """
    detected_cats = set()
    cleaned_selected = [s.strip().lower().replace(' ', '_') for s in selected_symptoms]

    for cat, symptoms in SYMPTOM_CATEGORIES.items():
        for symp in symptoms:
            for user_s in cleaned_selected:
                if symp in user_s or user_s in symp:
                    detected_cats.add(cat)
                    break
    return list(detected_cats)

def get_follow_up_questions(selected_symptoms):
    """
    Generates a tailored list of questions based on selected symptoms and active categories.
    """
    categories = identify_categories(selected_symptoms)

    # Core contextual questions applicable to any symptomatic assessment
    questions = [
        {
            'id': 'duration',
            'category': 'core',
            'title': 'Symptom Duration',
            'question': 'When did you first notice these symptoms?',
            'required': False,
            'type': 'select',
            'options': [
                'Less than 24 hours (Acute)',
                '1 to 3 days',
                '4 to 7 days',
                '1 to 2 weeks',
                'More than 2 weeks (Persistent / Chronic)'
            ],
            'help_text': 'Helps distinguish acute episodes from chronic conditions.'
        }
    ]

    # Individual severity question asking for each selected symptom separately
    if selected_symptoms:
        questions.append({
            'id': 'symptom_severities',
            'category': 'core',
            'title': 'Individual Symptom Severity',
            'question': 'Please select the severity level for each of your reported symptoms:',
            'required': False,
            'type': 'per_symptom_severity',
            'symptoms': [
                {
                    'name': s,
                    'key': s.strip().lower().replace(' ', '_'),
                    'label': s.replace('_', ' ').strip().title()
                } for s in selected_symptoms
            ],
            'options': [
                {'value': 1, 'label': '1 - Very Mild'},
                {'value': 2, 'label': '2 - Mild'},
                {'value': 3, 'label': '3 - Moderate'},
                {'value': 4, 'label': '4 - Severe'},
                {'value': 5, 'label': '5 - Very Severe'}
            ],
            'help_text': 'Rate each symptom individually to track its personal severity on your progression timeline.'
        })
    else:
        questions.append({
            'id': 'severity',
            'category': 'core',
            'title': 'Current Severity',
            'question': 'How would you rate the overall severity of your symptoms right now?',
            'required': False,
            'type': 'rating',
            'options': [
                {'value': 1, 'label': '1 - Very Mild (Barely noticeable)'},
                {'value': 2, 'label': '2 - Mild (Noticeable, minor annoyance)'},
                {'value': 3, 'label': '3 - Moderate (Interferes with routine)'},
                {'value': 4, 'label': '4 - Severe (Significantly limiting)'},
                {'value': 5, 'label': '5 - Very Severe (Incapacitating)'}
            ],
            'help_text': 'Used for symptom tracking on your progression timeline.'
        })

    questions.extend([
        {
            'id': 'frequency',
            'category': 'core',
            'title': 'Frequency / Occurrence',
            'question': 'How frequently do your symptoms occur?',
            'required': False,
            'type': 'select',
            'options': [
                'Occasional (Comes up rarely or once a day)',
                'Intermittent (Comes and goes in waves)',
                'Frequent (Present most of the day)',
                'Continuous (Constant and uninterrupted)'
            ],
            'help_text': 'Indicates whether the condition is episodic or continuous.'
        },
        {
            'id': 'progression',
            'category': 'core',
            'title': 'Symptom Progression',
            'question': 'How has the intensity of your symptoms changed since they started?',
            'required': False,
            'type': 'select',
            'options': [
                'Improving (Getting milder)',
                'Unchanged (Remaining steady)',
                'Gradually worsening',
                'Rapidly worsening'
            ],
            'help_text': 'Tracks whether the condition is resolving or escalating.'
        }
    ])

    # Category-specific targeted questions
    if 'respiratory' in categories:
        questions.append({
            'id': 'respiratory_trigger',
            'category': 'respiratory',
            'title': 'Respiratory Context',
            'question': 'Do you experience shortness of breath or wheezing when resting or lying down?',
            'required': False,
            'type': 'select',
            'options': [
                'No shortness of breath at rest',
                'Mild shortness of breath with physical exertion',
                'Yes, noticeable breathing difficulty even while resting'
            ],
            'help_text': 'Respiratory context for follow-up tracking.'
        })

    if 'dermatological' in categories:
        questions.append({
            'id': 'skin_spread',
            'category': 'dermatological',
            'title': 'Skin Condition Context',
            'question': 'Have the skin lesions, rash, or itching spread to larger body areas?',
            'required': False,
            'type': 'select',
            'options': [
                'No, confined to one isolated spot',
                'Gradually spreading to nearby areas',
                'Widespread across multiple body regions'
            ],
            'help_text': 'Tracks dermatological spread over time.'
        })

    if 'gastrointestinal' in categories:
        questions.append({
            'id': 'gi_hydration',
            'category': 'gastrointestinal',
            'title': 'Digestive & Hydration Context',
            'question': 'Are you able to keep water and light food down comfortably?',
            'required': False,
            'type': 'select',
            'options': [
                'Yes, eating and drinking without difficulty',
                'Occasional nausea/discomfort, but able to hydrate',
                'Difficulty keeping fluids down or severe cramping'
            ],
            'help_text': 'Evaluates hydration tolerance.'
        })

    if 'musculoskeletal' in categories:
        questions.append({
            'id': 'mobility_impact',
            'category': 'musculoskeletal',
            'title': 'Mobility & Joint Context',
            'question': 'Does joint stiffness or muscle pain impede walking or daily movement?',
            'required': False,
            'type': 'select',
            'options': [
                'Full mobility, minor stiffness only',
                'Moderate restriction (e.g. morning stiffness)',
                'Significant difficulty walking or standing'
            ],
            'help_text': 'Assesses physical mobility impact.'
        })

    if 'systemic' in categories:
        questions.append({
            'id': 'fever_pattern',
            'category': 'systemic',
            'title': 'Temperature & Energy Context',
            'question': 'Have you experienced high fever (above 101°F/38.3°C) or severe chills/sweating?',
            'required': False,
            'type': 'select',
            'options': [
                'No fever or normal temperature',
                'Low-grade fever with mild chills',
                'High fever with recurrent chills or night sweats'
            ],
            'help_text': 'Helps monitor systemic infection indicators.'
        })

    return {
        'categories': categories,
        'symptom_count': len(selected_symptoms),
        'questions': questions,
        'disclaimer': (
            'Notice: Follow-up answers are gathered solely for contextual tracking, '
            'personal record-keeping, and symptom progression visualization. They do not '
            'alter the underlying machine learning model prediction and do not constitute '
            'clinical medical advice.'
        )
    }
