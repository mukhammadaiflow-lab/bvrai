"""
Industry Profiles Module

This module provides comprehensive profiles for various industries,
including characteristics, services, and conversation patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .base import (
    IndustryType,
    IndustryProfile,
    RegulationType,
    ConversationPhase,
    ConversationPattern,
    BehaviorGuideline,
    ServiceOffering,
    EmergencyProtocol,
    IndustryMetric,
    SeasonalPattern,
)


# Healthcare - General Practice
HEALTHCARE_GENERAL_PROFILE = IndustryProfile(
    industry_type=IndustryType.HEALTHCARE_GENERAL,
    name="Healthcare - General Practice",
    description="Medical clinics, primary care physicians, and general healthcare providers",
    typical_customer_segments=[
        "New patients seeking primary care",
        "Existing patients needing appointments",
        "Patients with urgent care needs",
        "Patients needing prescription refills",
        "Insurance and billing inquiries",
    ],
    common_pain_points=[
        "Long wait times for appointments",
        "Difficulty reaching the office",
        "Confusion about insurance coverage",
        "Need for same-day appointments",
        "Prescription refill delays",
    ],
    value_propositions=[
        "Convenient appointment scheduling",
        "Compassionate patient care",
        "Comprehensive health services",
        "Insurance assistance",
        "Quick response to urgent needs",
    ],
    applicable_regulations=[
        RegulationType.HIPAA,
        RegulationType.HITECH,
    ],
    conversation_patterns=[
        ConversationPattern(
            id="healthcare_appointment",
            name="Appointment Scheduling",
            description="Standard appointment booking flow",
            phase=ConversationPhase.INFORMATION,
            trigger_intents=["schedule_appointment", "book_visit"],
            trigger_keywords=["appointment", "schedule", "see the doctor", "visit"],
            response_templates=[
                "I'd be happy to help you schedule an appointment. Are you an existing patient with us?",
                "Let me help you find a convenient appointment time. What type of visit do you need?",
            ],
            follow_up_questions=[
                "What is the reason for your visit?",
                "Do you have a preferred day or time?",
                "Will you be using insurance for this visit?",
            ],
        ),
        ConversationPattern(
            id="healthcare_urgent",
            name="Urgent Care Assessment",
            description="Assessing urgent care needs",
            phase=ConversationPhase.DISCOVERY,
            trigger_intents=["urgent_care", "emergency"],
            trigger_keywords=["urgent", "emergency", "right away", "pain", "sick"],
            response_templates=[
                "I understand you need to be seen soon. Can you tell me more about your symptoms?",
                "I hear that this is urgent. Let me see what we can do to help you today.",
            ],
        ),
    ],
    behavior_guidelines=[
        BehaviorGuideline(
            id="healthcare_empathy",
            category="Communication",
            guideline="Always acknowledge patient concerns with empathy",
            rationale="Patients calling healthcare offices are often worried or in discomfort",
            priority=1,
            do_examples=[
                "I understand you're not feeling well. Let me help you.",
                "I'm sorry to hear you're experiencing that. We'll get you taken care of.",
            ],
            dont_examples=[
                "What's wrong with you?",
                "Just tell me what you need.",
            ],
        ),
        BehaviorGuideline(
            id="healthcare_privacy",
            category="Compliance",
            guideline="Never discuss patient information without verification",
            rationale="HIPAA requires protection of patient health information",
            priority=1,
            do_examples=[
                "For your privacy, can you verify your date of birth?",
                "I'll need to verify your identity before discussing your records.",
            ],
        ),
    ],
    common_services=[
        ServiceOffering(
            id="annual_physical",
            name="Annual Physical Exam",
            description="Comprehensive yearly health examination",
            category="Preventive Care",
            typical_duration="45-60 minutes",
            benefits=[
                "Early detection of health issues",
                "Personalized health recommendations",
                "Insurance often covers at 100%",
            ],
            qualification_questions=[
                "When was your last physical exam?",
                "Do you have any specific health concerns to discuss?",
            ],
        ),
        ServiceOffering(
            id="sick_visit",
            name="Sick Visit",
            description="Evaluation and treatment for illness",
            category="Acute Care",
            typical_duration="15-20 minutes",
        ),
    ],
    emergency_protocols=[
        EmergencyProtocol(
            id="medical_emergency",
            trigger_keywords=["chest pain", "can't breathe", "stroke", "unconscious", "severe bleeding"],
            trigger_phrases=["having a heart attack", "can't feel my arm", "face is drooping"],
            severity="critical",
            immediate_response="This sounds like it could be a medical emergency.",
            transfer_required=True,
            acknowledgment_script="Please call 911 immediately or go to the nearest emergency room. "
                                 "If you're having chest pain, chew an aspirin if you're not allergic.",
        ),
    ],
    key_metrics=[
        IndustryMetric(
            id="patient_satisfaction",
            name="Patient Satisfaction Score",
            description="Overall patient satisfaction rating",
            unit="percentage",
            industry_average=85.0,
            good_threshold=90.0,
            excellent_threshold=95.0,
        ),
        IndustryMetric(
            id="appointment_booking_rate",
            name="Appointment Booking Rate",
            description="Percentage of calls resulting in booked appointments",
            unit="percentage",
            industry_average=65.0,
            good_threshold=75.0,
            excellent_threshold=85.0,
        ),
    ],
    seasonal_patterns=[
        SeasonalPattern(
            id="flu_season",
            name="Flu Season",
            description="Increased demand for flu shots and sick visits",
            start_month=10,
            end_month=3,
            peak_months=[11, 12, 1, 2],
            demand_multiplier=1.4,
            talking_points=[
                "Have you gotten your flu shot this season?",
                "We have flu shots available - would you like to schedule one?",
            ],
        ),
    ],
    tone="professional",
    formality_level=3,
    empathy_level=5,
    urgency_sensitivity=5,
    typical_business_hours={
        "monday": "8:00 AM - 5:00 PM",
        "tuesday": "8:00 AM - 5:00 PM",
        "wednesday": "8:00 AM - 5:00 PM",
        "thursday": "8:00 AM - 5:00 PM",
        "friday": "8:00 AM - 5:00 PM",
    },
    after_hours_handling="For medical emergencies, please call 911 or go to the nearest emergency room.",
)


# Dental Practice
DENTAL_PROFILE = IndustryProfile(
    industry_type=IndustryType.HEALTHCARE_DENTAL,
    name="Dental Practice",
    description="Dental clinics, orthodontists, and oral surgery practices",
    typical_customer_segments=[
        "New patients seeking a dentist",
        "Patients needing routine cleanings",
        "Patients with dental emergencies",
        "Cosmetic dentistry inquiries",
        "Orthodontic consultations",
    ],
    common_pain_points=[
        "Dental anxiety and fear",
        "Uncertainty about costs",
        "Insurance coverage questions",
        "Tooth pain or emergencies",
        "Finding appointments that fit schedule",
    ],
    value_propositions=[
        "Gentle and comfortable care",
        "Modern technology and techniques",
        "Flexible payment options",
        "Family-friendly environment",
        "Same-day emergency appointments",
    ],
    applicable_regulations=[
        RegulationType.HIPAA,
    ],
    conversation_patterns=[
        ConversationPattern(
            id="dental_new_patient",
            name="New Patient Welcome",
            description="Welcoming new patients",
            phase=ConversationPhase.DISCOVERY,
            trigger_intents=["new_patient", "first_visit"],
            trigger_keywords=["new patient", "never been", "looking for a dentist"],
            response_templates=[
                "Welcome! We'd love to have you as a patient. Have you been to a dentist recently?",
                "We're always happy to welcome new patients. What brought you to look for a new dentist?",
            ],
            follow_up_questions=[
                "When was your last dental visit?",
                "Do you have any immediate dental concerns?",
                "Will you be using dental insurance?",
            ],
        ),
        ConversationPattern(
            id="dental_emergency",
            name="Dental Emergency",
            description="Handling dental emergencies",
            phase=ConversationPhase.DISCOVERY,
            trigger_intents=["dental_emergency", "tooth_pain"],
            trigger_keywords=["emergency", "tooth pain", "broken tooth", "swelling", "knocked out"],
            response_templates=[
                "I'm sorry you're experiencing dental pain. Let me help you get seen as soon as possible.",
                "Dental emergencies can be very uncomfortable. Can you describe what happened?",
            ],
        ),
    ],
    behavior_guidelines=[
        BehaviorGuideline(
            id="dental_anxiety",
            category="Patient Care",
            guideline="Acknowledge and address dental anxiety",
            rationale="Many patients have dental phobia; acknowledgment helps build trust",
            priority=1,
            do_examples=[
                "I understand dental visits can be stressful. Our team is very gentle.",
                "We have many patients who were nervous at first but now feel comfortable with us.",
            ],
        ),
    ],
    common_services=[
        ServiceOffering(
            id="cleaning",
            name="Dental Cleaning",
            description="Professional teeth cleaning and examination",
            category="Preventive",
            typical_duration="45-60 minutes",
            benefits=[
                "Prevents cavities and gum disease",
                "Brighter, healthier smile",
                "Usually covered by insurance twice yearly",
            ],
            common_objections=["I don't have time", "I'm scared of the dentist"],
            objection_responses={
                "time": "We have early morning and evening appointments available to fit your schedule.",
                "fear": "We specialize in treating anxious patients. We go at your pace and can discuss sedation options.",
            },
        ),
        ServiceOffering(
            id="whitening",
            name="Teeth Whitening",
            description="Professional teeth whitening treatment",
            category="Cosmetic",
            typical_duration="60-90 minutes",
            benefits=[
                "Dramatically whiter smile",
                "Safe and effective",
                "Long-lasting results",
            ],
        ),
    ],
    emergency_protocols=[
        EmergencyProtocol(
            id="knocked_out_tooth",
            trigger_keywords=["knocked out", "tooth fell out", "avulsed"],
            severity="high",
            immediate_response="A knocked-out tooth is a dental emergency.",
            transfer_required=True,
            acknowledgment_script="Keep the tooth moist - in milk or saliva if possible. "
                                 "Do not scrub the tooth. Come in immediately or go to an emergency dentist.",
        ),
    ],
    tone="friendly",
    formality_level=2,
    empathy_level=5,
    urgency_sensitivity=4,
)


# Legal Services
LEGAL_PROFILE = IndustryProfile(
    industry_type=IndustryType.LEGAL,
    name="Legal Services",
    description="Law firms, attorneys, and legal practices",
    typical_customer_segments=[
        "Individuals needing legal representation",
        "Businesses requiring legal counsel",
        "Personal injury victims",
        "Family law matters",
        "Estate planning clients",
    ],
    common_pain_points=[
        "Uncertainty about legal rights",
        "Concern about attorney fees",
        "Need for confidential advice",
        "Time-sensitive legal matters",
        "Complex legal procedures",
    ],
    value_propositions=[
        "Expert legal guidance",
        "Confidential consultation",
        "Aggressive representation",
        "Clear communication",
        "Results-oriented approach",
    ],
    applicable_regulations=[
        RegulationType.ABA_ETHICS,
        RegulationType.ATTORNEY_CLIENT,
    ],
    conversation_patterns=[
        ConversationPattern(
            id="legal_consultation",
            name="Initial Consultation",
            description="Scheduling initial legal consultation",
            phase=ConversationPhase.QUALIFICATION,
            trigger_intents=["legal_help", "attorney_consultation"],
            trigger_keywords=["lawyer", "attorney", "legal help", "consultation", "case"],
            response_templates=[
                "Thank you for calling. I can help schedule a consultation with an attorney. What type of legal matter is this regarding?",
                "Our attorneys handle various legal matters. Can you tell me a bit about your situation?",
            ],
            follow_up_questions=[
                "What type of legal matter is this regarding?",
                "Is this a time-sensitive matter?",
                "Have you spoken with any other attorneys about this?",
            ],
        ),
    ],
    behavior_guidelines=[
        BehaviorGuideline(
            id="legal_no_advice",
            category="Compliance",
            guideline="Never provide legal advice - only attorneys can do that",
            rationale="Providing legal advice without a license is unauthorized practice of law",
            priority=1,
            do_examples=[
                "I can schedule you with an attorney who can advise you on that.",
                "That's a great question for our attorney to address during your consultation.",
            ],
            dont_examples=[
                "You should definitely sue them.",
                "I think you have a strong case.",
            ],
        ),
        BehaviorGuideline(
            id="legal_confidentiality",
            category="Compliance",
            guideline="Maintain strict confidentiality",
            rationale="Attorney-client privilege begins when someone seeks legal advice",
            priority=1,
            do_examples=[
                "Everything you share with us is kept strictly confidential.",
                "Your information is protected by attorney-client privilege.",
            ],
        ),
    ],
    common_services=[
        ServiceOffering(
            id="free_consultation",
            name="Free Initial Consultation",
            description="Complimentary case evaluation",
            category="Consultation",
            typical_duration="30-45 minutes",
            benefits=[
                "No-obligation evaluation",
                "Understand your legal options",
                "Meet with an experienced attorney",
            ],
        ),
        ServiceOffering(
            id="personal_injury",
            name="Personal Injury Representation",
            description="Legal representation for injury cases",
            category="Litigation",
            benefits=[
                "No fee unless we win",
                "Maximum compensation pursuit",
                "Handle all insurance negotiations",
            ],
        ),
    ],
    tone="professional",
    formality_level=4,
    empathy_level=4,
    urgency_sensitivity=4,
)


# Real Estate
REAL_ESTATE_PROFILE = IndustryProfile(
    industry_type=IndustryType.REAL_ESTATE,
    name="Real Estate",
    description="Real estate agencies, agents, and property management",
    typical_customer_segments=[
        "First-time home buyers",
        "Home sellers",
        "Investors",
        "Renters",
        "Commercial property clients",
    ],
    common_pain_points=[
        "Finding the right property",
        "Understanding market conditions",
        "Navigating the buying/selling process",
        "Mortgage and financing questions",
        "Timing the market",
    ],
    value_propositions=[
        "Local market expertise",
        "Personalized property search",
        "Negotiation skills",
        "Full-service support",
        "Network of trusted professionals",
    ],
    applicable_regulations=[
        RegulationType.FAIR_HOUSING,
        RegulationType.RESPA,
    ],
    conversation_patterns=[
        ConversationPattern(
            id="real_estate_buyer",
            name="Buyer Inquiry",
            description="Handling buyer inquiries",
            phase=ConversationPhase.QUALIFICATION,
            trigger_intents=["buy_home", "property_search"],
            trigger_keywords=["buy", "purchase", "looking for a home", "house hunting"],
            response_templates=[
                "How exciting that you're looking to buy! What areas are you interested in?",
                "I'd love to help you find your perfect home. What are you looking for?",
            ],
            follow_up_questions=[
                "What's your ideal location or neighborhood?",
                "How many bedrooms and bathrooms do you need?",
                "What's your budget range?",
                "Are you pre-approved for a mortgage?",
                "What's your timeline for buying?",
            ],
        ),
        ConversationPattern(
            id="real_estate_seller",
            name="Seller Inquiry",
            description="Handling seller inquiries",
            phase=ConversationPhase.QUALIFICATION,
            trigger_intents=["sell_home", "list_property"],
            trigger_keywords=["sell", "list", "selling my home", "what's my home worth"],
            response_templates=[
                "Thinking of selling? I can help you understand your home's market value.",
                "Great timing to consider selling. Would you like a free market analysis?",
            ],
            follow_up_questions=[
                "Where is your property located?",
                "What's your timeline for selling?",
                "Have you made any recent improvements?",
            ],
        ),
    ],
    behavior_guidelines=[
        BehaviorGuideline(
            id="real_estate_fair_housing",
            category="Compliance",
            guideline="Never discuss protected classes or steer clients",
            rationale="Fair Housing Act prohibits discrimination",
            priority=1,
            do_examples=[
                "I can show you properties in any neighborhood you're interested in.",
                "The neighborhood has highly-rated schools nearby.",
            ],
            dont_examples=[
                "That's a nice family neighborhood.",
                "This area is mostly [demographic].",
            ],
        ),
    ],
    common_services=[
        ServiceOffering(
            id="market_analysis",
            name="Free Market Analysis",
            description="Complimentary home value assessment",
            category="Seller Services",
            typical_duration="30 minutes",
            benefits=[
                "Know your home's true market value",
                "No obligation",
                "Insights into local market trends",
            ],
        ),
        ServiceOffering(
            id="buyer_representation",
            name="Buyer Representation",
            description="Full-service buyer's agent services",
            category="Buyer Services",
            benefits=[
                "Expert market guidance",
                "Skilled negotiation",
                "Handle all paperwork",
                "Access to listings before they hit market",
            ],
        ),
    ],
    tone="friendly",
    formality_level=2,
    empathy_level=4,
    urgency_sensitivity=3,
)


# HVAC Services
HVAC_PROFILE = IndustryProfile(
    industry_type=IndustryType.HVAC,
    name="HVAC Services",
    description="Heating, ventilation, and air conditioning services",
    typical_customer_segments=[
        "Homeowners with AC/heating issues",
        "New system installation needs",
        "Maintenance agreement customers",
        "Commercial HVAC clients",
        "New construction projects",
    ],
    common_pain_points=[
        "System not working in extreme weather",
        "High energy bills",
        "Uneven heating/cooling",
        "Strange noises or smells",
        "Old system replacement decisions",
    ],
    value_propositions=[
        "24/7 emergency service",
        "Expert technicians",
        "Upfront pricing",
        "Energy-efficient solutions",
        "Maintenance plans for prevention",
    ],
    conversation_patterns=[
        ConversationPattern(
            id="hvac_no_heat_ac",
            name="No Heat/AC Emergency",
            description="Handling no heat or AC situations",
            phase=ConversationPhase.DISCOVERY,
            trigger_intents=["hvac_emergency", "no_heat", "no_ac"],
            trigger_keywords=[
                "no heat", "no air", "not cooling", "not heating",
                "AC broken", "furnace not working", "frozen",
            ],
            response_templates=[
                "I'm sorry to hear you're without heat/AC. Let me get a technician out to you as quickly as possible.",
                "I understand how uncomfortable that must be. We'll prioritize getting your system running.",
            ],
            follow_up_questions=[
                "Is the system making any unusual sounds?",
                "When did you first notice the problem?",
                "Has anything changed recently - power outage, thermostat adjustment?",
                "What type of system do you have?",
            ],
        ),
        ConversationPattern(
            id="hvac_maintenance",
            name="Maintenance Request",
            description="Scheduling routine maintenance",
            phase=ConversationPhase.INFORMATION,
            trigger_intents=["hvac_maintenance", "tune_up"],
            trigger_keywords=["maintenance", "tune-up", "check", "service"],
            response_templates=[
                "Regular maintenance is great for keeping your system running efficiently. When would you like to schedule?",
                "I can help you schedule a maintenance visit. Are you interested in our maintenance plan for ongoing savings?",
            ],
        ),
    ],
    behavior_guidelines=[
        BehaviorGuideline(
            id="hvac_urgency",
            category="Service",
            guideline="Recognize weather-related urgency",
            rationale="Extreme temperatures make HVAC issues dangerous",
            priority=1,
            do_examples=[
                "I know it's freezing outside - let me see our earliest availability.",
                "With this heat, I want to get someone out to you as soon as possible.",
            ],
        ),
    ],
    common_services=[
        ServiceOffering(
            id="diagnostic",
            name="System Diagnostic",
            description="Complete system inspection and diagnosis",
            category="Service",
            typical_duration="60-90 minutes",
            price_range="$79-$129",
            benefits=[
                "Identify the root cause",
                "Upfront repair quote",
                "Applied to repair cost",
            ],
        ),
        ServiceOffering(
            id="maintenance_plan",
            name="Annual Maintenance Plan",
            description="Preventive maintenance agreement",
            category="Maintenance",
            benefits=[
                "Priority service",
                "Discounted repairs",
                "Extended equipment life",
                "Lower energy bills",
                "Peace of mind",
            ],
            common_objections=["I don't want to pay for something I might not need"],
            objection_responses={
                "cost": "Most repairs we see could have been prevented with regular maintenance. "
                       "The plan pays for itself with just one avoided repair.",
            },
        ),
    ],
    emergency_protocols=[
        EmergencyProtocol(
            id="gas_smell",
            trigger_keywords=["gas smell", "smell gas", "carbon monoxide", "gas leak"],
            severity="critical",
            immediate_response="If you smell gas, this is a safety emergency.",
            transfer_required=False,
            acknowledgment_script="Please leave the house immediately and call 911 or your gas company. "
                                 "Do not use any electrical switches or phones inside. "
                                 "Once you're safe, call us back and we'll send a technician.",
        ),
    ],
    seasonal_patterns=[
        SeasonalPattern(
            id="summer_cooling",
            name="Summer Cooling Season",
            description="Peak demand for AC services",
            start_month=5,
            end_month=9,
            peak_months=[6, 7, 8],
            demand_multiplier=1.8,
            talking_points=[
                "Have you had your AC tuned up for summer?",
                "Now's the time to schedule before the heat hits.",
            ],
            urgency_messaging="Summer is our busiest time - the sooner we schedule, "
                            "the better availability we'll have.",
        ),
        SeasonalPattern(
            id="winter_heating",
            name="Winter Heating Season",
            description="Peak demand for heating services",
            start_month=10,
            end_month=3,
            peak_months=[11, 12, 1, 2],
            demand_multiplier=1.6,
            talking_points=[
                "Is your furnace ready for winter?",
                "Schedule your heating tune-up before the cold arrives.",
            ],
        ),
    ],
    tone="friendly",
    formality_level=2,
    empathy_level=4,
    urgency_sensitivity=5,
    after_hours_handling="We offer 24/7 emergency service for heating and cooling emergencies.",
)


# Restaurant
RESTAURANT_PROFILE = IndustryProfile(
    industry_type=IndustryType.RESTAURANT,
    name="Restaurant",
    description="Restaurants, cafes, and food service establishments",
    typical_customer_segments=[
        "Walk-in diners",
        "Reservation seekers",
        "Takeout/delivery customers",
        "Event and catering inquiries",
        "Gift card purchasers",
    ],
    common_pain_points=[
        "Long wait times",
        "Reservation availability",
        "Dietary restrictions",
        "Special occasion planning",
        "Group dining coordination",
    ],
    value_propositions=[
        "Exceptional dining experience",
        "Fresh, quality ingredients",
        "Welcoming atmosphere",
        "Accommodating special requests",
        "Memorable celebrations",
    ],
    conversation_patterns=[
        ConversationPattern(
            id="restaurant_reservation",
            name="Reservation Request",
            description="Handling reservation inquiries",
            phase=ConversationPhase.INFORMATION,
            trigger_intents=["make_reservation", "book_table"],
            trigger_keywords=["reservation", "book", "table for", "party of"],
            response_templates=[
                "I'd be happy to help you make a reservation. What date and time are you looking for?",
                "We'd love to have you dine with us. How many will be in your party?",
            ],
            follow_up_questions=[
                "What date and time would you prefer?",
                "How many people will be dining?",
                "Is this for a special occasion?",
                "Do you have any seating preferences?",
            ],
        ),
        ConversationPattern(
            id="restaurant_dietary",
            name="Dietary Accommodations",
            description="Handling dietary restriction questions",
            phase=ConversationPhase.INFORMATION,
            trigger_intents=["dietary_info", "allergy_question"],
            trigger_keywords=[
                "allergy", "vegetarian", "vegan", "gluten-free",
                "dairy-free", "nut-free", "dietary",
            ],
            response_templates=[
                "We're happy to accommodate dietary needs. What restrictions should we be aware of?",
                "Our chef can prepare dishes to meet various dietary requirements. What do you need?",
            ],
        ),
    ],
    behavior_guidelines=[
        BehaviorGuideline(
            id="restaurant_hospitality",
            category="Service",
            guideline="Always convey warmth and hospitality",
            rationale="Restaurant experiences begin with the first contact",
            priority=1,
            do_examples=[
                "We'd be delighted to have you join us!",
                "We can't wait to welcome you to our restaurant.",
            ],
        ),
    ],
    common_services=[
        ServiceOffering(
            id="private_dining",
            name="Private Dining",
            description="Private event space for groups",
            category="Events",
            benefits=[
                "Exclusive space for your group",
                "Customized menus available",
                "Dedicated service staff",
            ],
            qualification_questions=[
                "How many guests are you expecting?",
                "What's the occasion?",
                "What's your budget per person?",
            ],
        ),
        ServiceOffering(
            id="catering",
            name="Catering Services",
            description="Off-site catering for events",
            category="Events",
            benefits=[
                "Restaurant-quality food at your venue",
                "Full service or drop-off options",
                "Customized menus",
            ],
        ),
    ],
    tone="friendly",
    formality_level=2,
    empathy_level=3,
    urgency_sensitivity=3,
)


# Salon/Spa
SALON_PROFILE = IndustryProfile(
    industry_type=IndustryType.SALON,
    name="Salon & Spa",
    description="Hair salons, beauty spas, and personal care services",
    typical_customer_segments=[
        "Regular maintenance clients",
        "Special occasion bookings",
        "New clients seeking services",
        "Bridal parties",
        "Gift certificate inquiries",
    ],
    common_pain_points=[
        "Finding appointment availability",
        "Communicating desired style",
        "Service pricing uncertainty",
        "Rushing to appointments",
        "Finding the right stylist",
    ],
    value_propositions=[
        "Expert stylists and technicians",
        "Relaxing atmosphere",
        "Quality products",
        "Personalized consultations",
        "Convenient online booking",
    ],
    conversation_patterns=[
        ConversationPattern(
            id="salon_appointment",
            name="Appointment Booking",
            description="Scheduling salon appointments",
            phase=ConversationPhase.INFORMATION,
            trigger_intents=["book_appointment", "schedule_service"],
            trigger_keywords=["appointment", "haircut", "color", "highlights", "manicure"],
            response_templates=[
                "I'd be happy to book an appointment for you. What service are you looking for?",
                "Let's get you scheduled. Which stylist do you prefer?",
            ],
            follow_up_questions=[
                "What service would you like to book?",
                "Do you have a preferred stylist?",
                "What days and times work best for you?",
            ],
        ),
    ],
    common_services=[
        ServiceOffering(
            id="haircut",
            name="Haircut & Style",
            description="Professional haircut and styling",
            category="Hair Services",
            typical_duration="45-60 minutes",
            benefits=[
                "Expert cut tailored to you",
                "Styling tips for home maintenance",
                "Product recommendations",
            ],
        ),
        ServiceOffering(
            id="color",
            name="Hair Color Services",
            description="Professional hair coloring",
            category="Hair Services",
            typical_duration="2-3 hours",
            benefits=[
                "Custom color formulation",
                "Long-lasting results",
                "Healthy hair techniques",
            ],
        ),
    ],
    tone="friendly",
    formality_level=1,
    empathy_level=4,
    urgency_sensitivity=2,
)


# Insurance
INSURANCE_PROFILE = IndustryProfile(
    industry_type=IndustryType.INSURANCE,
    name="Insurance Services",
    description="Insurance agencies and brokers",
    typical_customer_segments=[
        "New policy seekers",
        "Existing policyholders",
        "Claims inquiries",
        "Policy change requests",
        "Quote comparisons",
    ],
    common_pain_points=[
        "Understanding coverage options",
        "Premium costs",
        "Claims process confusion",
        "Coverage gaps",
        "Policy complexity",
    ],
    value_propositions=[
        "Comprehensive coverage options",
        "Competitive rates",
        "Local, personal service",
        "Claims assistance",
        "Policy reviews",
    ],
    applicable_regulations=[
        RegulationType.STATE_INSURANCE,
        RegulationType.NAIC,
    ],
    conversation_patterns=[
        ConversationPattern(
            id="insurance_quote",
            name="Quote Request",
            description="Handling insurance quote requests",
            phase=ConversationPhase.QUALIFICATION,
            trigger_intents=["get_quote", "insurance_pricing"],
            trigger_keywords=["quote", "how much", "price", "cost", "rate"],
            response_templates=[
                "I'd be happy to get you a quote. What type of insurance are you looking for?",
                "Let me help you find the right coverage at the best price. What do you need to insure?",
            ],
            follow_up_questions=[
                "What type of coverage are you looking for?",
                "Are you currently insured?",
                "When is your current policy up for renewal?",
            ],
        ),
        ConversationPattern(
            id="insurance_claim",
            name="Claims Inquiry",
            description="Handling claims questions",
            phase=ConversationPhase.INFORMATION,
            trigger_intents=["file_claim", "claim_status"],
            trigger_keywords=["claim", "accident", "damage", "incident", "file"],
            response_templates=[
                "I'm sorry to hear about your situation. Let me help you with your claim.",
                "I can assist you with your claim. Can you tell me what happened?",
            ],
        ),
    ],
    behavior_guidelines=[
        BehaviorGuideline(
            id="insurance_disclosure",
            category="Compliance",
            guideline="Provide required disclosures for quotes",
            rationale="Insurance regulations require specific disclosures",
            priority=1,
        ),
    ],
    common_services=[
        ServiceOffering(
            id="policy_review",
            name="Free Policy Review",
            description="Complimentary coverage analysis",
            category="Consultation",
            typical_duration="30 minutes",
            benefits=[
                "Identify coverage gaps",
                "Find potential savings",
                "No obligation",
            ],
        ),
    ],
    tone="professional",
    formality_level=3,
    empathy_level=4,
    urgency_sensitivity=3,
)


# Plumbing Services
PLUMBING_PROFILE = IndustryProfile(
    industry_type=IndustryType.PLUMBING,
    name="Plumbing Services",
    description="Plumbing repair, installation, and maintenance",
    typical_customer_segments=[
        "Emergency leak callers",
        "Clogged drain customers",
        "Water heater issues",
        "Remodeling projects",
        "Maintenance requests",
    ],
    common_pain_points=[
        "Water damage emergencies",
        "No hot water",
        "Slow or clogged drains",
        "High water bills",
        "Old plumbing systems",
    ],
    value_propositions=[
        "24/7 emergency service",
        "Licensed, insured plumbers",
        "Upfront pricing",
        "Guaranteed work",
        "Fast response times",
    ],
    conversation_patterns=[
        ConversationPattern(
            id="plumbing_emergency",
            name="Plumbing Emergency",
            description="Handling plumbing emergencies",
            phase=ConversationPhase.DISCOVERY,
            trigger_intents=["plumbing_emergency", "water_leak"],
            trigger_keywords=[
                "leak", "flooding", "burst pipe", "no water",
                "overflowing", "sewage", "water everywhere",
            ],
            response_templates=[
                "I understand you have a plumbing emergency. Have you been able to shut off the water?",
                "Let me get a plumber out to you right away. First, can you locate and turn off your main water valve?",
            ],
            follow_up_questions=[
                "Where is the leak coming from?",
                "Can you turn off the water supply?",
                "Is there water damage occurring?",
            ],
        ),
    ],
    emergency_protocols=[
        EmergencyProtocol(
            id="major_leak",
            trigger_keywords=["flooding", "burst pipe", "water everywhere", "ceiling dripping"],
            severity="high",
            immediate_response="This sounds like a major leak.",
            acknowledgment_script="First, please try to locate and shut off your main water valve - "
                                 "usually near your water meter or where the main line enters your home. "
                                 "I'm dispatching a plumber to you right now.",
        ),
    ],
    common_services=[
        ServiceOffering(
            id="drain_cleaning",
            name="Drain Cleaning",
            description="Professional drain clearing service",
            category="Service",
            typical_duration="30-60 minutes",
            benefits=[
                "Restore full drainage",
                "Prevent future clogs",
                "Camera inspection available",
            ],
        ),
        ServiceOffering(
            id="water_heater",
            name="Water Heater Service",
            description="Water heater repair and installation",
            category="Service",
            benefits=[
                "Fast hot water restoration",
                "Energy-efficient options",
                "Tank and tankless options",
            ],
        ),
    ],
    tone="friendly",
    formality_level=2,
    empathy_level=4,
    urgency_sensitivity=5,
    after_hours_handling="We offer 24/7 emergency plumbing service. "
                        "Emergency calls may have after-hours rates.",
)


# Auto Dealership
AUTO_DEALERSHIP_PROFILE = IndustryProfile(
    industry_type=IndustryType.AUTO_DEALERSHIP,
    name="Auto Dealership",
    description="New and used car dealerships",
    typical_customer_segments=[
        "New car shoppers",
        "Used car seekers",
        "Trade-in customers",
        "Service department customers",
        "Parts inquiries",
    ],
    common_pain_points=[
        "Finding the right vehicle",
        "Negotiation anxiety",
        "Financing concerns",
        "Trade-in value questions",
        "Overwhelming options",
    ],
    value_propositions=[
        "Wide vehicle selection",
        "Competitive financing",
        "Fair trade-in values",
        "Professional sales team",
        "Full service department",
    ],
    conversation_patterns=[
        ConversationPattern(
            id="auto_sales_inquiry",
            name="Vehicle Inquiry",
            description="Handling vehicle inquiries",
            phase=ConversationPhase.QUALIFICATION,
            trigger_intents=["buy_car", "vehicle_inquiry"],
            trigger_keywords=["looking for", "interested in", "test drive", "available"],
            response_templates=[
                "Great choice! Let me tell you about that vehicle. Would you like to schedule a test drive?",
                "We have several options that might work for you. What are you looking for in a vehicle?",
            ],
            follow_up_questions=[
                "What type of vehicle are you looking for?",
                "What's your budget range?",
                "Will you be trading in a vehicle?",
                "What features are most important to you?",
            ],
        ),
        ConversationPattern(
            id="auto_trade_in",
            name="Trade-In Inquiry",
            description="Handling trade-in questions",
            phase=ConversationPhase.QUALIFICATION,
            trigger_intents=["trade_in", "vehicle_value"],
            trigger_keywords=["trade in", "trade-in", "worth", "value my car"],
            response_templates=[
                "We'd be happy to appraise your vehicle. Can you tell me about it?",
                "Getting a trade-in value is easy. What year, make, and model is your vehicle?",
            ],
        ),
    ],
    common_services=[
        ServiceOffering(
            id="test_drive",
            name="Test Drive",
            description="Schedule a test drive",
            category="Sales",
            typical_duration="30-45 minutes",
            benefits=[
                "Experience the vehicle firsthand",
                "Ask questions while driving",
                "No obligation",
            ],
        ),
        ServiceOffering(
            id="trade_appraisal",
            name="Trade-In Appraisal",
            description="Free vehicle appraisal",
            category="Sales",
            typical_duration="15-20 minutes",
            benefits=[
                "Know your vehicle's value",
                "No obligation",
                "Instant offer",
            ],
        ),
    ],
    tone="friendly",
    formality_level=2,
    empathy_level=3,
    urgency_sensitivity=3,
)


# Financial Services
FINANCIAL_SERVICES_PROFILE = IndustryProfile(
    industry_type=IndustryType.FINANCIAL_SERVICES,
    name="Financial Services",
    description="Financial advisors, wealth management, and banking",
    typical_customer_segments=[
        "Retirement planning clients",
        "Investment seekers",
        "High-net-worth individuals",
        "Business owners",
        "Young professionals",
    ],
    common_pain_points=[
        "Retirement uncertainty",
        "Investment confusion",
        "Tax optimization",
        "Estate planning complexity",
        "Market volatility concerns",
    ],
    value_propositions=[
        "Personalized financial planning",
        "Investment expertise",
        "Tax-efficient strategies",
        "Comprehensive wealth management",
        "Fiduciary responsibility",
    ],
    applicable_regulations=[
        RegulationType.FINRA,
        RegulationType.SEC,
        RegulationType.GLBA,
    ],
    conversation_patterns=[
        ConversationPattern(
            id="financial_consultation",
            name="Financial Consultation",
            description="Scheduling financial consultations",
            phase=ConversationPhase.QUALIFICATION,
            trigger_intents=["financial_advice", "wealth_management"],
            trigger_keywords=["invest", "retirement", "financial planning", "wealth"],
            response_templates=[
                "I'd be happy to schedule a consultation with one of our advisors. What financial goals are you focused on?",
                "Our advisors help clients with various financial needs. What brings you to us today?",
            ],
            follow_up_questions=[
                "What are your primary financial goals?",
                "Do you have a timeline in mind for these goals?",
                "Are you currently working with a financial advisor?",
            ],
        ),
    ],
    behavior_guidelines=[
        BehaviorGuideline(
            id="financial_no_guarantees",
            category="Compliance",
            guideline="Never guarantee investment returns",
            rationale="SEC/FINRA prohibit guaranteeing investment performance",
            priority=1,
            do_examples=[
                "Our advisors can help you understand the potential risks and returns.",
                "Past performance doesn't guarantee future results.",
            ],
            dont_examples=[
                "You'll definitely make money with this.",
                "This investment is guaranteed to grow.",
            ],
        ),
    ],
    common_services=[
        ServiceOffering(
            id="financial_review",
            name="Complimentary Financial Review",
            description="Free comprehensive financial review",
            category="Consultation",
            typical_duration="60 minutes",
            benefits=[
                "Understand your current financial position",
                "Identify opportunities and gaps",
                "Get personalized recommendations",
            ],
        ),
    ],
    tone="professional",
    formality_level=4,
    empathy_level=3,
    urgency_sensitivity=2,
)


# Profile registry
INDUSTRY_PROFILES: Dict[IndustryType, IndustryProfile] = {
    IndustryType.HEALTHCARE_GENERAL: HEALTHCARE_GENERAL_PROFILE,
    IndustryType.HEALTHCARE_DENTAL: DENTAL_PROFILE,
    IndustryType.LEGAL: LEGAL_PROFILE,
    IndustryType.REAL_ESTATE: REAL_ESTATE_PROFILE,
    IndustryType.HVAC: HVAC_PROFILE,
    IndustryType.RESTAURANT: RESTAURANT_PROFILE,
    IndustryType.SALON: SALON_PROFILE,
    IndustryType.INSURANCE: INSURANCE_PROFILE,
    IndustryType.PLUMBING: PLUMBING_PROFILE,
    IndustryType.AUTO_DEALERSHIP: AUTO_DEALERSHIP_PROFILE,
    IndustryType.FINANCIAL_SERVICES: FINANCIAL_SERVICES_PROFILE,
}


def get_industry_profile(industry_type: IndustryType) -> Optional[IndustryProfile]:
    """Get the profile for an industry type."""
    return INDUSTRY_PROFILES.get(industry_type)


def get_all_industry_types() -> List[IndustryType]:
    """Get all available industry types."""
    return list(INDUSTRY_PROFILES.keys())


def get_industry_by_name(name: str) -> Optional[IndustryProfile]:
    """Get industry profile by name (case-insensitive search)."""
    name_lower = name.lower()

    for profile in INDUSTRY_PROFILES.values():
        if name_lower in profile.name.lower():
            return profile

    return None


__all__ = [
    "INDUSTRY_PROFILES",
    "get_industry_profile",
    "get_all_industry_types",
    "get_industry_by_name",
    "HEALTHCARE_GENERAL_PROFILE",
    "DENTAL_PROFILE",
    "LEGAL_PROFILE",
    "REAL_ESTATE_PROFILE",
    "HVAC_PROFILE",
    "RESTAURANT_PROFILE",
    "SALON_PROFILE",
    "INSURANCE_PROFILE",
    "PLUMBING_PROFILE",
    "AUTO_DEALERSHIP_PROFILE",
    "FINANCIAL_SERVICES_PROFILE",
]
