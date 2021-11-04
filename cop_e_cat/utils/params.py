adm_static = ['icustay', 'hadm', 'Age', 'Weight', 'Gender', 'Ethnicity', 'AdmitUnit', 'AdmitType', 'h_exp', 'exp']

#
# adm_dynamic = {'Weight': ["Daily Weight", "Admission Weight (Kg)", "Admission Weight (lbs.)"],
#               'Height': ["Height", "Height (cm)"]}
### Get weights from inputs table instead


procs_dict_icu = {'IVent': [225792],  # 'Invasive ventilation'
				  'NIVent': [225794],  # "Non-invasive Ventilation"
				  'CRRT': [227290],  # 'Dialysis - CRRT'
				  'PeritD': [225805],  # 'Peritoneal Dialysis'
				  'Xray': [225459],  # 'Chest X-Ray'
				  'Culture': [225401],  # 'Blood Cultured'
				  'EKG': [225402]  # 'EKG'
                  }  # [224385, 227194] 'Intubation', 'Extubation'

vitals_dict_icu = {'HR': [220045],  # "Heart Rate"
				   'RR': [220210],  # "Respiratory Rate", "Respiratory Rate (Total)"
				   'Temp': [223761, 223762],  # "Temperature Fahrenheit", "Temperature Celsius"
				   'SpO2': [220277],  # "O2 saturation pulseoxymetry"
				   'NIBPM': [220181],  # "Non Invasive Blood Pressure mean"
				   'NIBPD': [220180],  # "Non Invasive Blood Pressure diastolic"
				   'NIBPS': [220179],  # "Non Invasive Blood Pressure systolic"
                   # 'ArtpH': [223830], "PH (Arterial)"
                   # 'ArtBE': [224828], #"Arterial Base Excess"
                   # 'ArtBPS': [220050], #"Arterial Blood Pressure systolic"
                   # 'ArtBPD': [220051], #"Arterial Blood Pressure diastolic"
                   # 'ArtBPM': [220052], #"Arterial Blood Pressure mean"
				   'Vt': [224685, 224686],  # "Tidal Volume (observed)", "Tidal Volume (spontaneous)"
				   'Vm': [224687],  # "Minute Volume"
				   'PIP': [224695],  # "Peak Insp. Pressure"
                   # 'PEEP': [224700], #"Total PEEP Level"
				   'ArtCO2P': [220235],  # "Arterial CO2 Pressure"
				   'ArtO2P': [220224],  # "Arterial O2 pressure"
                   # 'ArtO2sat': [220227], #"Arterial O2 Saturation"
				   'InspRatio': [226873],  # "Inspiratory Ratio"
				   'ExpRatio': [226871],  # "Expiratory Ratio"
				   'InspTime': [224738],  # "Inspiratory Time"
				   'ApneaInterval': [223876],  # "Apnea Interval"
				   'Urine': [227489, 226566, 226627, 226631]  # "GU Irrigant/Urine Volume Out", "Urine and GU Irrigant Out", "OR Urine", "PACU Urine",
                   }

vent_dict_icu = {'O2flow': [223834],  # "O2 Flow"
				 'FiO2': [223835],  # "Inspired O2 Fraction"
				 'PEEPset': [220339],  # "PEEP set"
				 'Vtset': [224684],  # "Tidal Volume (set)"
				 'RRset': [224688],  # "Respiratory Rate (Set)"
				 'CuffP': [224417],  # "Cuff Pressure"
				 'MeanAP': [224697],  # "Mean Airway Pressure"
				 'PlateauP': [224696],  # "Plateau Pressure"
				 'PSV': [224701],  # "PSV Level"
				 'Humidification': [227517],  # 'Humidification'
				 'Proning': [224093],  # 'Position'
				 'Angle': [224080]  # 'Head of Bed'
                 # 'NO': [224749] #'Nitric Oxide'
                 }  # Other:[223873, 223874, 223875] "Paw High" ,"Vti High" ,"Fspn High"

procs_dict = {  # 'RBCtransfusion': [229620, 221193] # 'Auto transfusion', 'Massive Transfusion'; ORIG:'Packed cell transfusion',
	# 'PlateletsT': ['Platelet transfusion'] # DNE in d_items
	# 'SerumT': ['Serum transfusion NEC'], # DNE in d_items
	'Dialysis': [226499, 225955, 225809, 225805, 225803, 225805, 225803, 225802]
	# "Hemodialysis Output", "Dialysis - SCUF", "Dialysis - CVVHDF", "Peritoneal Dialysis", "Dialysis - CVVHD", "Dialysis - CRRT"; ORIG: ['Hemodialysis','Extracorporeal circulat','Ven cath renal dialysis']
	# 'InitiateMV': ['Insert endotracheal tube'],# 'Insertion of Endotracheal Airway into Trachea, Via Opening',
	# 'Insertion of Endotracheal Airway into Trachea, Endo']
	# 'lt96MV': ['Cont invsv mech vent <96 consec hrs'], # 'Respiratory Ventilation, 24-96 Consecutive Hours', 'Respiratory Ventilation, Less than 24 Consecutive Hours']
	# 'mt96MV': ['Cont invsv mech vent >=96 consec hrs'] # 'Respiratory Ventilation, Greater than 96 Consecutive Hours'
	# 'NMV': ['Non-invasive mechanicl vent', 'Non-invasive mech vent']
}

labs_dict_icu = {'K': [227442, 227464],  # "Potassium (serum)", "Potassium (whole blood)"
				 'Ca': [225625],  # "Calcium non-ionized"
				 'Mg': [220635],  # "Magnesium"
				 'P': [225677],  # "Phosphorous"
				 'Na': [220645, 226534, 228389, 228390],
                 # "Sodium (serum)", "Sodium (whole blood)", "Sodium (serum) (soft)", "Sodium (whole blood) (soft)"
				 'Hgb': [220228],  # "Hemoglobin"
				 'Chloride': [220602, 226536],  # "Chloride (serum)", "Chloride (whole blood)"
				 'AnionGap': [227073],  # "Anion gap"
				 'Glucose': [220621, 225664, 226537],  # "Glucose (serum)", "Glucose finger stick (range 70-100)", "Glucose (whole blood)"
				 'BUN': [225624],  # "BUN"
				 'WBC': [220546],  # "WBC"
				 'Creatinine': [220615, 229761],  # "Creatinine (serum)", "Creatinine (whole blood)"
				 'Platelets': [227457],  # "Platelet Count"
				 'Alb': [227456],  # Albumin
                 # 'DD': [], #DNE in d_items
				 'Troponin': [227429],  # "Troponin-T"
				 'CRP': [227444],  # "C Reactive Protein (CRP)"
				 'Fibrinogen': [227468]  # "Fibrinogen"
                 }

# labs_dict_hosp = {'Creatinine': [50912, 52541, 51972, 52019], # Blood creatinine only
#              'WBC': [51300, 51753, 51754, 51301],
#              'Glucose': [50809, 50931, 52564], # Blood Glucose only, whole blood not included
#              'Hgb': [51638, 51222],
#              'Na': [52618],
#              'K': [50971, 52605], # Blood potassium only, whole blood not included
#              'Platelets': [51702, 51265], # "Platelet Count"
#              'Alb': [51542, 50862, 52017],
#              'BUN': [51840],
#              'DD': [50915, 52546, 51196],
#              'Troponin': [51002, 52637], # Troponin I OR # 51003  "Troponin T"
#              'CRP': [51650], # "High-Sensitivity CRP"
#              'Fibrinogen': [51621, 52111]
#             }

meds_dict_icu = {'Sedation': [225942, 221668, 222168, 221744, 225154, 229420, 221833, 221385],
                 # 'Fentanyl (Concentrate)','Midazolam (Versed)', 'Propofol', 'Fentanyl', 'Morphine Sulfate',"Dexmedetomidine (Precedex)", 'Hydromorphone (Dilaudid)', 'Lorazepam (Ativan)'
				 'Vasoactive': [221289, 221906, 229617, 221749, 229630, 229631, 229632, 221986, 221653, 221662, 222315],
				 'Paralytics': [221555, 222062],
				 'Fluids': [220954, 220958, 220959, 220960, 220961, 220962, 220963, 220964, 220965, 220966, 220979, 221001, 221003,
							221195, 221212, 221213, 225161, 225797, 226401, 226402, 227344],
				 'K-IV': [225166, 225834, 222139, 225925],
				 'Ca-IV': [228317, 229640],
				 'Ca-nonIV': [221456, 227525, 229618],
				 'Mg-IV': [227524],
				 'Mg-nonIV': [227523, 222011],
				 'P-IV': [225925, 225834],
				 'P-nonIV': [225890],
				 'Insulin': [223257, 223258, 223259, 223260, 223261, 223262, 229299, 229619],
				 'Anticoagulants': ["Heparin Sodium (Prophylaxis)", "Heparin Sodium", "Coumadin (Warfarin)"],
				 'Vasopressors': [22315, 221289, 221906, 229617],
				 'BetaBlockers': [225974, 221429],
				 'CaBlockers': [221347, 229654, 228339, 221468],
                 # 'Acetazolamide': [],
				 'LoopDiuretics': [221794, 228340],
                 # 'Thiazides': [],
                 # 'Spironolactone': [],
				 'Dextrose': [220949, 220950, 220951, 220952, 220963, 220964, 220965, 220966, 220967, 220968, 221000, 221002, 221014, 221017, 228140,
							  228141, 228142],
                 # 'Kayaxelate': [],
				 'TPNutrition': [225916, 225917],
				 'PNutrition': [227090, 222190, 225801, 225916, 225917, 225920, 225947, 225948, 225969],
				 'POnutrition': [221036, 225931, 226880, 227518, 226017, 226016, 226018, 226019, 226028, 226029, 226030, 226881],
				 'PackedRBC': [227070, 226368, 225168],
				 'Platelets': [225170]}

# meds_dict_hosp = {'K': ['Potassium Chloride', 'Potassium Chloride (Powder)',
#        'Potassium Phosphate', 'Potassium Chloride Replacement (Oncology)',
#        'Losartan Potassium',
#        'Potassium Chloride Replacement (Critical Care and Oncology)',
#        'Potassium Phosphate Replacement (Oncology)',
#        'Potassium Chloride Replacement (Critical Care and Oncology) ',
#        'Potassium Bicarbonate Eff', 'Potassium Citrate',
#        'Potassium Chloride (Klor-conM10)', 'Potassium ',
#        'Potassium Chloride ', 'Alum Potassium', 'Potassium Iodide',
#        'Potassium Chloride Oral Sol', 'Potassium', 'Potassium Chl',
#        'Potassium Chloride 10meq Capsules', 'Potassium Acetate',
#        'Potassium Chloride Oral Soln'],
#              'Ca': ['Calcium Gluconate', 'Rosuvastatin Calcium',
#        'Calcium Gluconate sliding scale (Critical Care-Ionized calcium)',
#        'Calcium Carbonate', 'Calcium Replacement (Oncology)',
#        'Calcium Acetate', 'Calcium Gluconate Replacement (Oncology)',
#        'Calcium Chloride', 'Calcium Carbonate Suspension',
#        'Calcium Gluconate Replacement (Critical Care and Oncology)',
#        'Calcium', 'Leucovorin Calcium',
#        'Calcium Gluconate sliding scale – Ionized calcium',
#        'Prenatal Vit-Calcium-FA-Fe', 'Calcium Gluc',
#        'Calcium Carbonate (Tums)', 'Calcium 500 + D (D3)',
#        'Calcium Gluconate ', 'Calcium Citrate', 'Calcium Gluco',
#        'Calcium Chloride (For Overdose Use Only)', 'Calcium Citrate + D',
#        'Calcium Citrate + Vit D', 'Mupirocin Calcium',
#        'Leucovorin Calcium (Chemo)', 'Calcium 600 + D(3)', 'Calcium ',
#        'Calcium Citrate +', 'Calcium 600 + D', 'Calcium with Vitamin D'],
#              'Mg': ['Magnesium Sulfate', 'Aluminum-Magnesium Hydrox.-Simethicone',
#        'Magnesium Sulfate Replacement (Oncology)',
#        'Magnesium Sulfate Replacement (Critical Care and Oncology)',
#        'Magnesium Oxide', 'Magnesium Citrate', 'Magnesium Sulfate (OB)',
#        'Magnesium Sulfate (L&D)', 'Magnesium', 'Magnesium ',
#        'Esomeprazole Magnesium', 'Choline Magnesium Trisalicylate'],
#              'P': ['Potassium Phosphate', 'Sodium Phosphate',
#        'Potassium Phosphate Replacement (Oncology)', 'Codeine Phosphate',
#        'Guaifenesin-CODEINE Phosphate', 'Dexamethasone Sod Phosphate',
#        'Oseltamivir Phosphate', 'Primaquine Phosphate',
#        'GuaiFENesin-CODEINE Phosphate', 'Fludarabine Phosphate',
#        'Disopyramide Phosphate', 'Prednisolone Sodium Phosphate']}

#     'Vasoactive': ['Norepinephrine', 'Epinephrine', 'Phenylephrine', 'Milrinone', 'Dobutamine', 'Dopamine', 'Vasopressin'],
#     'Paralytics': ['Cisatracurium', 'Pancuronium', 'Vecuronium'],
#     'Fluids': ['LR','Prismasate K4','Calcium Gluconate','NaCl 3% (Hypertonic Saline)','Gastric Meds',
#                'Bicarbonate Base','Folic Acid','ZGastric/TF Residual Intake','Cath Lab Intake','Dextrose 50%',
#               'Solution','D5NS','Dextrose 40%','Hetastarch (Hespan) 6%','Potassium Chloride','Prismasate K2',
#               'K Phos','Citrate','NaCl 0.45%','Sodium Bicarbonate 8.4%','Free Water','PO Intake','Dextran 40',
#               'Dextrose 20%','Dextrose 10%','Albumin 5%','NaCl 23.4%','Na Phos','D5 1/2NS','Dextrose 5%',
#               'ACD-A Citrate (1000ml)','Piggyback','Multivitamins','D5 1/4NS','NaCl 0.9%',
#               'Sodium Bicarbonate 8.4% (Amp)','D5LR','OR Crystalloid Intake','Hydrochloric Acid - HCL',
#               'Trisodium Citrate 0.4%','ACD-A Citrate (500ml)','KCL (Bolus)','Dextrose 30%','Thiamine',
#               'Vitamin K (Phytonadione)','Magnesium Sulfate','Sterile Water','Albumin 25%','Magnesium Sulfate (Bolus)'],
#     'K': ['Potassium Chloride', 'KCl (CRRT)', 'KCL (Bolus)'],
#     'Insulin': ['Insulin - Regular', 'Insulin - Humalog', 'Insulin - NPH',
#                 'Insulin - Glargine', 'Insulin - 70/30', 'Insulin - Humalog 75/25'],
#     'Anticoagulants': ["Heparin Sodium (Prophylaxis)", "Heparin Sodium", "Coumadin (Warfarin)"],
#     'BetaBlockers': ['Esmolol', 'Metoprolol'],
#     'CaBlockers': ['Amiodarone', 'Amiodarone 600/500', 'Diltiazem'],
#     #'Acetazolamide': [],
#     'LoopDiuretics': ['Furosemide (Lasix)', 'Furosemide (Lasix) 500/100'],
#     #'Thiazides': [],
#     #'Spironolactone': [],
#     'Dextrose': ['Dextrose 50%', 'Dextran 40', 'Dextrose PN', 'Dextrose 40%',
#                  'Dextrose 20%', 'Dextrose 30%', 'Dextrose 5%', 'Dextrose 10%'],
#     #'Kayaxelate': [],
#     #'TPNutrition': [],
#     'PNutrition': ['TPN w/ Lipids', 'TPN without Lipids', 'Peripheral Parenteral Nutrition',
#                    'Amino Acids', 'Lipids 20%'],
#     'POnutrition': ['Nutren Renal (Full)', 'Impact (Full)', 'Peptamen VHP (Full)',
#                       'Nutren Pulmonary (Full)', 'Ensure Plus (Full)',
#                       'Nutren Pulmonary (1/2)', 'Nutren Pulmonary (3/4)', 'Nepro (Full)',
#                       'Nepro (1/2)', 'Glucerna (Full)', 'Pulmocare (1/4)',
#                       'Pulmocare (Full)', 'Two Cal HN (Full)',
#                       'Peptamen Bariatric (Full)', 'Nutren 2.0 (3/4)', 'Beneprotein',
#                       'Impact with Fiber (1/4)', 'Nutren 2.0 (1/4)', 'Nutren 2.0 (1/2)',
#                       'Impact (1/4)', 'Impact (1/2)', 'Impact (3/4)', 'Nutren 2.0 (2/3)',
#                       'Nutren Renal (1/2)', 'Nutren Renal (3/4)', 'Nutren Renal (1/4)',
#                       'Impact with Fiber (1/2)', 'Impact with Fiber (3/4)',
#                       'Peptamen 1.5 (1/4)', 'Peptamen 1.5 (3/4)', 'Peptamen 1.5 (1/2)',
#                       'Fibersource HN (Full)', 'Fibersource HN (1/2)',
#                       'Fibersource HN (1/4)', 'Fibersource HN (3/4)',
#                       'Boost Glucose Control (Full)', 'NovaSource Renal (Full)',
#                       'NovaSource Renal (3/4)', 'NovaSource Renal (1/2)',
#                       'Replete with Fiber (1/4)', 'Replete (3/4)',
#                       'Replete with Fiber (3/4)', 'Replete with Fiber (1/2)',
#                       'Replete (1/2)', 'Replete (1/4)', 'Replete with Fiber (2/3)',
#                       'Vivonex (3/4)', 'Vivonex (1/2)', 'Nutren 2.0 (Full)',
#                       'Peptamen 1.5 (Full)', 'Replete with Fiber (Full)',
#                       'Vivonex (Full)', 'Replete (Full)', 'ProBalance (Full)',
#                       'Impact with Fiber (Full)', 'Ensure (Full)', 'Isosource 1.5 (1/2)',
#                       'Isosource 1.5 (1/4)', 'Isosource 1.5 (Full)',
#                       'Isosource 1.5 (3/4)', 'Isosource 1.5 (2/3)'],
#     'PackedRBC': ['Packed Red Blood Cells'],
#     'Platelets': ['Platelets']
#     #'Plasma': ['Fresh Frozen Plasma']
# }


# meds_dict_icu = {'Sedation': [225942, 221668, 222168, 221744, 225154, 229420, 221833, 221385] } #, #'Fentanyl (Concentrate)','Midazolam (Versed)', 'Propofol', 'Fentanyl', 'Morphine Sulfate',"Dexmedetomidine (Precedex)", 'Hydromorphone (Dilaudid)', 'Lorazepam (Ativan)'
#     'Vasoactive': ['Norepinephrine', 'Epinephrine', 'Phenylephrine', 'Milrinone', 'Dobutamine', 'Dopamine', 'Vasopressin'],
#     'Paralytics': ['Cisatracurium', 'Pancuronium', 'Vecuronium'],
#     'Fluids': ['LR','Prismasate K4','Calcium Gluconate','NaCl 3% (Hypertonic Saline)','Gastric Meds',
#                'Bicarbonate Base','Folic Acid','ZGastric/TF Residual Intake','Cath Lab Intake','Dextrose 50%',
#               'Solution','D5NS','Dextrose 40%','Hetastarch (Hespan) 6%','Potassium Chloride','Prismasate K2',
#               'K Phos','Citrate','NaCl 0.45%','Sodium Bicarbonate 8.4%','Free Water','PO Intake','Dextran 40',
#               'Dextrose 20%','Dextrose 10%','Albumin 5%','NaCl 23.4%','Na Phos','D5 1/2NS','Dextrose 5%',
#               'ACD-A Citrate (1000ml)','Piggyback','Multivitamins','D5 1/4NS','NaCl 0.9%',
#               'Sodium Bicarbonate 8.4% (Amp)','D5LR','OR Crystalloid Intake','Hydrochloric Acid - HCL',
#               'Trisodium Citrate 0.4%','ACD-A Citrate (500ml)','KCL (Bolus)','Dextrose 30%','Thiamine',
#               'Vitamin K (Phytonadione)','Magnesium Sulfate','Sterile Water','Albumin 25%','Magnesium Sulfate (Bolus)'],
#     'K': ['Potassium Chloride', 'KCl (CRRT)', 'KCL (Bolus)'],
#     'Insulin': ['Insulin - Regular', 'Insulin - Humalog', 'Insulin - NPH',
#                'Insulin - Glargine', 'Insulin - 70/30', 'Insulin - Humalog 75/25'],
#     'Anticoagulants': ["Heparin Sodium (Prophylaxis)", "Heparin Sodium", "Coumadin (Warfarin)"],
#     'BetaBlockers': ['Esmolol', 'Metoprolol'],
#     'CaBlockers': ['Amiodarone', 'Amiodarone 600/500', 'Diltiazem'],
#     #'Acetazolamide': [],
#     'LoopDiuretics': ['Furosemide (Lasix)', 'Furosemide (Lasix) 500/100'],
#     #'Thiazides': [],
#     #'Spironolactone': [],
#     'Dextrose': ['Dextrose 50%', 'Dextran 40', 'Dextrose PN', 'Dextrose 40%',
#                  'Dextrose 20%', 'Dextrose 30%', 'Dextrose 5%', 'Dextrose 10%'],
#     #'Kayaxelate': [],
#     #'TPNutrition': [],
#     'PNutrition': ['TPN w/ Lipids', 'TPN without Lipids', 'Peripheral Parenteral Nutrition',
#                    'Amino Acids', 'Lipids 20%'],
#     'POnutrition': ['Nutren Renal (Full)', 'Impact (Full)', 'Peptamen VHP (Full)',
#                       'Nutren Pulmonary (Full)', 'Ensure Plus (Full)',
#                       'Nutren Pulmonary (1/2)', 'Nutren Pulmonary (3/4)', 'Nepro (Full)',
#                       'Nepro (1/2)', 'Glucerna (Full)', 'Pulmocare (1/4)',
#                       'Pulmocare (Full)', 'Two Cal HN (Full)',
#                       'Peptamen Bariatric (Full)', 'Nutren 2.0 (3/4)', 'Beneprotein',
#                       'Impact with Fiber (1/4)', 'Nutren 2.0 (1/4)', 'Nutren 2.0 (1/2)',
#                       'Impact (1/4)', 'Impact (1/2)', 'Impact (3/4)', 'Nutren 2.0 (2/3)',
#                       'Nutren Renal (1/2)', 'Nutren Renal (3/4)', 'Nutren Renal (1/4)',
#                       'Impact with Fiber (1/2)', 'Impact with Fiber (3/4)',
#                       'Peptamen 1.5 (1/4)', 'Peptamen 1.5 (3/4)', 'Peptamen 1.5 (1/2)',
#                       'Fibersource HN (Full)', 'Fibersource HN (1/2)',
#                       'Fibersource HN (1/4)', 'Fibersource HN (3/4)',
#                       'Boost Glucose Control (Full)', 'NovaSource Renal (Full)',
#                       'NovaSource Renal (3/4)', 'NovaSource Renal (1/2)',
#                       'Replete with Fiber (1/4)', 'Replete (3/4)',
#                       'Replete with Fiber (3/4)', 'Replete with Fiber (1/2)',
#                       'Replete (1/2)', 'Replete (1/4)', 'Replete with Fiber (2/3)',
#                       'Vivonex (3/4)', 'Vivonex (1/2)', 'Nutren 2.0 (Full)',
#                       'Peptamen 1.5 (Full)', 'Replete with Fiber (Full)',
#                       'Vivonex (Full)', 'Replete (Full)', 'ProBalance (Full)',
#                       'Impact with Fiber (Full)', 'Ensure (Full)', 'Isosource 1.5 (1/2)',
#                       'Isosource 1.5 (1/4)', 'Isosource 1.5 (Full)',
#                       'Isosource 1.5 (3/4)', 'Isosource 1.5 (2/3)'],
#     'PackedRBC': ['Packed Red Blood Cells'],
#     'Platelets': ['Platelets']
#     #'Plasma': ['Fresh Frozen Plasma']
# }

# meds_dict_hosp = {'Sedation': ['Fentanyl (Concentrate)','Midazolam (Versed)', 'Propofol', 'Fentanyl', 'Morphine Sulfate',
#                           "Dexmedetomidine (Precedex)", 'Hydromorphone (Dilaudid)', 'Lorazepam (Ativan)'],
#              'Vasoactive': ['Norepinephrine', 'Epinephrine', 'Phenylephrine', 'Milrinone',
#                             'Dobutamine', 'Dopamine', 'Vasopressin'],
#              'Paralytics': ['Cisatracurium', 'Pancuronium', 'Vecuronium'],
#              'Fluids': ['LR','Prismasate K4','Calcium Gluconate','NaCl 3% (Hypertonic Saline)','Gastric Meds',
#                         'Bicarbonate Base','Folic Acid','ZGastric/TF Residual Intake','Cath Lab Intake','Dextrose 50%',
#                         'Solution','D5NS','Dextrose 40%','Hetastarch (Hespan) 6%','Potassium Chloride','Prismasate K2',
#                         'K Phos','Citrate','NaCl 0.45%','Sodium Bicarbonate 8.4%','Free Water','PO Intake','Dextran 40',
#                         'Dextrose 20%','Dextrose 10%','Albumin 5%','NaCl 23.4%','Na Phos','D5 1/2NS','Dextrose 5%',
#                         'ACD-A Citrate (1000ml)','Piggyback','Multivitamins','D5 1/4NS','NaCl 0.9%',
#                         'Sodium Bicarbonate 8.4% (Amp)','D5LR','OR Crystalloid Intake','Hydrochloric Acid - HCL',
#                         'Trisodium Citrate 0.4%','ACD-A Citrate (500ml)','KCL (Bolus)','Dextrose 30%','Thiamine',
#                         'Vitamin K (Phytonadione)','Magnesium Sulfate','Sterile Water','Albumin 25%','Magnesium Sulfate (Bolus)'],
#              'K': ['Potassium Chloride', 'KCl (CRRT)', 'KCL (Bolus)'],
#              'Insulin': ['Insulin - Regular', 'Insulin - Humalog', 'Insulin - NPH',
#                          'Insulin - Glargine', 'Insulin - 70/30', 'Insulin - Humalog 75/25'],
#              'Anticoagulants': ["Heparin Sodium (Prophylaxis)", "Heparin Sodium", "Coumadin (Warfarin)"],
#              'BetaBlockers': ['Esmolol', 'Metoprolol'],
#              'CaBlockers': ['Amiodarone', 'Amiodarone 600/500', 'Diltiazem'],
#              #'Acetazolamide': [],
#              'LoopDiuretics': ['Furosemide (Lasix)', 'Furosemide (Lasix) 500/100'],
#              #'Thiazides': [],
#              #'Spironolactone': [],
#              'Dextrose': ['Dextrose 50%', 'Dextran 40', 'Dextrose PN', 'Dextrose 40%',
#                           'Dextrose 20%', 'Dextrose 30%', 'Dextrose 5%', 'Dextrose 10%'],
#              #'Kayaxelate': [],
#              #'TPNutrition': [],
#              'PNutrition': ['TPN w/ Lipids', 'TPN without Lipids', 'Peripheral Parenteral Nutrition',
#                             'Amino Acids', 'Lipids 20%'],
#              'POnutrition': ['Nutren Renal (Full)', 'Impact (Full)', 'Peptamen VHP (Full)',
#                                'Nutren Pulmonary (Full)', 'Ensure Plus (Full)',
#                                'Nutren Pulmonary (1/2)', 'Nutren Pulmonary (3/4)', 'Nepro (Full)',
#                                'Nepro (1/2)', 'Glucerna (Full)', 'Pulmocare (1/4)',
#                                'Pulmocare (Full)', 'Two Cal HN (Full)',
#                                'Peptamen Bariatric (Full)', 'Nutren 2.0 (3/4)', 'Beneprotein',
#                                'Impact with Fiber (1/4)', 'Nutren 2.0 (1/4)', 'Nutren 2.0 (1/2)',
#                                'Impact (1/4)', 'Impact (1/2)', 'Impact (3/4)', 'Nutren 2.0 (2/3)',
#                                'Nutren Renal (1/2)', 'Nutren Renal (3/4)', 'Nutren Renal (1/4)',
#                                'Impact with Fiber (1/2)', 'Impact with Fiber (3/4)',
#                                'Peptamen 1.5 (1/4)', 'Peptamen 1.5 (3/4)', 'Peptamen 1.5 (1/2)',
#                                'Fibersource HN (Full)', 'Fibersource HN (1/2)',
#                                'Fibersource HN (1/4)', 'Fibersource HN (3/4)',
#                                'Boost Glucose Control (Full)', 'NovaSource Renal (Full)',
#                                'NovaSource Renal (3/4)', 'NovaSource Renal (1/2)',
#                                'Replete with Fiber (1/4)', 'Replete (3/4)',
#                                'Replete with Fiber (3/4)', 'Replete with Fiber (1/2)',
#                                'Replete (1/2)', 'Replete (1/4)', 'Replete with Fiber (2/3)',
#                                'Vivonex (3/4)', 'Vivonex (1/2)', 'Nutren 2.0 (Full)',
#                                'Peptamen 1.5 (Full)', 'Replete with Fiber (Full)',
#                                'Vivonex (Full)', 'Replete (Full)', 'ProBalance (Full)',
#                                'Impact with Fiber (Full)', 'Ensure (Full)', 'Isosource 1.5 (1/2)',
#                                'Isosource 1.5 (1/4)', 'Isosource 1.5 (Full)',
#                                'Isosource 1.5 (3/4)', 'Isosource 1.5 (2/3)'],
#              'PackedRBC': ['Packed Red Blood Cells'],
#              'Platelets': ['Platelets']
#              #'Plasma': ['Fresh Frozen Plasma']
#             }

comorbididites_dict = {
	'cad': ['Atherosclerotic heart disease of native coronary artery without angina pectoris',
			'Atherosclerotic heart disease of native coronary artery with unstable angina pectoris',
			'Atherosclerotic heart disease of native coronary artery with angina pectoris with documented spasm',
			'Atherosclerotic heart disease of native coronary artery with other forms of angina pectoris',
			'Atherosclerotic heart disease of native coronary artery with unspecified angina pectoris,'],
	'afib': ['Atrial fibrillation'],
	'chf': ['Rheumatic heart failure (congestive)', 'Congestive heart failure, unspecified'],
	'ckd': ['Chronic kidney disease, Stage I',
			'Chronic kidney disease, Stage II (mild)',
			'Chronic kidney disease, Stage III (moderate)',
			'Chronic kidney disease, Stage IV (severe)',
			'Chronic kidney disease, Stage V',
			'Chronic kidney disease, unspecified'],
	'esrd': ["End stage renal disease", "end stage renal disease"],
	"paralysis": ['Paralysis agitans',
				  'Paralysis, unspecified',
				  'Periodic paralysis'],
	'parathyroid': ['Primary hyperparathyroidism',
					'Secondary hyperparathyroidism, non-renal'],
	'rhabdo': ['Rhabdomyolysis'],
	'sarcoid': ["Sarcoidosis"],
	'sepsis': ['Sepsis',
			   'Severe sepsis', 'Other specified sepsis'
								'Sepsis, unspecified organism']
}

# # Top 10 poe_id matches from mimic_hosp.emar medication column
# meds_dict_hosp = {'Sedation': ["10001884-1618", "10001884-1798", "10002221-257", "10002221-383", "10002618-11",
#                           "10003502-552", "10003502-558", "10003502-559", "10003637-176", "10004606-111", # Fentanyl
#                           "10004720-34", "10005606-292", "10005606-333", "10007818-881", "10009129-70",
#                           "10010058-427", "10010058-481", "10010058-552", "10012261-718", "10016742-79", # Midazolam
#                           "10001884-1619", "10001884-1757", "10002618-12", "10003400-1671", "10004606-76",
#                           "10004606-92", "10004720-29", "10004764-75", "10005606-231", "10005606-281", # Propofol
#                           "10000032-217", "10000032-63", "10000285-20", "10001180-53", "10001401-76",
#                           "10001884-1465", "10001884-1516", "10001884-1713", "10001884-1896", "10001884-592",  # Morphine
#                           "10005606-364", "10005606-409", "10005817-1084", "10010867-154", "10010867-209",
#                           "10011365-467", "10011427-891", "10011849-1003", "10011849-1153", "10011849-274", # Dexmedetomidine (Precedex)
#                           "10000117-92", "10001180-63", "10001401-114", "10001401-117", "10001401-137",
#                           "10001401-163", "10001401-233", "10001401-239", "10001401-272", "10001401-313", # Hydromorphone (Dilaudid)
#                           "10000117-121", "10000117-96", "10001401-234", "10001401-246", "10001401-419",
#                           "10001401-424", "10001401-432", "10001401-470", "10001401-501", "10001401-582", # Lorazepam (Ativan)
#                         ],

#              'Vasoactive': ["10004720-46", "10005817-858", "10007818-309", "10007818-737",
#                             "10010058-425", "10010471-140", "10010471-99", "10011365-333",
#                             "10011427-302", "10011427-407", # Norepinephrine
#                             "10007818-311", "10014610-1819",, "10014610-1743", "10029291-44",
#                             "10033740-156", "10038992-172", "10047864-187", "10047864-698",
#                             "10057482-153", "10057482-299", # Epinephrine
#                             "10003400-1670", "10004457-385", "10004764-77", "10005817-912",
#                             "10005866-905", "10005866-920", "10005866-949", "10007818-312",
#                             "10007818-617", "10007818-860", # Phenylephrine
#                             "10033740-278", "10033740-331", "10033740-485", "10033740-496",
#                             "10033740-515", "10033740-523", "10037861-507", "10037861-508",
#                             "10037861-519", "10102878-442", #Milrinone
#                             "10022620-152", "10022620-157", "10023117-1264", "10023117-1288",
#                             "10023117-1358", "10023117-1360", "10023117-889", "10039708-435",
#                             "10055344-600", "10060531-1071", # Dobutamine
#                             "10001884-1634", "10003502-555", "10010058-452", "10013015-11",
#                             "10013015-47", "10023117-1369", "10023117-1408", "10023117-871",
#                             "10025038-135", "10025038-170", # Dopamine
#                             "10007818-310", "10010471-263", "10011427-962", "10011427-980",
#                             "10015931-883", "10021927-1372", "10023117-884", "10033740-158",
#                             "10039708-185", "10047727-288", # Vasopressin
#                             ],

#              'Paralytics': ["10035631-1573", "10049041-264", "10049041-307", "10057482-162",
#                             "10057482-300", "10097659-12", "10097659-134", "10097659-15",
#                             "10102878-77", "10156659-222", # Cisatracurium
#                             "12001921-489", "17509426-559", "19006032-56", # Pancuronium
#                             "10037861-222", "10070614-857", "10336392-36", "10336392-39",
#                             "10408681-71", "10408681-79", "10588180-798", "10702616-23",
#                             "10745745-1259", "10925424-122", # Vecuronium
#                             ],

#              'Fluids': ["10000032-127", "10000032-257", "10000032-271", "10002221-591", "10004606-106",
#                         "10004764-279", "10004764-79", "10005606-155", "10005817-916", "10005866-147", # Calcium Gluconate
#                         "10004457-400", "10067389-50", "10068026-149", "10069780-52", "10069992-111",
#                         "10177209-58", "10205923-228", "10205923-326", "10206125-480", "10206125-509", # NaCl
#                         # No match for Gastric Meds -- probably should be more specific with query
#                         "10006513-140", "10007818-101", "10011427-124", "10011427-246", "10013310-864",
#                         "10014610-1751", "10015860-1067", "10015860-1114", "10015860-1115", "10015860-812", # Sodium Bicarbonate Base
#                         "10002930-216", "10002930-239", "10002930-269", "10002930-296", "10002930-392",
#                         "10002930-426", "10003299-258", "10003299-697", "10005024-54", "10005348-220", # Folic Acid
#                         # No matches for 'ZGastric/TF Residual Intake'
#                         "10047864-1147", "10047864-726", "10123949-10378", "10123949-10386", "10123949-10430",
#                         "10123949-10431", "10123949-10435", "10123949-10438", "10123949-10458", "10123949-10459", # Cath
#                         "10000032-126", "10000032-259", "10000032-273", "10002930-399", "10005368-98",
#                         "10005749-314", "10005866-1127", "10006029-577", "10006457-381", "10006457-395", # Dextrose
#                         # No matches for D5NS
#                         "11431240-184", "11431240-246", # Hespan
#                         "10000032-36", "10000980-533", "10000980-697", "10000980-900", "10001401-428",
#                         "10001401-534", "10001401-592", "10001401-633", "10001884-1000", "10001884-1027", # Potassium Chloride
#                         # No matches for  D5LR
#                         "10001884-1618", "10001884-1798", "10002221-257", "10002221-383", "10002618-11",
#                         "10003400-418", "10003400-419", "10003400-846", "10003502-552", "10003502-558", # Citrate
#                         "10000032-108", "10000032-128", "10000032-200", "10000032-268", "10003400-1604",
#                         "10004764-117", "10004764-174", "10004764-175", "10005123-46", "10005606-234", # Albumin
#                         "10000980-519", "10000980-590", "10000980-877", "10000980-886", "10000980-889",
#                         "10000980-903", "10001401-429", "10001401-632", "10001884-1344", "10001884-633", # Magnesium Sulfate
#                         "16389788-26", # Water
#                         "10000117-116", "10000117-122", "10000980-466", "10000980-492", "10000980-580",
#                         "10000980-778", "10000980-852", "10001401-623", "10001401-719", "10001884-1125", #  Multivitamins
#                         "10001884-1536", "10002528-175", "10002930-217", "10002930-240", "10002930-271",
#                         "10002930-304", "10002930-327", "10002930-391", "10002930-398", "10003299-702", # Thiamine
#                         "10000044-5", "10001421-9", "10002160-9", "10003219-3", "10003619-7",
#                         "10007211-6", "10007338-12", "10007862-21", "10010000-13", "10018289-22", # Vitamin K
#                         "10001401-351", "10001401-381", "10001658-16", "10002013-762", "10002800-262",
#                         "10002800-263", "10002800-268", "10002800-269", "10004235-646", "10004587-20", # HCL
#                         "10004457-400", "10021395-141", "10021395-142", "10031575-179", "10031575-181",
#                         "10031575-210", "10069780-52", "10127517-719", "10127517-733", "10177209-58", # Dextran
#                         "10005001-164", "10005606-561", "10006692-52", "10007134-59", "10010231-115",
#                         "10010231-681", "10010231-688", "10010231-789", "10011365-408", "10011365-449", # Na Phos
#                         "10155871-75", "10155871-80", "10420013-193", "11579251-238", "11857739-81",
#                         "12673104-135", "12683989-338", "12896286-105", "13920430-736", "13998532-49", # Bolus
#                         "10004606-209", "10005866-1037", "10005866-202", "10005866-210", "10005866-252",
#                         "10005866-580", "10005866-590", "10005866-597", "10005866-875", "10005866-970", # Potassium Phosphate
#                         ],

#              'K': ["10000032-36", "10000980-533", "10000980-697", "10000980-900", "10001401-428",
#                    "10001401-534", "10001401-592", "10001401-633", "10001884-1000", "10001884-1027", # Potassium Chloride
#                   ],

#              'Insulin': ["10000032-125", "10000032-258", "10000032-272", "10000980-1013", "10000980-457",
#                          "10000980-515", "10000980-521", "10000980-528", "10000980-531", "10000980-593"],

#              'Anticoagulants': ["10000032-176", "10000032-238", "10000032-27", "10000032-82", "10000084-23",
#                                 "10000764-59", "10000980-451", "10000980-493", "10000980-549", "10000980-571", # Heparin
#                                 "16639746-465", # Coumadin,
#                                 ],

#              'BetaBlockers': ["10005817-887", "10005817-888", "10021927-1402", "10035631-1770", "10057482-83",
#                               "10067859-163", "10067859-211", "10069992-111", "10108433-73", "10108433-75", # Esmolol
#                               "10000764-27", "10000764-55", "10000764-73", "10002013-1021", "10002013-1087",
#                               "10002013-1140", "10002013-1218", "10002013-752", "10002013-823", "10002013-863", # Metoprolol
#                               ],

#              'CaBlockers': ["18365649-628", "16446532-3110", "17873397-436", "13158798-169", "10051074-851",
#                             "18365649-702", "16218892-13", "10938096-776", "18083807-532", "16597435-35", # Amiodarone
#                             "10001884-1001", "10001884-1002", "10001884-1049", "10001884-1079", "10001884-1115",
#                             "10001884-1191", "10001884-1240", "10001884-1298", "10001884-1362", "10001884-1417", # Diltiazem
#                             ],

#              'LoopDiuretics': ["10000032-103", "10000032-266", "10000032-30", "10000032-50", "10000980-1005",
#                                "10000980-435", "10000980-502", "10000980-523", "10000980-724", "10000980-729"], # Furosemide

#              'Dextrose': ["10000032-126", "10000032-259", "10000032-273", "10002930-399", "10005368-98",
#                           "10005749-314", "10005866-1127", "10006029-577", "10006457-381", "10006457-395"],

#              'PNutrition': ["11681918-779"],

#              'POnutrition': ["12200551-88", "12251059-131", "12679677-254", "14238144-379", "14287045-532",
#                              "17183590-129", "17524332-1297", "19237689-465", "19652719-104", # Fiber
#                              "11495932-831", "14061397-5557", "14067967-2952", # Renal
#                              "10116409-58", "10439110-5508", "10496352-1494", "10576313-401", "10711408-1038",
#                              "10711408-1276", "10711408-882", "10711408-916", "10843264-262", "11293220-42", # Pulmonary
#                              ],
#              'PackedRBC': [], # no matches
#              'Platelets': [], # no matches
#             }


