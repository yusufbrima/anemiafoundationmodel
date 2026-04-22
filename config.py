from pathlib import Path

DATA_ROOT_DIR = Path("/content/drive/MyDrive/ResearchProjects/DHS/DHS_Databases")

RESULTS_DIR = Path("/content/drive/MyDrive/ResearchProjects/DHS/Scripts/results")
FIGURES_DIR = Path("/content/drive/MyDrive/ResearchProjects/DHS/Scripts/figures")

# SRC: https://www.researchsquare.com/article/rs-7918744/v1
VARIABLES = [
    "hw1",    # Age (months)
    "b4",     # Sex of child
    "h31",    # Cough last 2 weeks
    'hw70',   #'height/age standard deviation (new who)'
    "v012",   # Mother age
    'v212',       # Mother's age at first birth
    'hw72',  #'weight/height standard deviation (new who)'
    # 'v457a',      # Mother's Hemoglobin (Top predictor)
    "v190",   # Wealth index
    'v040',       # Altitude (adjusts the biological threshold)
    
    # Health & Infection Proxies
    'h22',        # Fever in last 2 weeks (inflammation proxy)
    'h11',        # Diarrhea in last 2 weeks (nutrient loss proxy)
    'h43',        # Child took deworming medication
    'h34',       # Vitamin A dose in last 6 months

    # Socio-Economic & Environmental
    'v191',       # Wealth Score (continuous factor score)
    'v113',       # Source of drinking water (parasite risk)
    'v116',       # Type of toilet facility (sanitation)
    'v106',       # Mother's education (behavioral proxy)
    'v025',       # Type of residence (Urban/Rural)

    "v201",   # Number of children (parity)
    "m13",  # approximate / must verify
    'v463z',     # Mother smokes (environmental factor)
    "v151",    # Sex of household head
    'hw57' # Target Predictor
]


CAL_COLS = [
    "h22", "h11", "h43", "h34",'v190',
    "v113", "v116", "v106",'h31',
    "v025", "b4", "v151", "v463z"
]

MAX_ROWS = 24000
