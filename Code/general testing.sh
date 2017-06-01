################################### PREPARE ###################################

# SAMPLE BY CODE
python preparation.py --file 'sample data(50000)' \
                    --verbose 'False' \
                    --create_voc 'True' \
                    --plot 'False' \
                    --sampleBy 'Code'

# SAMPLE BY TRIP
python preparation.py --file 'sample data(50000)' \
                    --verbose 'False' \
                    --create_voc 'True' \
                    --plot 'False' \
                    --sampleBy 'Trip'

# FULL
python preparation.py --file 'sample data(50000)' \
                    --verbose 'False' \
                    --create_voc 'True' \
                    --plot 'False' \
                    --sampleBy 'False'

############################### SAMPLE  BY CODE ###############################

# FIRST CREATES VOCABULARY
python preparation.py --file 'de_cxl_cxtz20151115' \
                    --verbose 'False' \
                    --create_voc 'True' \
                    --plot 'False' \
                    --sampleBy 'Code'

# SUBSEQUENT WRITE IN EXISTING VOCABULARY
python preparation.py --file 'de_cxl_cxtz20151115' \
                    --verbose 'False' \
                    --create_voc 'False' \
                    --plot 'False' \
                    --sampleBy 'Code'

python preparation.py --file 'de_cxl_cxtz20151116' \
                    --verbose 'False' \
                    --create_voc 'False' \
                    --plot 'False' \
                    --sampleBy 'Code'

python preparation.py --file 'de_cxl_cxtz20151117' \
                    --verbose 'False' \
                    --create_voc 'False' \
                    --plot 'False' \
                    --sampleBy 'Code'

python preparation.py --file 'de_cxl_cxtz20151118' \
                    --verbose 'False' \
                    --create_voc 'False' \
                    --plot 'False' \
                    --sampleBy 'Code'

python preparation.py --file 'de_cxl_cxtz20151119' \
                    --verbose 'False' \
                    --create_voc 'False' \
                    --plot 'False' \
                    --sampleBy 'Code'

python preparation.py --file 'de_cxl_cxtz20151120' \
                    --verbose 'False' \
                    --create_voc 'False' \
                    --plot 'False' \
                    --sampleBy 'Code'

python preparation.py --file 'de_cxl_cxtz20151121' \
                    --verbose 'False' \
                    --create_voc 'False' \
                    --plot 'False' \
                    --sampleBy 'Code'

python preparation.py --file 'de_cxl_cxtz20151122' \
                    --verbose 'False' \
                    --create_voc 'False' \
                    --plot 'False' \
                    --sampleBy 'Code'

python preparation.py --file 'de_cxl_cxtz20151123' \
                    --verbose 'False' \
                    --create_voc 'False' \
                    --plot 'False' \
                    --sampleBy 'Code'

################################### LABELED ###################################

# FIRST CREATES CUBES
python preprocessing.py --file 'sample data(50000)_full' \
                    --verbose 'True' \
                    --plot 'True' \
                    --labeled 'True' \
                    --std 'True' \
                    --scriptMode 'short' \
                    --create_cubeDict 'True'

# SUBSEQUENT WRITE IN EXISTING CUBES
python preprocessing.py --file 'de_cxl_cxtz20151116' \
                    --verbose 'True' \
                    --plot 'True' \
                    --labeled 'True' \
                    --std 'True' \
                    --scriptMode 'short' \
                    --create_cubeDict 'False'


################################## UNLABELED ##################################
