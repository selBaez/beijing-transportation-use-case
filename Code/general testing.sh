################################### PREPARE ###################################

# FIRST CREATES VOCABULARY
python preparation.py --file 'de_cxl_cxtz20151115' \
                    --verbose 'False' \
                    --min_records 0 \
                    --create_voc 'True' \
                    --plot 'False' \
                    --scriptMode 'long' \
                    --minisample 'True'

# SUBSEQUENT WRITE IN EXISTING VOCABULARY
python preparation.py --file 'sample data(50000)' \
                    --verbose 'False' \
                    --min_records 0 \
                    --create_voc 'True' \
                    --plot 'False' \
                    --scriptMode 'long' \
                    --minisample 'True'

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
