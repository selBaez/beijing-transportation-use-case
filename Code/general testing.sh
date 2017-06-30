# ################################### PREPARE ###################################
#
# # SAMPLE BY CODE
# python preparation.py --file 'sample data(50000)' \
#                     --verbose 'False' \
#                     --create_voc 'True' \
#                     --plot 'False' \
#                     --sampleBy 'Code'
#
# # SAMPLE BY TRIP
# python preparation.py --file 'sample data(50000)' \
#                     --verbose 'False' \
#                     --create_voc 'True' \
#                     --plot 'False' \
#                     --sampleBy 'Trip'
#
# FULL
python preparation.py --file 'sample data(50000)-parallel' \
                    --verbose 'False' \
                    --create_voc 'True' \
                    --plot 'False' \
                    --sampleBy 'False'

# ############################### SAMPLE  BY CODE ###############################
#
# # FIRST CREATES VOCABULARY
# python preparation.py --file 'de_cxl_cxtz20151115' \
#                     --verbose 'True' \
#                     --create_voc 'True' \
#                     --plot 'True' \
#                     --sampleBy 'Code'
#
# # SUBSEQUENT WRITE IN EXISTING VOCABULARY
# python preparation.py --file 'de_cxl_cxtz20151115' \
#                     --verbose 'True' \
#                     --create_voc 'False' \
#                     --plot 'True' \
#                     --sampleBy 'Trip'
#
# python preparation.py --file 'de_cxl_cxtz20151116' \
#                     --verbose 'True' \
#                     --create_voc 'False' \
#                     --plot 'True' \
#                     --sampleBy 'Trip'
#
# python preparation.py --file 'de_cxl_cxtz20151117' \
#                     --verbose 'True' \
#                     --create_voc 'False' \
#                     --plot 'True' \
#                     --sampleBy 'Trip'
#
# python preparation.py --file 'de_cxl_cxtz20151118' \
#                     --verbose 'True' \
#                     --create_voc 'False' \
#                     --plot 'True' \
#                     --sampleBy 'Trip'
#
# python preparation.py --file 'de_cxl_cxtz20151119-1' \
#                     --verbose 'True' \
#                     --create_voc 'False' \
#                     --plot 'True' \
#                     --sampleBy 'Trips'
#
# python preparation.py --file 'de_cxl_cxtz20151119-2' \
#                     --verbose 'True' \
#                     --create_voc 'False' \
#                     --plot 'True' \
#                     --sampleBy 'Trips'
#
# python preparation.py --file 'de_cxl_cxtz20151120' \
#                     --verbose 'True' \
#                     --create_voc 'False' \
#                     --plot 'True' \
#                     --sampleBy 'Code'
#
# python preparation.py --file 'de_cxl_cxtz20151121' \
#                     --verbose 'True' \
#                     --create_voc 'False' \
#                     --plot 'True' \
#                     --sampleBy 'Trip'
#
# python preparation.py --file 'de_cxl_cxtz20151122' \
#                     --verbose 'True' \
#                     --create_voc 'False' \
#                     --plot 'True' \
#                     --sampleBy 'Trip'
#
# python preparation.py --file 'de_cxl_cxtz20151123' \
#                     --verbose 'True' \
#                     --create_voc 'False' \
#                     --plot 'True' \
#                     --sampleBy 'Code'

# #################################  PREPROCESS #################################
#
# # FIRST CREATES CUBES
# python preprocessing.py --file 'sample data(50000)-full' \
#                     --verbose 'False' \
#                     --plot 'True' \
#                     --labeled 'True' \
#                     --std 'True' \
#                     --scriptMode 'long' \
#                     --create_cubeDict 'True'
#
# ################################### LABELED ###################################
#
# # FIRST CREATES CUBES
# python preprocessing.py --file 'de_cxl_cxtz20151115- sample codes' \
#                     --verbose 'True' \
#                     --plot 'False' \
#                     --labeled 'True' \
#                     --std 'True' \
#                     --scriptMode 'long' \
#                     --create_cubeDict 'True'
#
# # SUBSEQUENT WRITE IN EXISTING CUBES
# python preprocessing.py --file 'de_cxl_cxtz20151118- sample codes' \
#                     --verbose 'True' \
#                     --plot 'False' \
#                     --labeled 'True' \
#                     --std 'True' \
#                     --scriptMode 'long' \
#                     --create_cubeDict 'False'
#
# python preprocessing.py --file 'de_cxl_cxtz20151119-1- sample codes' \
#                     --verbose 'True' \
#                     --plot 'False' \
#                     --labeled 'True' \
#                     --std 'True' \
#                     --scriptMode 'long' \
#                     --create_cubeDict 'False'
#
# python preprocessing.py --file 'de_cxl_cxtz20151116- sample trips' \
#                     --verbose 'True' \
#                     --plot 'False' \
#                     --labeled 'True' \
#                     --std 'True' \
#                     --scriptMode 'long' \
#                     --create_cubeDict 'False'
#
# python preprocessing.py --file 'de_cxl_cxtz20151117- sample trips' \
#                     --verbose 'True' \
#                     --plot 'False' \
#                     --labeled 'True' \
#                     --std 'True' \
#                     --scriptMode 'long' \
#                     --create_cubeDict 'False'
#
# python preprocessing.py --file 'de_cxl_cxtz20151121- sample trips' \
#                     --verbose 'True' \
#                     --plot 'False' \
#                     --labeled 'True' \
#                     --std 'True' \
#                     --scriptMode 'long' \
#                     --create_cubeDict 'False'
#
# # PLOT ONLY IN LAST
#
# python preprocessing.py --file 'de_cxl_cxtz20151122- sample trips' \
#                     --verbose 'True' \
#                     --plot 'True' \
#                     --labeled 'True' \
#                     --std 'True' \
#                     --scriptMode 'long' \
#                     --create_cubeDict 'False'
#
# ################################# SAMPLE TRIPS #################################
#
# python preprocessing.py --file 'de_cxl_cxtz20151115- sample trips' \
#                     --verbose 'False' \
#                     --plot 'False' \
#                     --labeled 'False' \
#                     --std 'True' \
#                     --scriptMode 'long' \
#                     --create_cubeDict 'True'
#
# python preprocessing.py --file 'de_cxl_cxtz20151116- sample trips' \
#                     --verbose 'False' \
#                     --plot 'False' \
#                     --labeled 'False' \
#                     --std 'True' \
#                     --scriptMode 'long' \
#                     --create_cubeDict 'False'
#
# python preprocessing.py --file 'de_cxl_cxtz20151117- sample trips' \
#                     --verbose 'False' \
#                     --plot 'False' \
#                     --labeled 'False' \
#                     --std 'True' \
#                     --scriptMode 'long' \
#                     --create_cubeDict 'False'
#
# python preprocessing.py --file 'de_cxl_cxtz20151118- sample trips' \
#                     --verbose 'False' \
#                     --plot 'False' \
#                     --labeled 'False' \
#                     --std 'True' \
#                     --scriptMode 'long' \
#                     --create_cubeDict 'False'
#
# python preprocessing.py --file 'de_cxl_cxtz20151121- sample trips' \
#                     --verbose 'False' \
#                     --plot 'False' \
#                     --labeled 'False' \
#                     --std 'True' \
#                     --scriptMode 'long' \
#                     --create_cubeDict 'False'
#
# python preprocessing.py --file 'de_cxl_cxtz20151122- sample trips' \
#                     --verbose 'False' \
#                     --plot 'False' \
#                     --labeled 'False' \
#                     --std 'True' \
#                     --scriptMode 'long' \
#                     --create_cubeDict 'False'

# ############################## FEATURE SELECTION ##############################
python featureSelection.py --file 'original' \
                    --plot 'True'

python featureSelection.py --file 'original' \
                    --plot 'True'

# ################################# SPLIT FILES #################################
# tail -n +2 de_cxl_cxtz20151119.csv | split -l 5000000 - split_
# for file in split_*
# do
#     head -n 1 de_cxl_cxtz20151119.csv > tmp_file
#     cat $file >> tmp_file
#     mv -f tmp_file $file
# done
