# ################################# SPLIT FILES #################################
# tail -n +2 de_cxl_cxtz20151119.csv | split -l 5000000 - split_
# for file in split_*
# do
#     head -n 1 de_cxl_cxtz20151119.csv > tmp_file
#     cat $file >> tmp_file
#     mv -f tmp_file $file
# done

# ################################### PREPARE ###################################
# CREATE VOCABULARY DURING FIRST RUN AND WRITE ON IT AFTERWARDS
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
# # FULL
# python preparation.py --file 'sample data(50000)' \
#                     --verbose 'False' \
#                     --create_voc 'True' \
#                     --plot 'False' \
#                     --sampleBy 'False'

# #################################  PREPROCESS #################################
# CREATE CUBES DURING FIRST RUN AND WRITE ON THEM AFTERWARDS.
# PLOT ONLY ON LAST RUN
#
# LABELED
python preprocessing.py --file 'sample data(50000)-full' \
                    --verbose 'False' \
                    --plot 'False' \
                    --labeled 'True' \
                    --scriptMode 'long' \
                    --create_cubeDict 'True'

# ALL
python preprocessing.py --file 'sample data(50000)-full' \
                    --verbose 'False' \
                    --plot 'False' \
                    --labeled 'False' \
                    --scriptMode 'long' \
                    --create_cubeDict 'True'

# ############################## FEATURE SELECTION ##############################
python featureSelection.py --plot 'True'
