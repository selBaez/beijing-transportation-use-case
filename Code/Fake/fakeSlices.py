import numpy as np
import shared

slices = np.loadtxt('fake.txt')
shared._featureSliceHeatmap(slices, 'Day', 'Non-commuter', '0000000')
