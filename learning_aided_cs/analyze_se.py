import numpy as np

model = 'SuperOSEN2_q3_0.25'

precision = []
specificity = []
sensitivity = []
f1_scores = []
f2_scores = []
accuracy = []

for i in range(0, 5):
    #if i == 0:
    #    file_name = model + '.txt'  
    #else:
    file_name = model + '_' + str(i + 1) + '.txt'

    with open(file_name, 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]

    lines = lines[-8::]

    for j in range(0, len(lines)):
        if 'Obtained Test Precision:  ' in lines[j]:
            for jj in lines[j].split():
                try:
                    result = float(jj)
                    break
                except:
                    continue
            precision.append(result)
        elif 'Obtained Test Specificity:  ' in lines[j]:
            for jj in lines[j].split():
                try:
                    result = float(jj)
                    break
                except:
                    continue
            specificity.append(result)
        elif 'Obtained Test Recall:  ' in lines[j]:
            for jj in lines[j].split():
                try:
                    result = float(jj)
                    break
                except:
                    continue
            sensitivity.append(result)
        elif 'Obtained Test F1 score:  ' in lines[j]:
            for jj in lines[j].split():
                try:
                    result = float(jj)
                    break
                except:
                    continue
            f1_scores.append(result)

        elif 'Obtained Test F2 score:  ' in lines[j]:
            for jj in lines[j].split():
                try:
                    result = float(jj)
                    break
                except:
                    continue
            f2_scores.append(result)

        elif 'Obtained Test CE:  ' in lines[j]:
            for jj in lines[j].split():
                try:
                    result = float(jj)
                    break
                except:
                    continue
            accuracy.append(1 - result)

print('Number of runs: ', len(f1_scores))
print('Precision')
print(np.mean(precision))
print(np.std(precision))
print('Specificity')
print(np.mean(specificity))
print(np.std(specificity))
print('Sensitivity')
print(np.mean(sensitivity))
print(np.std(sensitivity))
print('F1-Score')
print(np.mean(f1_scores))
print(np.std(f1_scores))
print('F2-score')
print(np.mean(f2_scores))
print(np.std(f2_scores))
print('Accuracy')
print(np.mean(accuracy))
print(np.std(accuracy))