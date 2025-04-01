import torch

file_path = '/home/tasi2425111/for_hpc/baru/ti_mf/10_new_train/best_model.pth'

model_data = torch.load(file_path, map_location=torch.device('cpu'))

print("Isi file .pth:")
if isinstance(model_data, dict):
    for key, value in model_data.items():
        if key == 'epoch' :
            print(f"\nKey: {key}")
            print(f"Value: {value}")
else:
    print(model_data)
