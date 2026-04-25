import torch
try:
    checkpoint = torch.load('models/dl/echo_classifier.pth', map_location='cpu')
    print("Checkpoint keys:", checkpoint.keys())
    print(f"Epoch: {checkpoint.get('epoch')}")
    print(f"Val Acc: {checkpoint.get('val_acc')}")
    print(f"Num Frames: {checkpoint.get('num_frames')}")
    print(f"Frame Size: {checkpoint.get('frame_size')}")
    print(f"Classes: {checkpoint.get('classes')}")
    
    state_dict = checkpoint['model_state_dict']
    print("\nAll FC keys:")
    for key in state_dict.keys():
        if key.startswith('fc.'):
            print(f"  {key}: {state_dict[key].shape}")
except Exception as e:
    print(f"Error: {e}")
