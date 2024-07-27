import albumentations as A
import cv2
import os

def preprocessing(data_dir, horizonal_p=0.5, 
                  vertical_p=0.5, rotate_p=0.5, 
                  rotate_limit=45, shift_limit=0.0625, 
                  shift_rotate_limit=45, shift_scale_limit=0.1, shift_p=0.5,
                  interpolation=cv2.INTER_LINEAR, 
                  crop_w=32, crop_h=32, crop_p=0.5):
    
    for class_dir in ["normal", "tuberculos"]:
        class_path = os.path.join(data_dir, class_dir)
        img_path = os.listdir(class_path)
        
        for img_name in img_path:
            process_img = os.path.join(class_path, img_name)
            img_read = cv2.imread(process_img)
            
            if img_read is None:
                print(f"Warning: Image {process_img} could not be read.")
                continue

            '''img_read = img_read.astype('float32') / 255.0 # NORMALIZATION'''
            
            # Define the transformations
            transform = A.Compose([
                A.HorizontalFlip(p=horizonal_p),
                A.VerticalFlip(p=vertical_p),
                A.Rotate(limit=rotate_limit, interpolation=interpolation, p=rotate_p),
                A.ShiftScaleRotate(shift_limit=shift_limit, rotate_limit=shift_rotate_limit, scale_limit=shift_scale_limit, interpolation=interpolation, p=shift_p),
                A.RandomCrop(width=crop_w, height=crop_h, p=crop_p)
            ])

            # Apply the transformations
            augmented = transform(image=(img_read * 255).astype('uint8'))['image'] # Convert back to uint8

            # Save the augmented image
            augmented_path = os.path.join(class_path, "aug_" + img_name)
            cv2.imwrite(augmented_path, augmented)

    print("Data augmentation completed for all images in the directory.")
