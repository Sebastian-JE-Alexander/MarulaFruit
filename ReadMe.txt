This project is a prototype for testing Marula Shell classification, current state and scope of the model only
handles two classes that we determine as either good or bad shells.
Future additions will seek to expand to 5 classes to further classify the shells.




Below is the sequence of commands to be run in the terminal to train and test this model.
Before running, ensure that all libraries and modules have been installed, use requirements.txt to pip install from

1) python segment_grid_photos.py --input_dir path/to/good_train_photos --val_input_dir path/to/good_validation_photos --class_name good

2) python train.py (will produce confusion matrix and model loss graph to evaluate how well the model trained before checking the next steps)

3) python infer.py dataset/validation/good/<some_filename>.png
   python infer.py dataset/validation/missing_open_eyelid/<some_filename>.png

4) python detect_and_classify.py path/to/any_photo.png --output path/to/annotated_result.png