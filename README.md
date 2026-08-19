# PassDNVAE

This repository contains the implementation of PassDNVAE: A Lightweight Password Dictionary Generation Model Using DenseNet-Based Variational AutoEncoder.

Paper:
https://doi.org/10.1109/JIOT.2026.3706412

# Requirements
Install the required packages: pip install -r requirements.txt

For Python 3 environments: pip3 install -r requirements.txt

# Model Training
python train.py --data_dir /path/to/training/data/directory --train_file training_dataset_filename --max_sequence_length maximum_password_length --save_dir /path/to/model/save/directory

Arguments:
--data_dir: Path to the training dataset directory
--train_file: Training dataset filename
--max_sequence_length: Maximum password length used for training
--save_dir: Directory where the trained model will be saved

# Password Generation
python generation.py --data_dir /path/to/training/data/directory --vocab_file vocabulary_filename.json --model_path /path/to/model/save/directory/passdnvae.pt --output_file /path/directory/PassDNVAE_rockyou.txt --num_samples number_of_passwords --max_sequence_length same_as_trained_model

Arguments:
--data_dir: Path to the dataset directory
--vocab_file: Vocabulary filename generated or used during training
--model_path: Path to the trained PassDNVAE model
--output_file: Path to the generated password output file
--num_samples: Number of passwords to generate
--max_sequence_length: Must be the same value used during model training

# Password Generation (Unique)
Use the --unique option to save only unique generated passwords

python generation.py --unique --data_dir /path/to/training/data/directory --vocab_file vocabulary_filename.json --model_path /path/to/model/save/directory/passdnvae.pt --output_file /path/directory/PassDNVAE_rockyou.txt --num_samples number_of_passwords --max_sequence_length same_as_trained_model
