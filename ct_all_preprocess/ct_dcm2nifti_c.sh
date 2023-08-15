read -p "Enter the directory path: " directory
cd "$directory" || { echo "Error: Directory not found"; exit 1; }

for subfolder in */; do
  cd "$subfolder" || { echo "Error: Subfolder '$subfolder' not found"; continue; }
  echo "Processing: ${subfolder%?}"
  
  if [ -z "$(ls -A "$subfolder")" ]; then
    # If no sub1folder found, create 'dicom' folder and move .dcm files
    mkdir "./dicom"
    find . -type f -name "*.dcm" -exec mv {} "./dicom/" \;
    
    for sub1folder in */; do
      cd "$sub1folder" || { echo "Error: Subfolder '$sub1folder' not found"; continue; }
      dcm2niix -o ../../ -z y -f "%i_${sub1folder%?}" ./
      rm -f ../../*.json
      cd ..
    done
  fi
  
  cd ..
  for file in *"${subfolder%?}"*.nii.gz; do
    mv "$file" "${subfolder%?}${file#_}"
  done

done


